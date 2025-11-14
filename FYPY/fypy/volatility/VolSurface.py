import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, Sequence, Optional

from scipy.interpolate import interp1d

from fypy.market.MarketSurface import MarketSurface
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.volatility.StrikeConverter import StrikeConverter, LogRelativeStrikeConverter, NoStrikeConverter
from fypy.volatility.VolSlice import VolSlice, ModelVolSlice, InterpolatedVolSlice
from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator


class VolSurface(ABC):
    @abstractmethod
    def vol(self, K: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def total_var(self, K: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return T * np.square(self.vol(K=K, T=T))


class ModelVolSurface(VolSurface):
    def __init__(self,
                 strike_converter: StrikeConverter = NoStrikeConverter()):
        self._strike_converter = strike_converter

    def vol(self, K: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return self.model_vol(x=self._strike_converter.convert(K=K, T=T), T=T)

    @abstractmethod
    def model_vol(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def model_var(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return T * np.square(self.model_vol(x=x, T=T))


class ConstantVolSurface(ModelVolSurface):
    def __init__(self, vol: float):
        super().__init__()
        self._vol = vol

    def model_vol(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return self.vol(K=x, T=T)

    def vol(self, K: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return np.full(K.shape, self._vol) if isinstance(K, np.ndarray) else self._vol


class ModelVolSurfaceSlices(ModelVolSurface):
    def __init__(self,
                 slices: Dict[float, ModelVolSlice],
                 strike_converter: Optional[StrikeConverter] = None):
        self._slices = slices
        self._ttms = np.array(list(slices.keys()))
        self._ttms.sort()
        converter = strike_converter or slices[self._ttms[0]].strike_converter
        super().__init__(strike_converter=converter)

    @staticmethod
    def from_market_surface(market_surface: MarketSurface,
                            strike_converter: Optional[StrikeConverter] = None
                            ) -> 'ModelVolSurfaceSlices':
        if strike_converter is None:
            strike_converter = LogRelativeStrikeConverter(market_surface.forward_curve)
        elif isinstance(strike_converter, LogRelativeStrikeConverter) and not strike_converter.has_forward_curve:
            strike_converter = strike_converter.set_forward_curve(market_surface.forward_curve)

        slices = {}
        for ttm, market_slice in market_surface.slices.items():
            slices[ttm] = InterpolatedVolSlice.from_market_slice(market_slice=market_slice,
                                                                 strike_converter=strike_converter)

        return ModelVolSurfaceSlices(slices=slices, strike_converter=strike_converter)

    @staticmethod
    def from_pricer(pricer: StrikesPricer,
                    ttms: Sequence[float],
                    ivc: ImpliedVolCalculator,
                    forward_curve: Optional[ForwardCurve] = None,
                    num_strikes: int = 100,
                    std_devs: float = 4,
                    sigma: float = 0.4,
                    strike_converter: Optional[StrikeConverter] = None,
                    **kwargs
                    ) -> 'ModelVolSurfaceSlices':
        if forward_curve is None:
            forward_curve = kwargs.pop('foward_curve', None)
        if forward_curve is None:
            raise ValueError("forward_curve must be provided (legacy name 'foward_curve' is also accepted).")
        if strike_converter is None:
            strike_converter = LogRelativeStrikeConverter(forward_curve)
        elif isinstance(strike_converter, LogRelativeStrikeConverter) and not strike_converter.has_forward_curve:
            strike_converter = strike_converter.set_forward_curve(forward_curve)

        slices = {}
        is_calls = np.ones(num_strikes, dtype=bool)
        is_calls[:int(num_strikes / 2)] = False
        for ttm in ttms:
            F = forward_curve(ttm)
            dev = np.exp(sigma * std_devs * np.sqrt(ttm))
            left = F / dev
            right = F * dev
            Ks = np.linspace(left, right, num_strikes)
            prices = pricer.price_strikes(T=ttm, K=Ks, is_calls=is_calls)
            vols = ivc.imply_vols(strikes=Ks, prices=prices, is_calls=is_calls, ttm=ttm)

            slices[ttm] = InterpolatedVolSlice.from_market_strikes(ttm=ttm, fwd=F, vols=vols, K=Ks,
                                                                   strike_converter=strike_converter)

        return ModelVolSurfaceSlices(slices=slices, strike_converter=strike_converter)

    def model_vol(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        T = max(1e-07, T)
        return np.sqrt(self.model_var(x, T) / T)

    @property
    def first_ttm(self) -> float:
        return self._ttms[0]

    @property
    def last_ttm(self) -> float:
        return self._ttms[-1]

    def model_var(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        if T <= self.first_ttm:
            return self._extrapolate_var_near(x=x, T=T)
        if T >= self.last_ttm:
            return self._extrapolate_var_far(x=x, T=T)
        return self._interpolate_var(x=x, T=T)

    def _extrapolate_var_near(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return self._slices[self.first_ttm].model_var(x=x) / self.first_ttm * T

    def _extrapolate_var_far(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        return self._slices[self.last_ttm].model_var(x=x) / self.last_ttm * T

    def _interpolate_var(self, x: Union[float, np.ndarray], T: float) -> Union[float, np.ndarray]:
        right_index = np.searchsorted(self._ttms, T, side='right')
        left_index = right_index - 1

        ttm_left = self._ttms[left_index]
        ttm_right = self._ttms[right_index]

        var_left = self._slices[ttm_left].model_var(x)

        mult = (T - ttm_left) / (ttm_right - ttm_left)
        return var_left + (self._slices[ttm_right].model_var(x) - var_left) * mult
