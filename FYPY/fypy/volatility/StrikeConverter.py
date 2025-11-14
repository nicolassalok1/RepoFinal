from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np

from fypy.termstructures.ForwardCurve import ForwardCurve


class StrikeConverter(ABC):
    @abstractmethod
    def convert(self,
                K: Union[float, np.ndarray],
                T: float,
                F: Optional[float] = None) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def invert(self,
               x: Union[float, np.ndarray],
               T: float,
               F: Optional[float] = None) -> Union[float, np.ndarray]:
        raise NotImplementedError


class NoStrikeConverter(StrikeConverter):
    def convert(self,
                K: Union[float, np.ndarray],
                T: float,
                F: Optional[float] = None) -> Union[float, np.ndarray]:
        return K

    def invert(self,
               x: Union[float, np.ndarray],
               T: float,
               F: Optional[float] = None) -> Union[float, np.ndarray]:
        return x


class LogRelativeStrikeConverter(StrikeConverter):
    def __init__(self, fwd_curve: Optional[ForwardCurve] = None):
        self._fwd_curve = fwd_curve

    def set_forward_curve(self, fwd_curve: ForwardCurve):
        """Attach or replace the forward curve used during conversion."""
        self._fwd_curve = fwd_curve
        return self

    @property
    def has_forward_curve(self) -> bool:
        return self._fwd_curve is not None

    def convert(self,
                K: Union[float, np.ndarray],
                T: float,
                F: Optional[float] = None) -> Union[float, np.ndarray]:
        if F is None:
            if self._fwd_curve is None:
                raise ValueError("Forward curve is not set on LogRelativeStrikeConverter. "
                                 "Provide F explicitly or call set_forward_curve.")
            F = self._fwd_curve(T)

        return np.log(K / F)

    def invert(self,
               x: Union[float, np.ndarray],
               T: float,
               F: Optional[float] = None) -> Union[float, np.ndarray]:
        if F is None:
            if self._fwd_curve is None:
                raise ValueError("Forward curve is not set on LogRelativeStrikeConverter. "
                                 "Provide F explicitly or call set_forward_curve.")
            F = self._fwd_curve(T)

        return F * np.exp(x)
