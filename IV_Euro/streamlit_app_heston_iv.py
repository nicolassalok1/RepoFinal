#!/usr/bin/env python3
"""Streamlit app to visualize Heston-model implied vol surfaces for calls and puts."""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

MAX_LOOKAHEAD_YEARS = 3
MIN_MATURITY = 0.1
PHI_MAX = 200.0
PHI_STEPS = 2000  # will be adjusted to be odd for Simpson's rule

st.set_page_config(page_title="Heston IV Surfaces", layout="wide")
st.title("Heston-Implied Volatility Surfaces (Calls & Puts)")
st.write(
    "Download option data via `yfinance`, price them with the Heston model using user-defined parameters, "
    "convert to Black-Scholes implied volatilities, and visualize the resulting call and put surfaces."
)


def fetch_spot(ticker: yf.Ticker) -> float:
    history = ticker.history(period="1d")
    if history.empty:
        raise RuntimeError("Unable to retrieve spot price.")
    return float(history["Close"].iloc[-1])


def _select_monthly_expirations(
    expirations: List[str], years_ahead: float
) -> List[Tuple[dt.datetime, str]]:
    today = dt.datetime.utcnow().date()
    limit_date = today + dt.timedelta(days=365 * years_ahead)
    monthly: Dict[Tuple[int, int], Tuple[dt.date, str]] = {}
    for exp in expirations:
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        if not (today < exp_date <= limit_date):
            continue
        key = (exp_date.year, exp_date.month)
        if key not in monthly or exp_date < monthly[key][0]:
            monthly[key] = (exp_date, exp)
    selected = sorted(monthly.values(), key=lambda item: item[0])
    return [(dt.datetime.combine(item[0], dt.time()), item[1]) for item in selected]


@st.cache_data(show_spinner=False, ttl=1800)
def download_option_data(symbol: str, years_ahead: float, option_type: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = fetch_spot(ticker)
    expirations = ticker.options
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}")
    selected = _select_monthly_expirations(expirations, years_ahead)
    if not selected:
        raise RuntimeError("No expirations found within the requested horizon.")

    rows: List[dict] = []
    now = dt.datetime.utcnow()
    for expiry_dt, expiry_str in selected:
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        chain = ticker.option_chain(expiry_str)
        data = chain.calls if option_type == "call" else chain.puts
        price_col = "C_mkt" if option_type == "call" else "P_mkt"
        for _, row in data.iterrows():
            rows.append(
                {
                    "S0": spot,
                    "K": float(row["strike"]),
                    "T": T,
                    price_col: float(row["lastPrice"]),
                }
            )
    return pd.DataFrame(rows)


def _bs_price(S0: float, K: float, T: float, vol: float, r: float, option_type: str) -> float:
    if T <= 0.0 or vol <= 0.0:
        intrinsic_call = max(0.0, S0 - K * math.exp(-r * T))
        if option_type == "call":
            return intrinsic_call
        return intrinsic_call - S0 + K * math.exp(-r * T)
    sqrt_T = math.sqrt(T)
    vol_sqrt_T = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    call = S0 * nd1 - K * math.exp(-r * T) * nd2
    if option_type == "call":
        return call
    put = call - S0 + K * math.exp(-r * T)
    return put


def implied_vol(price: float, S0: float, K: float, T: float, r: float, option_type: str) -> float:
    intrinsic = _bs_price(S0, K, T, 0.0, r, option_type)
    if price <= max(intrinsic, 1e-12):
        return 0.0
    vol_low, vol_high = 1e-6, 1.0
    price_high = _bs_price(S0, K, T, vol_high, r, option_type)
    while price_high < price and vol_high < 5.0:
        vol_high *= 2.0
        price_high = _bs_price(S0, K, T, vol_high, r, option_type)
    if price_high < price:
        return float("nan")
    for _ in range(100):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = _bs_price(S0, K, T, vol_mid, r, option_type)
        if abs(price_mid - price) < 1e-6:
            return vol_mid
        if price_mid > price:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
    return 0.5 * (vol_low + vol_high)


def _simpson_integral(values: np.ndarray, x: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    h = x[1] - x[0]
    return h / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * np.sum(values[1:-1:2])
        + 2.0 * np.sum(values[2:-2:2])
    )


def _heston_integrand(
    phi: np.ndarray,
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    Pnum: int,
) -> np.ndarray:
    u = 0.5 if Pnum == 1 else -0.5
    b = kappa - rho * sigma if Pnum == 1 else kappa
    a = kappa * theta
    x = math.log(S0)
    phi_complex = phi - 0j
    d = np.sqrt((rho * sigma * 1j * phi_complex - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi_complex - phi_complex ** 2))
    g = (b - rho * sigma * 1j * phi_complex + d) / (b - rho * sigma * 1j * phi_complex - d)
    exp_dt = np.exp(-d * T)
    log_term = np.log((1.0 - g * exp_dt) / (1.0 - g))
    C = r * 1j * phi_complex * T + (a / (sigma ** 2)) * ((b - rho * sigma * 1j * phi_complex + d) * T - 2.0 * log_term)
    D = ((b - rho * sigma * 1j * phi_complex + d) / (sigma ** 2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.real(np.exp(C + D * v0 + 1j * phi_complex * (x - math.log(K))) / (1j * phi_complex))


def heston_probability(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    Pnum: int,
) -> float:
    n = PHI_STEPS
    if n % 2 == 0:
        n += 1
    phi = np.linspace(1e-5, PHI_MAX, n)
    integrand = _heston_integrand(phi, S0, K, T, r, kappa, theta, sigma, rho, v0, Pnum)
    integral = _simpson_integral(integrand, phi)
    return 0.5 + (1.0 / math.pi) * integral


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> float:
    P1 = heston_probability(S0, K, T, r, kappa, theta, sigma, rho, v0, 1)
    P2 = heston_probability(S0, K, T, r, kappa, theta, sigma, rho, v0, 2)
    return S0 * P1 - K * math.exp(-r * T) * P2


def heston_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> float:
    call = heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
    return call - S0 + K * math.exp(-r * T)


def add_heston_iv(
    df: pd.DataFrame,
    option_type: str,
    params: Dict[str, float],
    r: float,
) -> pd.DataFrame:
    df_out = df.copy()
    iv_values = []
    price_values = []
    price_col = "C_mkt" if option_type == "call" else "P_mkt"
    for row in df_out.itertuples(index=False):
        S0 = float(row.S0)
        K = float(row.K)
        T = float(row.T)
        if option_type == "call":
            price = heston_call_price(S0, K, T, r, **params)
        else:
            price = heston_put_price(S0, K, T, r, **params)
        iv = implied_vol(price, S0, K, T, r, option_type)
        price_values.append(price)
        iv_values.append(iv)
    df_out[f"{price_col}_heston"] = price_values
    df_out["iv_heston"] = iv_values
    return df_out


@st.cache_data(show_spinner=False)
def prepare_surface(df_raw: pd.DataFrame, strike_width: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    required_cols = {"S0", "K", "T", "iv_heston"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")
    spot = float(df_raw["S0"].median())
    lower_bound = math.ceil((spot - strike_width) / 10.0) * 10.0
    upper_bound = math.ceil((spot + strike_width) / 10.0) * 10.0
    mask = (df_raw["K"] >= lower_bound) & (df_raw["K"] <= upper_bound)
    df = df_raw.loc[mask, ["S0", "K", "T", "iv_heston"]].copy()
    df = df[df["T"] >= MIN_MATURITY]
    if df.empty:
        raise ValueError("No strikes/maturities within the specified constraints.")
    df = df.sort_values(["T", "K"]).reset_index(drop=True)
    df = df.rename(columns={"iv_heston": "iv"})

    k_values = np.sort(df["K"].unique())
    t_values = np.sort(df["T"].unique())
    surface = df.pivot_table(index="T", columns="K", values="iv", aggfunc="mean")
    surface = surface.reindex(index=t_values, columns=k_values)
    surface = surface.interpolate(axis=1, limit_direction="both").interpolate(
        axis=1, limit_direction="both"
    )
    surface = surface.loc[surface.index >= MIN_MATURITY]
    if surface.empty:
        raise ValueError("Not enough maturities to build a surface.")
    return df, surface, spot


def plot_surface(surface: pd.DataFrame, spot: float, title: str) -> go.Figure:
    k_values = surface.columns.to_numpy(dtype=float)
    t_values = surface.index.to_numpy(dtype=float)
    t_min = float(t_values.min())
    t_max = float(t_values.max())
    KK, TT = np.meshgrid(k_values, t_values)
    plot_data = surface.to_numpy(dtype=float)
    if np.isnan(plot_data).any():
        plot_data = np.where(np.isnan(plot_data), np.nanmean(plot_data), plot_data)
    z_mean = float(np.nanmean(plot_data))
    z_std = float(np.nanstd(plot_data))
    z_min = z_mean - 2.0 * z_std
    z_max = z_mean + 2.0 * z_std

    fig = go.Figure(
        data=[
            go.Surface(
                x=KK,
                y=TT,
                z=plot_data,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=dict(text=f"Strike K — spot price ≈ {spot:.2f}")),
            yaxis=dict(title="Time to Maturity T (years)", range=[t_min, t_max]),
            zaxis=dict(title="Implied Volatility", range=[z_min, z_max]),
        ),
        height=600,
    )
    return fig


with st.sidebar:
    ticker_input = st.text_input("Ticker", value="SPY").strip().upper()
    st.caption(f"Expirations pulled up to {MAX_LOOKAHEAD_YEARS} years ahead.")
    strike_width = st.slider("Strike window around S₀", min_value=50, max_value=200, value=100, step=10)
    risk_free_rate = st.number_input("Risk-free rate r", value=0.02, format="%.4f")
    st.markdown("### Heston Parameters")
    kappa = st.number_input("Mean reversion (kappa)", min_value=0.01, max_value=10.0, value=2.0, step=0.1)
    theta = st.number_input("Long-term variance (theta)", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
    sigma = st.number_input("Volatility of variance (sigma)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
    rho = st.slider("Correlation (rho)", min_value=-0.99, max_value=0.99, value=-0.7, step=0.01)
    v0 = st.number_input("Initial variance (v0)", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
    run_button = st.button("Fetch & Plot")

params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}

if run_button:
    if not ticker_input:
        st.warning("Enter a ticker to proceed.")
    else:
        try:
            df_calls_raw = download_option_data(ticker_input, MAX_LOOKAHEAD_YEARS, "call")
            df_puts_raw = download_option_data(ticker_input, MAX_LOOKAHEAD_YEARS, "put")
            st.success(
                f"Fetched {len(df_calls_raw)} call rows and {len(df_puts_raw)} put rows for {ticker_input}."
            )
            df_calls = add_heston_iv(df_calls_raw, "call", params, risk_free_rate)
            df_puts = add_heston_iv(df_puts_raw, "put", params, risk_free_rate)
            calls_surface_data, surface_calls, spot = prepare_surface(df_calls, strike_width)
            puts_surface_data, surface_puts, _ = prepare_surface(df_puts, strike_width)

            st.write(
                f"Spot ≈ {spot:.2f}. Strike window "
                f"[{calls_surface_data['K'].min():.2f}, {calls_surface_data['K'].max():.2f}] "
                f"with maturities ≥ {MIN_MATURITY:.2f} years. Using Heston parameters: "
                f"kappa={kappa:.3f}, theta={theta:.3f}, sigma={sigma:.3f}, rho={rho:.3f}, v0={v0:.3f}, "
                f"risk-free rate r = {risk_free_rate:.3f}."
            )

            col_calls, col_puts = st.columns(2)
            with col_calls:
                st.subheader("Call IV Surface (Heston)")
                st.plotly_chart(plot_surface(surface_calls, spot, "Call IV (Heston)"), use_container_width=True)
            with col_puts:
                st.subheader("Put IV Surface (Heston)")
                st.plotly_chart(plot_surface(surface_puts, spot, "Put IV (Heston)"), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download or process data: {exc}")
else:
    st.info("Configure parameters in the sidebar and click **Fetch & Plot** to build the Heston surfaces.")
