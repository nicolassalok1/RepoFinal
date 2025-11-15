#!/usr/bin/env python3
"""Streamlit app to visualize market and BS implied volatility surfaces for calls and puts."""

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

st.set_page_config(page_title="IV Surfaces (Market & BS)", layout="wide")
st.title("Market vs. Black-Scholes Implied Volatility Surfaces")
st.write(
    "Download monthly call and put data via `yfinance`, filter strikes around the underlying spot, "
    "and visualize both market IV and recalculated Black-Scholes IV surfaces side-by-side."
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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S0: float, K: float, T: float, vol: float, r: float, option_type: str) -> float:
    if T <= 0.0 or vol <= 0.0:
        intrinsic_call = max(0.0, S0 - K * math.exp(-r * T))
        intrinsic_put = max(0.0, K * math.exp(-r * T) - S0)
        return intrinsic_call if option_type == "call" else intrinsic_put
    sqrt_T = math.sqrt(T)
    vol_sqrt_T = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    discount = math.exp(-r * T)
    if option_type == "call":
        return S0 * _normal_cdf(d1) - K * discount * _normal_cdf(d2)
    return K * discount * _normal_cdf(-d2) - S0 * _normal_cdf(-d1)


def implied_vol(price: float, S0: float, K: float, T: float, r: float, option_type: str) -> float:
    if T <= 0.0 or price <= 0.0 or S0 <= 0.0 or K <= 0.0:
        return 0.0
    intrinsic = _bs_price(S0, K, T, 0.0, r, option_type)
    if price <= intrinsic + 1e-8:
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


def add_implied_vol(df: pd.DataFrame, price_col: str, option_type: str, r: float) -> pd.DataFrame:
    df_out = df.copy()
    iv_values = []
    for row in df_out.itertuples(index=False):
        price = getattr(row, price_col, 0.0)
        iv = implied_vol(float(price), float(row.S0), float(row.K), float(row.T), r, option_type)
        iv_values.append(iv)
    df_out["iv_bs"] = iv_values
    return df_out


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
                    "iv_market": float(row["impliedVolatility"]),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def prepare_surface(
    df_raw: pd.DataFrame, strike_width: float, iv_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    required_cols = {"S0", "K", "T", iv_column}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")
    spot = float(df_raw["S0"].median())
    lower_bound = math.ceil((spot - strike_width) / 10.0) * 10.0
    upper_bound = math.ceil((spot + strike_width) / 10.0) * 10.0
    mask = (df_raw["K"] >= lower_bound) & (df_raw["K"] <= upper_bound)
    df = df_raw.loc[mask, ["S0", "K", "T", iv_column]].copy()
    df = df[df["T"] >= MIN_MATURITY]
    if df.empty:
        raise ValueError("No strikes/maturities within the specified constraints.")
    df = df.sort_values(["T", "K"]).reset_index(drop=True)
    df = df.rename(columns={iv_column: "iv"})

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


def plot_surface(surface: pd.DataFrame, spot: float, title_suffix: str) -> go.Figure:
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
        title=f"{title_suffix} Implied Volatility Surface",
        scene=dict(
            xaxis=dict(title=dict(text=f"Strike K — spot price ≈ {spot:.2f}")),
            yaxis=dict(title="Time to Maturity T (years)", range=[t_min, t_max]),
            zaxis=dict(title="Implied Volatility", range=[z_min, z_max]),
        ),
        height=550,
    )
    return fig


with st.sidebar:
    ticker_input = st.text_input("Ticker", value="SPY").strip().upper()
    st.caption(f"Expirations pulled up to {MAX_LOOKAHEAD_YEARS} years ahead.")
    strike_width = st.slider("Strike window around S₀", min_value=50, max_value=200, value=100, step=10)
    risk_free_rate = st.slider("Risk-free rate r", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    run_button = st.button("Fetch & Plot")


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
            df_calls = add_implied_vol(df_calls_raw, "C_mkt", "call", risk_free_rate)
            df_puts = add_implied_vol(df_puts_raw, "P_mkt", "put", risk_free_rate)

            calls_market, surface_calls_market, spot = prepare_surface(df_calls, strike_width, "iv_market")
            calls_bs, surface_calls_bs, _ = prepare_surface(df_calls, strike_width, "iv_bs")
            puts_market, surface_puts_market, _ = prepare_surface(df_puts, strike_width, "iv_market")
            puts_bs, surface_puts_bs, _ = prepare_surface(df_puts, strike_width, "iv_bs")

            st.write(
                f"Spot ≈ {spot:.2f}. Strike window "
                f"[{calls_market['K'].min():.2f}, {calls_market['K'].max():.2f}] "
                f"with maturities ≥ {MIN_MATURITY:.2f} years."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Call IV Surface (Market)")
                st.plotly_chart(plot_surface(surface_calls_market, spot, "Call (Market)"), use_container_width=True)
            with col2:
                st.subheader("Put IV Surface (Market)")
                st.plotly_chart(plot_surface(surface_puts_market, spot, "Put (Market)"), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Call IV Surface (Black-Scholes)")
                st.plotly_chart(plot_surface(surface_calls_bs, spot, "Call (BS)"), use_container_width=True)
            with col4:
                st.subheader("Put IV Surface (Black-Scholes)")
                st.plotly_chart(plot_surface(surface_puts_bs, spot, "Put (BS)"), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download or process data: {exc}")
else:
    st.info("Configure parameters in the sidebar and click **Fetch & Plot** to build all four surfaces.")
