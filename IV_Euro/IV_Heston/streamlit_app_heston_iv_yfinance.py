#!/usr/bin/env python3
"""Streamlit app: calibrate Heston to yfinance options and plot IV surfaces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf

st.set_page_config(page_title="Heston IV Surfaces from yfinance", layout="wide")
st.title("Heston Calibration from yfinance and Implied Volatility Surfaces")
st.write(
    "Download call/put options via `yfinance`, calibrate Heston parameters on calls using PyTorch, "
    "then plot Heston-implied Black–Scholes IV surfaces as functions of moneyness K/S₀ and time to maturity."
)

# Import Heston torch utilities
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Heston" / "NN"))

from heston_torch import HestonParams, carr_madan_call_torch  # type: ignore  # noqa: E402

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")


def fetch_spot(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        raise RuntimeError("Unable to retrieve spot price.")
    return float(hist["Close"].iloc[-1])


def _select_monthly_expirations(expirations, years_ahead: float = 2.5) -> list[str]:
    today = pd.Timestamp.utcnow().date()
    limit_date = today + pd.Timedelta(days=365 * years_ahead)
    monthly: Dict[Tuple[int, int], Tuple[pd.Timestamp, str]] = {}
    for exp in expirations:
        exp_ts = pd.Timestamp(exp)
        exp_date = exp_ts.date()
        if not (today < exp_date <= limit_date):
            continue
        key = (exp_date.year, exp_date.month)
        if key not in monthly or exp_ts < monthly[key][0]:
            monthly[key] = (exp_ts, exp)
    return [item[1] for item in sorted(monthly.values(), key=lambda x: x[0])]


@st.cache_data(show_spinner=True)
def download_options(symbol: str, option_type: str, years_ahead: float = 2.5) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = fetch_spot(symbol)
    expirations = ticker.options
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}")
    selected = _select_monthly_expirations(expirations, years_ahead)
    rows: list[dict] = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for expiry in selected:
        expiry_dt = pd.Timestamp(expiry)
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        chain = ticker.option_chain(expiry)
        data = chain.calls if option_type == "call" else chain.puts
        price_col = "C_mkt" if option_type == "call" else "P_mkt"
        for _, row in data.iterrows():
            rows.append({"S0": spot, "K": float(row["strike"]), "T": T, price_col: float(row["lastPrice"])})
    return pd.DataFrame(rows)


def prices_from_unconstrained(
    u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, r: float, q: float
) -> torch.Tensor:
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    prices = []
    for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
        price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
        prices.append(price_i)
    return torch.stack(prices)


def loss(
    u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, C_mkt_t: torch.Tensor, r: float, q: float
) -> torch.Tensor:
    model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
    diff = model_prices - C_mkt_t
    return 0.5 * (diff**2).mean()


def calibrate_heston_from_yf(
    symbol: str,
    r: float,
    q: float,
    years_ahead: float,
    max_points: int,
    max_iters: int,
    lr: float,
    progress_callback=None,
    log_callback=None,
) -> tuple[dict[str, float], list[float], pd.DataFrame, pd.DataFrame]:
    calls_df = download_options(symbol, "call", years_ahead=years_ahead)
    puts_df = download_options(symbol, "put", years_ahead=years_ahead)
    df = calls_df[["S0", "K", "T", "C_mkt"]].dropna().copy()
    n_total = len(df)
    if n_total > max_points:
        df = df.sort_values("T")
        idx = np.linspace(0, n_total - 1, max_points, dtype=int)
        df = df.iloc[idx]
    df = df.reset_index(drop=True)

    S0_t = torch.tensor(df["S0"].to_numpy(), dtype=torch.float64, device=DEVICE)
    K_t = torch.tensor(df["K"].to_numpy(), dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor(df["T"].to_numpy(), dtype=torch.float64, device=DEVICE)
    C_mkt_t = torch.tensor(df["C_mkt"].to_numpy(), dtype=torch.float64, device=DEVICE)

    u = torch.tensor([1.0, -3.0, -0.5, -0.5, -3.0], dtype=torch.float64, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)
    history: list[float] = []

    for it in range(max_iters):
        optimizer.zero_grad()
        L = loss(u, S0_t, K_t, T_t, C_mkt_t, r, q)
        L.backward()
        optimizer.step()
        history.append(float(L.detach().cpu()))
        if progress_callback is not None:
            progress_callback(it, max_iters, history[-1])
        if log_callback is not None:
            log_callback(it, history[-1])

    with torch.no_grad():
        params_fin = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    calib = {
        "kappa": float(params_fin.kappa.cpu()),
        "theta": float(params_fin.theta.cpu()),
        "sigma": float(params_fin.sigma.cpu()),
        "rho": float(params_fin.rho.cpu()),
        "v0": float(params_fin.v0.cpu()),
    }
    summary = pd.DataFrame(
        {
            "used_quotes": [len(df)],
            "total_quotes": [n_total],
            "loss_final": [history[-1] if history else float("nan")],
        }
    )
    return calib, history, summary, puts_df


def bs_price(S0: float, K: float, T: float, vol: float, r: float) -> float:
    if T <= 0.0 or vol <= 0.0:
        return max(0.0, S0 - K * math.exp(-r * T))
    sqrt_T = math.sqrt(T)
    v = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / v
    d2 = d1 - v
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return S0 * nd1 - K * math.exp(-r * T) * nd2


def implied_vol(price: float, S0: float, K: float, T: float, r: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    intrinsic = max(0.0, S0 - K * math.exp(-r * T))
    if price <= intrinsic + 1e-12:
        return 0.0
    low, high = 1e-6, 1.0
    p_high = bs_price(S0, K, T, high, r)
    while p_high < price and high < 5.0:
        high *= 2.0
        p_high = bs_price(S0, K, T, high, r)
    if p_high < price:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = bs_price(S0, K, T, mid, r)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


@st.cache_data(show_spinner=True)
def compute_iv_surfaces(
    calib: dict[str, float], r: float, S0_ref: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K_grid = np.linspace(S0_ref - 100.0, S0_ref + 100.0, 21)
    T_grid_K = np.arange(0.1, 2.6, 0.1)
    iv_KT_call = np.zeros((len(T_grid_K), len(K_grid)))
    iv_KT_put = np.zeros_like(iv_KT_call)

    params_tensor = HestonParams(
        kappa=torch.tensor(calib["kappa"], dtype=torch.float64, device=DEVICE),
        theta=torch.tensor(calib["theta"], dtype=torch.float64, device=DEVICE),
        sigma=torch.tensor(calib["sigma"], dtype=torch.float64, device=DEVICE),
        rho=torch.tensor(calib["rho"], dtype=torch.float64, device=DEVICE),
        v0=torch.tensor(calib["v0"], dtype=torch.float64, device=DEVICE),
    )

    with torch.no_grad():
        for i, T_val in enumerate(T_grid_K):
            K_vec = torch.tensor(K_grid, dtype=torch.float64, device=DEVICE)
            prices_call_t = carr_madan_call_torch(
                torch.tensor(S0_ref, dtype=torch.float64, device=DEVICE), r, 0.0, T_val, params_tensor, K_vec
            )
            prices_call = prices_call_t.cpu().numpy()
            for j, price_c in enumerate(prices_call):
                K_val = float(K_grid[j])
                iv_KT_call[i, j] = implied_vol(price_c, S0_ref, K_val, T_val, r)
                put_price = price_c - S0_ref + K_val * math.exp(-r * T_val)
                iv_KT_put[i, j] = implied_vol(put_price, S0_ref, K_val, T_val, r)

    return K_grid, T_grid_K, iv_KT_call, iv_KT_put


with st.sidebar:
    st.header("Data & Calibration")
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    years_ahead = 2.5
    rf_rate = st.slider("Risk-free rate r", min_value=-0.01, max_value=0.10, value=0.02, step=0.005)
    max_quotes = 300
    max_iters = 100
    lr = 5e-3
    run_button = st.button("Download, Calibrate & Plot")


if run_button:
    if not ticker:
        st.warning("Enter a ticker to proceed.")
    else:
        try:
            calls_df = download_options(ticker, "call", years_ahead=years_ahead)
            puts_df = download_options(ticker, "put", years_ahead=years_ahead)
            st.write(f"Downloaded {len(calls_df)} call rows and {len(puts_df)} put rows for {ticker}.")
            S0_ref = float(calls_df["S0"].median())

            progress_bar = st.progress(0.0, text="Calibrating Heston parameters...")
            log_box = st.empty()
            log_messages: list[str] = []

            def progress_cb(iter_idx: int, total: int, loss_val: float) -> None:
                fraction = (iter_idx + 1) / total
                progress_bar.progress(fraction, text=f"Calibrating... loss={loss_val:.3e}")

            def log_cb(iter_idx: int, loss_val: float) -> None:
                log_messages.append(f"Iter {iter_idx:03d} | loss = {loss_val:.6e}")
                recent = "\n\n".join(log_messages[-10:])
                log_box.write(f"Calibration log:\n{recent}")

            calib, history, summary, _ = calibrate_heston_from_yf(
                ticker,
                r=rf_rate,
                q=0.0,
                years_ahead=years_ahead,
                max_points=max_quotes,
                max_iters=max_iters,
                lr=lr,
                progress_callback=progress_cb,
                log_callback=log_cb,
            )
            progress_bar.empty()
            log_box.empty()
            st.subheader("Calibration summary")
            st.dataframe(summary)
            st.subheader("Calibrated Heston parameters")
            st.dataframe(pd.Series(calib, name="Heston params").to_frame())

            st.subheader("Calibration loss history")
            loss_df = pd.DataFrame({"iteration": range(len(history)), "loss": history})
            st.line_chart(loss_df.set_index("iteration"))

            st.subheader("Implied volatility surfaces (Heston → BS)")
            K_grid, T_grid_K, iv_KT_call, iv_KT_put = compute_iv_surfaces(calib, r=rf_rate, S0_ref=S0_ref)

            KK, TT_K = np.meshgrid(K_grid, T_grid_K)
            fig_K_call = go.Figure(data=[go.Surface(x=KK, y=TT_K, z=iv_KT_call, colorscale="Viridis")])
            fig_K_call.update_layout(
                title="Heston Implied Vol Surface - Calls (K vs T)",
                scene=dict(
                    xaxis_title="Strike K",
                    yaxis_title="T (years)",
                    zaxis_title="IV",
                ),
                width=900,
                height=600,
            )

            fig_K_put = go.Figure(data=[go.Surface(x=KK, y=TT_K, z=iv_KT_put, colorscale="Viridis")])
            fig_K_put.update_layout(
                title="Heston Implied Vol Surface - Puts (K vs T)",
                scene=dict(
                    xaxis_title="Strike K",
                    yaxis_title="T (years)",
                    zaxis_title="IV",
                ),
                width=900,
                height=600,
            )

            col_call, col_put = st.columns(2)
            with col_call:
                st.subheader("Call IV Surface (K vs T)")
                st.plotly_chart(fig_K_call, use_container_width=True)
            with col_put:
                st.subheader("Put IV Surface (K vs T)")
                st.plotly_chart(fig_K_put, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download or process data: {exc}")
else:
    st.info("Configure ticker and calibration settings in the sidebar and click **Download, Calibrate & Plot**.")
