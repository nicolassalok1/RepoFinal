#!/usr/bin/env python3
"""Streamlit app: Heston calibration from yfinance and call/put heatmaps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf

# Streamlit page config
st.set_page_config(page_title="Heston Heatmaps from yfinance", layout="wide")
st.title("Heston Calibration (yfinance) & Price Heatmaps")
st.write(
    "Télécharge des options via `yfinance`, calibre les paramètres de Heston avec PyTorch, "
    "puis trace des heatmaps des prix de calls/puts Heston en fonction de S et K pour une maturité fixe."
)

# Import du module Heston torch
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


def calibrate_heston_from_calls(
    calls_df: pd.DataFrame,
    r: float,
    q: float,
    max_points: int,
    max_iters: int,
    lr: float,
    progress_callback: Callable[[int, int, float], None] | None = None,
    log_callback: Callable[[int, float], None] | None = None,
) -> tuple[dict[str, float], list[float], pd.DataFrame]:
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
        curr_loss = float(L.detach().cpu())
        history.append(curr_loss)
        if progress_callback is not None:
            progress_callback(it, max_iters, curr_loss)
        if log_callback is not None:
            log_callback(it, curr_loss)

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
    return calib, history, summary


def params_from_calib(calib: dict[str, float]) -> HestonParams:
    return HestonParams(
        kappa=torch.tensor(calib["kappa"], dtype=torch.float64, device=DEVICE),
        theta=torch.tensor(calib["theta"], dtype=torch.float64, device=DEVICE),
        sigma=torch.tensor(calib["sigma"], dtype=torch.float64, device=DEVICE),
        rho=torch.tensor(calib["rho"], dtype=torch.float64, device=DEVICE),
        v0=torch.tensor(calib["v0"], dtype=torch.float64, device=DEVICE),
    )


def compute_price_heatmaps(
    calib: dict[str, float],
    r: float,
    q: float,
    S0_ref: float,
    S_span: float,
    K_span: float,
    points: int,
    maturity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S_grid = np.linspace(S0_ref - S_span, S0_ref + S_span, points)
    K_grid = np.linspace(S0_ref - K_span, S0_ref + K_span, points)
    call_heatmap = np.zeros((len(S_grid), len(K_grid)))
    put_heatmap = np.zeros_like(call_heatmap)

    params_tensor = params_from_calib(calib)

    with torch.no_grad():
        T_tensor = torch.tensor(maturity, dtype=torch.float64, device=DEVICE)
        for i, S_val in enumerate(S_grid):
            S_tensor = torch.tensor(S_val, dtype=torch.float64, device=DEVICE)
            for j, K_val in enumerate(K_grid):
                K_tensor = torch.tensor(K_val, dtype=torch.float64, device=DEVICE)
                call_price = carr_madan_call_torch(S_tensor, r, q, T_tensor, params_tensor, K_tensor)
                call_val = float(call_price.cpu())
                call_heatmap[i, j] = call_val
                put_heatmap[i, j] = call_val - S_val + K_val * math.exp(-r * maturity)
    return S_grid, K_grid, call_heatmap, put_heatmap


def plot_heatmap(matrix: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=np.round(x_grid, 2),
            y=np.round(y_grid, 2),
            colorscale="Viridis",
            colorbar=dict(title=title),
        )
    )
    fig.update_layout(xaxis_title="Strike K", yaxis_title="Spot S₀", yaxis_autorange="reversed", title=title)
    return fig


with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    rf_rate = st.slider("Taux sans risque r", min_value=-0.01, max_value=0.10, value=0.02, step=0.005)
    years_ahead = 2.5
    max_quotes = 300
    max_iters = 100
    lr = 5e-3
    heatmap_span = st.number_input("Écart autour de S₀ (±)", min_value=10.0, max_value=150.0, value=100.0, step=5.0)
    heatmap_points = st.slider("Points sur chaque axe", min_value=5, max_value=30, value=21, step=2)
    maturity = st.number_input("Maturité heatmap (années)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    run_button = st.button("Télécharger & Calibrer")


if run_button:
    if not ticker:
        st.warning("Merci de renseigner un ticker.")
    else:
        try:
            calls_df = download_options(ticker, "call", years_ahead=years_ahead)
            puts_df = download_options(ticker, "put", years_ahead=years_ahead)
            st.write(f"{len(calls_df)} calls et {len(puts_df)} puts téléchargés pour {ticker}.")
            S0_ref = float(calls_df["S0"].median())

            progress_bar = st.progress(0.0, text="Calibration Heston...")
            log_box = st.empty()
            log_messages: list[str] = []

            def progress_cb(iter_idx: int, total: int, loss_val: float) -> None:
                fraction = (iter_idx + 1) / total
                progress_bar.progress(fraction, text=f"Calibration... loss={loss_val:.3e}")

            def log_cb(iter_idx: int, loss_val: float) -> None:
                log_messages.append(f"Iter {iter_idx:03d} | loss = {loss_val:.6e}")
                log_box.write("\n".join(log_messages[-10:]))

            calib, history, summary = calibrate_heston_from_calls(
                calls_df,
                r=rf_rate,
                q=0.0,
                max_points=max_quotes,
                max_iters=max_iters,
                lr=lr,
                progress_callback=progress_cb,
                log_callback=log_cb,
            )
            progress_bar.empty()
            log_box.empty()

            st.subheader("Résumé calibration")
            st.dataframe(summary)
            st.subheader("Paramètres Heston calibrés")
            st.dataframe(pd.Series(calib, name="params").to_frame())

            st.subheader("Historique de loss")
            loss_df = pd.DataFrame({"iteration": range(len(history)), "loss": history})
            st.line_chart(loss_df.set_index("iteration"))

            st.subheader("Heatmaps Heston (prix)")
            S_grid, K_grid, call_heatmap, put_heatmap = compute_price_heatmaps(
                calib,
                r=rf_rate,
                q=0.0,
                S0_ref=S0_ref,
                S_span=heatmap_span,
                K_span=heatmap_span,
                points=heatmap_points,
                maturity=maturity,
            )
            summary_heatmap = pd.DataFrame(
                {
                    "Reference spot": [S0_ref],
                    "Rate": [rf_rate],
                    "Maturity T": [maturity],
                    "Strike range": [f"{K_grid[0]:.2f} → {K_grid[-1]:.2f} ({len(K_grid)} pts)"],
                    "Spot range": [f"{S_grid[0]:.2f} → {S_grid[-1]:.2f} ({len(S_grid)} pts)"],
                    "κ": [calib["kappa"]],
                    "θ": [calib["theta"]],
                    "σ": [calib["sigma"]],
                    "ρ": [calib["rho"]],
                    "v₀": [calib["v0"]],
                }
            )
            st.dataframe(summary_heatmap)

            fig_call = plot_heatmap(call_heatmap, K_grid, S_grid, "Call Price (Heston)")
            fig_put = plot_heatmap(put_heatmap, K_grid, S_grid, "Put Price (Heston)")
            col_call, col_put = st.columns(2)
            with col_call:
                st.plotly_chart(fig_call, use_container_width=True)
            with col_put:
                st.plotly_chart(fig_put, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Échec du téléchargement ou de la calibration: {exc}")
else:
    st.info("Configure les paramètres dans la barre latérale puis clique sur **Télécharger & Calibrer**.")
