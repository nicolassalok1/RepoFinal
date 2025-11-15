#!/usr/bin/env python3
"""Streamlit app: Heston prices vs moneyness and maturity."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Heston Prices (Moneyness vs T)", layout="wide")
st.title("Heston Option Prices by Strike and Spot")
st.write(
    "Choose the spot and strike ranges, set the maturity with the slider, and tune the Heston parameters. "
    "Call prices are displayed as heatmaps over the (K, S₀) plane."
)

PHI_MAX = 200.0
PHI_STEPS = 2001  # odd for Simpson


@st.cache_resource(show_spinner=False)
def _get_theme() -> None:
    sns.set_theme(style="whitegrid")


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
    phi = np.linspace(1e-5, PHI_MAX, PHI_STEPS)
    u = 0.5 if Pnum == 1 else -0.5
    b = kappa - rho * sigma if Pnum == 1 else kappa
    a_param = kappa * theta
    x = math.log(S0)
    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
    exp_dt = np.exp(-d * T)
    log_term = np.log((1.0 - g * exp_dt) / (1.0 - g))
    C = r * 1j * phi * T + (a_param / (sigma ** 2)) * ((b - rho * sigma * 1j * phi + d) * T - 2.0 * log_term)
    D = ((b - rho * sigma * 1j * phi + d) / (sigma ** 2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    integrand = np.real(np.exp(C + D * v0 + 1j * phi * (x - math.log(K))) / (1j * phi))
    h = phi[1] - phi[0]
    integral = h / 3.0 * (
        integrand[0]
        + integrand[-1]
        + 4.0 * np.sum(integrand[1:-1:2])
        + 2.0 * np.sum(integrand[2:-2:2])
    )
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


def price_grid(
    spots: np.ndarray,
    strikes: np.ndarray,
    maturity: float,
    params: dict[str, float],
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    call_prices = np.zeros((len(spots), len(strikes)))
    put_prices = np.zeros_like(call_prices)
    for i, S0 in enumerate(spots):
        for j, K in enumerate(strikes):
            call_prices[i, j] = round(heston_call_price(S0, K, maturity, rate, **params), 2)
            put_prices[i, j] = round(heston_put_price(S0, K, maturity, rate, **params), 2)
    return call_prices, put_prices


def plot_heatmap(values: np.ndarray, strikes: np.ndarray, spots: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        values,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        xticklabels=np.round(strikes, 2),
        yticklabels=np.round(spots, 2),
        ax=ax,
        cbar_kws={"label": title},
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot S₀")
    ax.set_title(title)
    plt.tight_layout()
    return fig


_get_theme()
with st.sidebar:
    st.header("Inputs")
    ref_spot = st.number_input("Reference spot price", min_value=0.01, value=100.0, step=1.0)
    rate = st.slider("Risk-free rate r", min_value=-0.01, max_value=0.1, value=0.02, step=0.005)
    maturity = st.slider("Time to maturity T (years)", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
    st.subheader("Strike grid")
    k_min = st.number_input("Min strike", min_value=0.01, value=0.8 * ref_spot, step=1.0)
    k_max = st.number_input("Max strike", min_value=k_min + 0.01, value=1.2 * ref_spot, step=1.0)
    k_points = st.slider("Strike points", min_value=5, max_value=25, value=10)
    st.subheader("Spot grid")
    s_min = st.number_input("Min spot", min_value=0.01, value=0.8 * ref_spot, step=1.0)
    s_max = st.number_input("Max spot", min_value=s_min + 0.01, value=1.2 * ref_spot, step=1.0)
    s_points = st.slider("Spot points", min_value=5, max_value=25, value=10)
    st.subheader("Heston parameters")
    kappa = st.slider("κ (mean reversion)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    theta = st.slider("θ (long-term variance)", min_value=0.001, max_value=0.5, value=0.04, step=0.005)
    sigma = st.slider("σ (vol of variance)", min_value=0.01, max_value=1.5, value=0.5, step=0.01)
    rho = st.slider("ρ (correlation)", min_value=-0.99, max_value=0.99, value=-0.7, step=0.01)
    v0 = st.slider("v₀ (initial variance)", min_value=0.001, max_value=0.5, value=0.04, step=0.005)

params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}
strike_grid = np.linspace(k_min, k_max, k_points)
spot_grid = np.linspace(s_min, s_max, s_points)

with st.spinner("Pricing options via Heston model..."):
    call_grid, put_grid = price_grid(spot_grid, strike_grid, maturity, params, rate)

summary = pd.DataFrame(
    {
        "Reference spot": [ref_spot],
        "Rate": [rate],
        "Maturity T": [maturity],
        "Strike range": [f"{k_min:.2f} → {k_max:.2f} ({k_points} pts)"],
        "Spot range": [f"{s_min:.2f} → {s_max:.2f} ({s_points} pts)"],
        "κ": [kappa],
        "θ": [theta],
        "σ": [sigma],
        "ρ": [rho],
        "v₀": [v0],
    }
)
st.subheader("Input summary")
st.dataframe(summary)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Prices (Heston)")
    st.pyplot(plot_heatmap(call_grid, strike_grid, spot_grid, "Call Price"))
with col2:
    st.subheader("Put Prices (Heston)")
    st.pyplot(plot_heatmap(put_grid, strike_grid, spot_grid, "Put Price"))
