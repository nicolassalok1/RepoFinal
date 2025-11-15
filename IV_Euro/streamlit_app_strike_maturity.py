#!/usr/bin/env python3
"""Streamlit app to visualize Black-Scholes prices vs strike and maturity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from numpy import log, sqrt, exp

st.set_page_config(page_title="BS Prices (Strike vs Maturity)", layout="wide")
st.title("Black-Scholes Price Heatmaps (Strike & Maturity)")
st.write(
    "Adjust the spot price, volatility, and interest rate, then explore call and put prices over a grid of "
    "strike values and time-to-maturity points."
)


class BlackScholes:
    def __init__(self, current_price: float, strike: float, time_to_maturity: float, volatility: float, interest_rate: float) -> None:
        self.S = float(current_price)
        self.K = float(strike)
        self.T = float(time_to_maturity)
        self.sigma = float(volatility)
        self.r = float(interest_rate)

    def price(self) -> tuple[float, float]:
        t = self.T
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * t) / (self.sigma * sqrt(t))
        d2 = d1 - self.sigma * sqrt(t)
        call = self.S * norm.cdf(d1) - self.K * exp(-self.r * t) * norm.cdf(d2)
        put = self.K * exp(-self.r * t) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put


def price_grid(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    sigma: float,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    call_prices = np.zeros((len(maturities), len(strikes)))
    put_prices = np.zeros_like(call_prices)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            bs = BlackScholes(spot, K, T, sigma, rate)
            call_prices[i, j], put_prices[i, j] = bs.price()
    return call_prices, put_prices


def plot_heatmap(values: np.ndarray, strikes: np.ndarray, maturities: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        values,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        xticklabels=np.round(strikes, 2),
        yticklabels=np.round(maturities, 2),
        cbar_kws={"label": title},
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Time to Maturity T (years)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


with st.sidebar:
    st.header("Model Inputs")
    spot = st.number_input("Spot price", min_value=0.01, value=100.0, step=1.0)
    volatility = st.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
    rate = st.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.3f")
    st.subheader("Strike Grid")
    strike_min = st.number_input("Min strike", min_value=0.01, value=80.0, step=1.0)
    strike_max = st.number_input("Max strike", min_value=strike_min + 0.01, value=120.0, step=1.0)
    strike_points = st.slider("Strike points", min_value=5, max_value=25, value=10)
    st.subheader("Maturity Grid")
    maturity_min = st.number_input("Min maturity (years)", min_value=0.01, value=0.1, step=0.05)
    maturity_max = st.number_input("Max maturity (years)", min_value=maturity_min + 0.01, value=2.0, step=0.1)
    maturity_points = st.slider("Maturity points", min_value=5, max_value=25, value=10)

strikes = np.linspace(strike_min, strike_max, strike_points)
maturities = np.linspace(maturity_min, maturity_max, maturity_points)
call_grid, put_grid = price_grid(spot, strikes, maturities, volatility, rate)

summary_df = pd.DataFrame(
    {
        "Spot": [spot],
        "Volatility": [volatility],
        "Risk-free rate": [rate],
        "Strike range": [f"{strike_min:.2f} → {strike_max:.2f} ({strike_points} pts)"],
        "Maturity range": [f"{maturity_min:.2f} → {maturity_max:.2f} ({maturity_points} pts)"],
    }
)
st.subheader("Input Summary")
st.dataframe(summary_df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Prices")
    fig_call = plot_heatmap(call_grid, strikes, maturities, "Call Price")
    st.pyplot(fig_call)
with col2:
    st.subheader("Put Prices")
    fig_put = plot_heatmap(put_grid, strikes, maturities, "Put Price")
    st.pyplot(fig_put)
