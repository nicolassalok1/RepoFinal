#!/usr/bin/env python3
"""Generate a uniform maturity grid for option data using Black-Scholes pricing."""

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

INPUT_CSV = "test_calls.csv"
OUTPUT_CSV = "uniform_surface.csv"
S0 = 100.0
R = 0.0
Q = 0.0
MATURITY_GRID_MONTHS = 12
MATURITY_TOL = 1e-6


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S0: float, K: float, T: float, vol: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0 or vol <= 0:
        return max(0.0, S0 * math.exp(-q * T) - K * math.exp(-r * T))
    sqrt_T = math.sqrt(T)
    vol_sqrt_T = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return S0 * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def _infer_price_column(df: pd.DataFrame) -> str:
    price_aliases = {"price", "c_mkt", "p_mkt", "call_price", "option_price", "c"}
    lower_columns = {col.lower(): col for col in df.columns}
    for alias in price_aliases:
        if alias in lower_columns:
            return lower_columns[alias]
    raise ValueError(
        "Input CSV must contain a price column (e.g., 'price' or 'C_mkt'). "
        f"Columns found: {list(df.columns)}"
    )


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    price_column = _infer_price_column(df)
    if price_column != "price":
        df = df.rename(columns={price_column: "price"})
    missing_columns = {"K", "T", "price", "iv"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV missing required columns: {missing_columns}")
    if "type" in df.columns:
        df = df[df["type"].str.upper() == "C"].copy()
    if df.empty:
        raise ValueError("Input CSV does not contain any call option rows.")
    for col in ["K", "T", "price", "iv"]:
        df[col] = df[col].astype(float)
    return df[["K", "T", "price", "iv"]]


def build_uniform_surface(
    df: pd.DataFrame, S0_param: float, r: float, q: float
) -> Tuple[pd.DataFrame, List[float], List[float]]:
    unique_K = sorted(df["K"].unique())
    T_grid = [m / 12 for m in range(1, MATURITY_GRID_MONTHS + 1)]
    rows_out = []
    for K in unique_K:
        df_K = df[df["K"] == K].sort_values("T").reset_index(drop=True)
        T_values = df_K["T"].values
        if df_K.empty:
            continue
        for T_target in T_grid:
            matches = np.isclose(T_values, T_target, atol=MATURITY_TOL)
            if matches.any():
                row = df_K.iloc[np.where(matches)[0][0]]
                price_out = float(row["price"])
                iv_out = float(row["iv"])
                source = "original"
            else:
                idx_nearest = int(np.argmin(np.abs(T_values - T_target)))
                iv_out = float(df_K.iloc[idx_nearest]["iv"])
                price_out = bs_call(S0_param, float(K), float(T_target), iv_out, r=r, q=q)
                source = "synthetic"
            rows_out.append(
                {
                    "K": float(K),
                    "T": float(T_target),
                    "price": price_out,
                    "iv": iv_out,
                    "source": source,
                }
            )
    df_out = pd.DataFrame(rows_out)
    df_out = df_out.sort_values(["K", "T"]).reset_index(drop=True)
    return df_out, unique_K, T_grid


def main() -> None:
    df = load_input(INPUT_CSV)
    df_out, unique_K, T_grid = build_uniform_surface(df, S0, R, Q)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Unique strikes: {len(unique_K)}")
    expected_rows = len(unique_K) * len(T_grid)
    print(f"Output rows: {len(df_out)} (expected {expected_rows})")
    source_counts = df_out["source"].value_counts()
    for label in ["original", "synthetic"]:
        print(f"{label.capitalize()} rows: {int(source_counts.get(label, 0))}")


if __name__ == "__main__":
    main()
