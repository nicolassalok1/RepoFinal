"""
Download SPY option data from Yahoo Finance and keep only contracts that fall on a
regular (T, K) grid: T ∈ [1/12, 2.0] with monthly (≈0.08333) increments, and K ∈ [S0-100, S0+100]
with a stride of 10. The filtered dataset is stored as test.csv for downstream use.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


def compute_time_to_maturity(expiry_str: str) -> float:
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = expiry_dt - now
    return max(delta.total_seconds(), 0.0) / (365.0 * 24 * 3600)


def clean_price(row: pd.Series) -> float:
    last_price = row.get("lastPrice", np.nan)
    if not pd.isna(last_price):
        return float(last_price)
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    return float((bid + ask) / 2.0)


def download_grid(ticker: str) -> pd.DataFrame:
    asset = yf.Ticker(ticker)
    history = asset.history(period="1d")
    if history.empty:
        raise RuntimeError("No spot history returned by Yahoo Finance.")
    S0 = float(history["Close"].iloc[-1])
    expirations: List[str] = getattr(asset, "options", [])
    if not expirations:
        raise RuntimeError("No option expirations available for ticker.")

    min_T, max_T, step_T = 1.0 / 12.0, 2.0, 1.0 / 12.0
    min_K = math.floor((S0 - 100.0) / 10.0) * 10.0
    max_K = math.ceil((S0 + 100.0) / 10.0) * 10.0
    strike_step = 10.0

    records = []
    for expiry in expirations:
        T_years = compute_time_to_maturity(expiry)
        T_rounded = round(T_years, 4)
        if T_rounded < min_T or T_rounded > max_T:
            continue
        # snap maturities to the nearest monthly (1/12) grid
        T_snap = float(np.round(T_rounded / step_T) * step_T)
        if not (min_T <= T_snap <= max_T):
            continue

        try:
            chain = asset.option_chain(expiry)
        except Exception:
            continue

        for opt_type, frame in (("C", chain.calls), ("P", chain.puts)):
            if frame is None or frame.empty:
                continue
            for _, row in frame.iterrows():
                strike = row.get("strike", np.nan)
                if pd.isna(strike):
                    continue
                strike_snap = float(np.round(strike / strike_step) * strike_step)
                if abs(strike - strike_snap) > 1e-6:
                    continue
                if strike_snap < min_K or strike_snap > max_K:
                    continue
                price = clean_price(row)
                if pd.isna(price):
                    continue
                records.append(
                    {
                        "S0": S0,
                        "K": strike_snap,
                        "T": round(T_snap, 2),
                        "C_mkt": price,
                        "type": opt_type,
                        "iv": row.get("impliedVolatility", np.nan),
                    }
                )

    if not records:
        raise RuntimeError("No contracts matched the target grid.")

    df = pd.DataFrame(records).dropna(subset=["S0", "K", "T", "C_mkt"])
    df = df.sort_values(by=["T", "K", "type"]).reset_index(drop=True)
    return df


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    try:
        df = download_grid(ticker)
    except Exception as exc:
        print(f"Error while downloading grid-aligned options: {exc}")
        sys.exit(1)

    base_dir = Path(__file__).parent
    full_path = base_dir / "test.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved grid-aligned dataset ({len(df)} rows) to {full_path.name}")

    calls = df[df["type"] == "C"].copy()
    puts = df[df["type"] == "P"].copy()
    if not calls.empty:
        call_path = base_dir / "test_calls.csv"
        calls.to_csv(call_path, index=False)
        print(f"Saved call subset: {call_path.name} ({len(calls)} rows)")
    if not puts.empty:
        put_path = base_dir / "test_puts.csv"
        puts.to_csv(put_path, index=False)
        print(f"Saved put subset: {put_path.name} ({len(puts)} rows)")


if __name__ == "__main__":
    main()
