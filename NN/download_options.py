# download_options.py
"""
Standalone utility to download a full option chain from Yahoo Finance and store it
as a clean CSV with S0, strike, market price, time-to-maturity, contract type, and
Yahoo's implied volatility. Designed for quick quantitative explorations.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def compute_time_to_maturity(expiry_str: str) -> float:
    """Convert an expiry date string (YYYY-MM-DD) into an ACT/365 year fraction."""
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = expiry_dt - now
    return max(delta.total_seconds(), 0.0) / (365.0 * 24 * 3600)


def clean_price(row: pd.Series) -> float:
    """
    Return the best available market price for an option row.
    Prefer Yahoo's lastPrice; if missing, fall back to the mid of bid/ask.
    """
    last_price = row.get("lastPrice", np.nan)
    if not pd.isna(last_price):
        return float(last_price)

    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    return float((bid + ask) / 2.0)


def parse_ticker() -> Optional[str]:
    """Extract the ticker from the command line; print usage if absent."""
    if len(sys.argv) < 2:
        print("Usage: python download_options.py TICKER")
        return None
    return sys.argv[1].strip().upper()


def main() -> None:
    ticker = parse_ticker()
    if not ticker:
        sys.exit(1)

    print("Downloading spot price...")
    asset = yf.Ticker(ticker)
    try:
        history = asset.history(period="1d")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: invalid ticker or no data available. ({exc})")
        sys.exit(1)

    if history.empty:
        print("Error: invalid ticker or no data available.")
        sys.exit(1)

    S0 = float(history["Close"].iloc[-1])
    if np.isnan(S0):
        print("Error: could not determine a valid spot price.")
        sys.exit(1)

    print("Fetching option expirations...")
    expirations = getattr(asset, "options", [])
    if not expirations:
        print("Error: invalid ticker or no data available.")
        sys.exit(1)

    records = []
    for expiry in expirations:
        T = compute_time_to_maturity(expiry)
        if T <= 0:
            continue  # Skip already expired contracts

        print(f"Processing expiry: {expiry} (T = {T:.4f} years)...")
        try:
            chain = asset.option_chain(expiry)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed to fetch {expiry}: {exc}")
            continue

        for opt_type, frame in (("C", chain.calls), ("P", chain.puts)):
            if frame is None or frame.empty:
                continue
            for _, row in frame.iterrows():
                strike = row.get("strike", np.nan)
                iv = row.get("impliedVolatility", np.nan)
                price = clean_price(row)
                records.append(
                    {
                        "S0": S0,
                        "K": float(strike) if not pd.isna(strike) else np.nan,
                        "C_mkt": price,
                        "T": T,
                        "type": opt_type,
                        "iv": float(iv) if not pd.isna(iv) else np.nan,
                    }
                )

    if not records:
        print("Error: no option data could be collected.")
        sys.exit(1)

    data = pd.DataFrame(records)
    data = data[(data["K"] > 0) & (data["T"] > 0)]
    data = data.dropna(subset=["S0", "K", "C_mkt", "T", "type", "iv"])

    # Retain strikes within ±150 of the contemporaneous spot to focus on relevant moneyness range.
    strike_lower = data["S0"] - 150.0
    strike_upper = data["S0"] + 150.0
    data = data[(data["K"] >= strike_lower) & (data["K"] <= strike_upper)]

    data = data.sort_values(by=["T", "K"]).reset_index(drop=True)
    data['T'] = data['T'].round(2)
    data['iv'] = data['iv'].round(2)
    data['S0'] = data['S0'].round(2)
    data["T"] = data["T"].round(2)
    data["iv"] = data["iv"].round(2)

    if data.empty:
        print("Error: no clean option rows remain after filtering.")
        sys.exit(1)

    base_path = Path(__file__).with_name(f"options_{ticker}.csv")
    data.to_csv(base_path, index=False)
    print(f"Total options collected: {len(data)}")
    print(f"Saved file: {base_path.name}")

    calls = data[data["type"] == "C"]
    puts = data[data["type"] == "P"]
    if not calls.empty:
        call_path = Path(__file__).with_name(f"options_{ticker}_calls.csv")
        calls.to_csv(call_path, index=False)
        print(f"Saved call subset: {call_path.name} ({len(calls)} rows)")
    if not puts.empty:
        put_path = Path(__file__).with_name(f"options_{ticker}_puts.csv")
        puts.to_csv(put_path, index=False)
        print(f"Saved put subset: {put_path.name} ({len(puts)} rows)")


if __name__ == "__main__":
    main()
