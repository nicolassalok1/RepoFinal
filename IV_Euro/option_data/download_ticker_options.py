#!/usr/bin/env python3
"""Download call option data for a ticker and store S0, K, T, C_mkt, iv in a CSV."""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch monthly call option quotes (next year) for a ticker."
    )
    parser.add_argument("--ticker", "-t", required=True, help="Equity ticker symbol (e.g., SPY).")
    parser.add_argument(
        "--output",
        "-o",
        default="options_data.csv",
        help="Destination CSV file for S0, K, T, C_mkt, iv columns.",
    )
    parser.add_argument(
        "--max-expirations",
        type=int,
        default=None,
        help="Optional cap on the number of monthly expirations (default: up to 12).",
    )
    return parser.parse_args()


def fetch_spot(ticker: yf.Ticker) -> float:
    history = ticker.history(period="1d")
    if history.empty:
        raise RuntimeError("Unable to retrieve spot price.")
    return float(history["Close"].iloc[-1])


def _select_monthly_expirations(expirations: List[str]) -> List[Tuple[dt.datetime, str]]:
    """Pick at most one expiration per calendar month over the next year."""
    today = dt.datetime.utcnow().date()
    limit_date = today + dt.timedelta(days=365)
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


def fetch_options_data(symbol: str, max_expirations: int | None) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = fetch_spot(ticker)
    expirations = ticker.options
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}")
    selected_expirations = _select_monthly_expirations(expirations)
    if max_expirations is not None:
        selected_expirations = selected_expirations[:max_expirations]
    if not selected_expirations:
        raise RuntimeError("No expirations found within the next year.")

    rows: List[dict] = []
    now = dt.datetime.utcnow()

    for expiry_dt, expiry_str in selected_expirations:
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        chain = ticker.option_chain(expiry_str).calls
        for _, row in chain.iterrows():
            rows.append(
                {
                    "S0": spot,
                    "K": float(row["strike"]),
                    "T": T,
                    "C_mkt": float(row["lastPrice"]),
                    "iv": float(row["impliedVolatility"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = fetch_options_data(args.ticker, args.max_expirations)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
