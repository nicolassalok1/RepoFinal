#!/usr/bin/env python3
"""
Script pour récupérer des données d'options depuis Yahoo Finance (via yfinance)
et produire un CSV au même format que `Basket/data/train.csv`.

Colonnes générées :
    - S/K        : spot / strike
    - Maturity   : maturité en années (T)
    - Volatility : volatilité implicite (decimal)
    - Rate       : taux sans risque (constant, paramétrable)
    - Labels     : prix de l'option (lastPrice, côté Yahoo)
    - Prices     : spot sous-jacent S0 (répété sur toutes les lignes)
    - Strikes    : strike de l'option
    - Weight_0   : poids 1er actif du panier (ici 1.0 par défaut)
    - Weight_1   : poids 2e actif (0.0)
    - Weight_2   : poids 3e actif (0.0)

Exemple d'utilisation :
    python Basket/scripts/fetch_yfinance_train.py \\
        --ticker SPY \\
        --output Basket/data/train_yf_SPY.csv \\
        --rate 0.01 \\
        --max-expiries 5

Prérequis :
    pip install yfinance pandas numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class OptionRow:
    s_over_k: float
    maturity: float
    volatility: float
    rate: float
    label: float
    spot: float
    strike: float
    w0: float
    w1: float
    w2: float

    def to_dict(self) -> dict:
        return {
            "S/K": self.s_over_k,
            "Maturity": self.maturity,
            "Volatility": self.volatility,
            "Rate": self.rate,
            "Labels": self.label,
            "Prices": self.spot,
            "Strikes": self.strike,
            "Weight_0": self.w0,
            "Weight_1": self.w1,
            "Weight_2": self.w2,
        }


def year_fraction(expiry_str: str, now: datetime | None = None) -> float:
    """Transforme une date d'échéance 'YYYY-MM-DD' en maturité T (années)."""
    if now is None:
        now = datetime.now(timezone.utc).date()
    else:
        now = now.date()
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (expiry_date - now).days
    if days <= 0:
        return 0.0
    return days / 365.0


def fetch_option_rows(
    ticker: str,
    rate: float,
    max_expiries: int | None = None,
    include_puts: bool = True,
    include_calls: bool = True,
    max_rows: int | None = None,
) -> List[OptionRow]:
    """
    Récupère des options pour un ticker donné et renvoie une liste de OptionRow.
    """
    tk = yf.Ticker(ticker)

    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Aucune donnée historique pour le ticker {ticker!r}")
    spot = float(hist["Close"].iloc[-1])

    expiries: Iterable[str] = tk.options or []
    if not expiries:
        raise RuntimeError(f"Aucune échéance d'options disponible pour {ticker!r} sur Yahoo.")

    if max_expiries is not None:
        expiries = list(expiries)[:max_expiries]

    rows: List[OptionRow] = []
    now = datetime.now(timezone.utc)

    for expiry in expiries:
        T = year_fraction(expiry, now=now)
        if T <= 0.0:
            continue

        chain = tk.option_chain(expiry)

        if include_calls:
            rows.extend(
                _rows_from_chain_side(
                    chain.calls,
                    spot=spot,
                    maturity=T,
                    rate=rate,
                )
            )

        if include_puts:
            rows.extend(
                _rows_from_chain_side(
                    chain.puts,
                    spot=spot,
                    maturity=T,
                    rate=rate,
                )
            )

        if max_rows is not None and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break

    if not rows:
        raise RuntimeError("Aucune option exploitable (iv ou strikes manquants).")

    return rows


def _rows_from_chain_side(df: pd.DataFrame, spot: float, maturity: float, rate: float) -> List[OptionRow]:
    """Convertit un DataFrame d'options yfinance (calls ou puts) en OptionRow."""
    rows: List[OptionRow] = []

    # Colonnes typiques : ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice',
    # 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest',
    # 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
    for _, opt in df.iterrows():
        strike = float(opt.get("strike", np.nan))
        iv = float(opt.get("impliedVolatility", np.nan))
        last_price = float(opt.get("lastPrice", np.nan))

        if not np.isfinite(strike) or strike <= 0:
            continue
        if not np.isfinite(iv) or iv <= 0:
            continue
        if not np.isfinite(last_price) or last_price <= 0:
            continue

        s_over_k = spot / strike

        rows.append(
            OptionRow(
                s_over_k=s_over_k,
                maturity=maturity,
                volatility=iv,
                rate=rate,
                label=last_price,
                spot=spot,
                strike=strike,
                w0=1.0,
                w1=0.0,
                w2=0.0,
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Récupère des options via yfinance et génère un CSV "
            "compatible avec Basket/data/train.csv."
        )
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker Yahoo Finance (ex: SPY, ^SPX, ^STOXX50E).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Chemin du CSV de sortie. "
            "Par défaut: Basket/data/train_yf_<ticker>.csv (relatif au dépôt)."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.01,
        help="Taux sans risque utilisé dans la colonne 'Rate' (défaut: 0.01).",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=5,
        help="Nombre maximum d'échéances à récupérer (défaut: 5).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Nombre maximum de lignes à garder dans le CSV (défaut: 2000).",
    )
    parser.add_argument(
        "--no-calls",
        action="store_true",
        help="N'inclut pas les calls (uniquement les puts).",
    )
    parser.add_argument(
        "--no-puts",
        action="store_true",
        help="N'inclut pas les puts (uniquement les calls).",
    )

    args = parser.parse_args()

    include_calls = not args.no_calls
    include_puts = not args.no_puts

    if not include_calls and not include_puts:
        parser.error("Vous devez inclure au moins les calls ou les puts.")

    rows = fetch_option_rows(
        ticker=args.ticker,
        rate=args.rate,
        max_expiries=args.max_expiries,
        include_puts=include_puts,
        include_calls=include_calls,
        max_rows=args.max_rows,
    )

    df = pd.DataFrame([row.to_dict() for row in rows])
    df = df[
        [
            "S/K",
            "Maturity",
            "Volatility",
            "Rate",
            "Labels",
            "Prices",
            "Strikes",
            "Weight_0",
            "Weight_1",
            "Weight_2",
        ]
    ]

    if args.output is None:
        # Par défaut : Basket/data/train_yf_<ticker>.csv
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / "Basket" / "data" / f"train_yf_{args.ticker}.csv"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✔ Fichier généré : {output_path} ({len(df)} lignes)")


if __name__ == "__main__":
    main()

