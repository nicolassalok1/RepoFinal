#!/usr/bin/env python3
"""
Plot 3D implied volatility surface from a CSV file with columns:
    K, T, iv

- K : strike
- T : maturity (en années)
- iv : implied volatility (décimal, pas en %)

Le script :
1) charge le CSV
2) construit une grille régulière (K, T)
3) moyenne les iv quand plusieurs points tombent dans la même case
4) affiche une surface 3D + une heatmap 2D
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, pour activer projection='3d'


def build_grid(
    df: pd.DataFrame,
    S0: float,
    n_K: int = 200,
    n_T: int = 200,
    K_span: float = 100.0,
    T_min: float = 0.0,
    T_max: float = 2.0,
):
    """
    Construit une grille régulière en (K, T) et moyenne les iv dans chaque cellule.

    On utilise:
        K in [S0 - K_span, S0 + K_span]
        T in [T_min, T_max]

    Paramètres
    ---------
    df : DataFrame
        Doit contenir au minimum les colonnes 'K', 'T', 'iv'.
    S0 : float
        Spot actuel.
    n_K : int
        Nombre de points de grille en K.
    n_T : int
        Nombre de points de grille en T.
    K_span : float
        Intervalle de part et d'autre de S0 (en absolu).
    T_min, T_max : float
        Bornes pour les maturités.

    Retourne
    --------
    K_grid : ndarray (n_T, n_K)
    T_grid : ndarray (n_T, n_K)
    iv_grid : ndarray (n_T, n_K) avec NaN quand aucune donnée
    """

    # Définition de la grille
    K_min = S0 - K_span
    K_max = S0 + K_span

    K_vals = np.linspace(K_min, K_max, n_K)
    T_vals = np.linspace(T_min, T_max, n_T)

    # Filtre les données dans le range de la grille
    df = df.copy()
    df = df[(df["K"] >= K_min) & (df["K"] <= K_max)]
    df = df[(df["T"] >= T_min) & (df["T"] <= T_max)]

    if df.empty:
        raise ValueError(
            "Après filtrage, il ne reste plus aucun point dans le domaine de la grille."
        )

    # Indices de grille (0..n_K-1 et 0..n_T-1)
    df["K_idx"] = np.searchsorted(K_vals, df["K"], side="left")
    df["T_idx"] = np.searchsorted(T_vals, df["T"], side="left")

    # Corrige les indices qui tombent à n_K ou n_T (bord droit)
    df["K_idx"] = df["K_idx"].clip(0, n_K - 1)
    df["T_idx"] = df["T_idx"].clip(0, n_T - 1)

    # Groupby sur (T_idx, K_idx) et moyenne de iv
    grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

    # Initialise la grille avec NaN
    iv_grid = np.full((n_T, n_K), np.nan, dtype=float)

    # Remplit la grille
    for _, row in grouped.iterrows():
        ti = int(row["T_idx"])
        ki = int(row["K_idx"])
        iv_grid[ti, ki] = row["iv"]

    # Crée les grilles de coordonnées
    K_grid, T_grid = np.meshgrid(K_vals, T_vals)

    return K_grid, T_grid, iv_grid


def plot_surface(K_grid, T_grid, iv_grid, title_suffix=""):
    """Affiche une surface 3D et une heatmap 2D de iv(K,T)."""
    fig = plt.figure(figsize=(12, 5))

    # --- Surface 3D ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # Remplace les NaN par la moyenne globale des IV pour éviter les trous
    iv_flat = iv_grid[~np.isnan(iv_grid)]
    if iv_flat.size == 0:
        raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
    iv_mean = iv_flat.mean()
    iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)

    surf = ax3d.plot_surface(
        K_grid,
        T_grid,
        iv_grid_filled,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        antialiased=True,
        cmap="viridis",
    )

    ax3d.set_xlabel("Strike K")
    ax3d.set_ylabel("Maturity T (years)")
    ax3d.set_zlabel("Implied vol")
    ax3d.set_title(f"IV surface 3D{title_suffix}")

    fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

    # --- Heatmap 2D ---
    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(
        iv_grid_filled,
        extent=[K_grid.min(), K_grid.max(), T_grid.min(), T_grid.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax2d.set_xlabel("Strike K")
    ax2d.set_ylabel("Maturity T (years)")
    ax2d.set_title(f"IV heatmap{title_suffix}")
    fig.colorbar(im, ax=ax2d, label="iv")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D IV surface from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with columns: K, T, iv.",
    )
    parser.add_argument(
        "--S0",
        type=float,
        required=True,
        help="Spot price S0 (used to center K range [S0-100, S0+100] by default).",
    )
    parser.add_argument(
        "--nK",
        type=int,
        default=200,
        help="Number of grid points in K direction (default: 200).",
    )
    parser.add_argument(
        "--nT",
        type=int,
        default=200,
        help="Number of grid points in T direction (default: 200).",
    )
    parser.add_argument(
        "--Kspan",
        type=float,
        default=100.0,
        help="Half-width of K interval around S0 (default: 100 => [S0-100,S0+100]).",
    )
    parser.add_argument(
        "--Tmax",
        type=float,
        default=2.0,
        help="Max maturity T (min is fixed to 0, default: 2.0).",
    )

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    required_cols = {"K", "T", "iv"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in CSV: {missing}")

    K_grid, T_grid, iv_grid = build_grid(
        df,
        S0=args.S0,
        n_K=args.nK,
        n_T=args.nT,
        K_span=args.Kspan,
        T_min=0.0,
        T_max=args.Tmax,
    )

    title_suffix = f" (S0={args.S0})"
    plot_surface(K_grid, T_grid, iv_grid, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
