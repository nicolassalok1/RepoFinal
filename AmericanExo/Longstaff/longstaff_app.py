import streamlit as st

from Longstaff.option import Option
from Longstaff.pricing import (
    black_scholes_merton,
    crr_pricing,
    monte_carlo_simulation,
    monte_carlo_simulation_LS,
)
from Longstaff.process import GeometricBrownianMotion, HestonProcess


st.title("Longstaff–Schwartz & autres pricers")

st.sidebar.header("Paramètres de l'option")
S0 = st.sidebar.number_input("S0 (spot)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("K (strike)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("T (maturité en années)", value=1.0, min_value=0.01)
is_call = st.sidebar.selectbox("Type d'option", ["Call", "Put"]) == "Call"

st.sidebar.header("Paramètres du modèle")
mu = st.sidebar.number_input("μ (drift, utilisé pour MC)", value=0.05)
sigma = st.sidebar.number_input("σ (volatilité)", value=0.2, min_value=0.0001)
r = st.sidebar.number_input("Taux sans risque r (BSM / CRR)", value=0.05)

st.sidebar.header("Paramètres Monte Carlo / Arbre")
n_paths = st.sidebar.number_input("Nombre de trajectoires Monte Carlo (n)", value=10_000, min_value=100)
m_steps = st.sidebar.number_input("Nombre de pas de temps (m)", value=50, min_value=1)
n_tree = st.sidebar.number_input("Nombre de pas de l'arbre CRR", value=250, min_value=10)

st.sidebar.header("Processus sous-jacent")
process_type = st.sidebar.selectbox("Processus", ["Geometric Brownian Motion", "Heston"])

if process_type == "Geometric Brownian Motion":
    process = GeometricBrownianMotion(mu=mu, sigma=sigma)
    v0 = None
else:
    st.sidebar.subheader("Paramètres Heston")
    kappa = st.sidebar.number_input("κ (vitesse de rappel)", value=2.0)
    theta = st.sidebar.number_input("θ (variance de long terme)", value=0.04)
    eta = st.sidebar.number_input("η (vol de la variance)", value=0.5)
    rho = st.sidebar.number_input("ρ (corrélation)", value=-0.7, min_value=-0.99, max_value=0.99)
    v0 = st.sidebar.number_input("v0 (variance initiale)", value=0.04, min_value=0.0001)
    process = HestonProcess(mu=mu, kappa=kappa, theta=theta, eta=eta, rho=rho)

option = Option(s0=S0, T=T, K=K, v0=v0, call=is_call)

st.subheader("Méthode de valorisation")
method = st.selectbox(
    "Choisir une méthode",
    [
        "Monte Carlo classique",
        "Monte Carlo Longstaff–Schwartz (américaine)",
        "Black–Scholes–Merton (européenne)",
        "Arbre CRR (américaine)",
    ],
)

if st.button("Calculer le prix"):
    if method == "Monte Carlo classique":
        price = monte_carlo_simulation(option=option, process=process, n=int(n_paths), m=int(m_steps))
        st.write(f"**Prix Monte Carlo**: {price:.4f}")

    elif method == "Monte Carlo Longstaff–Schwartz (américaine)":
        price = monte_carlo_simulation_LS(option=option, process=process, n=int(n_paths), m=int(m_steps))
        if price is not None:
            st.write(f"**Prix Longstaff–Schwartz**: {price:.4f}")

    elif method == "Black–Scholes–Merton (européenne)":
        price = black_scholes_merton(r=r, sigma=sigma, option=option)
        st.write(f"**Prix BSM**: {price:.4f}")

    elif method == "Arbre CRR (américaine)":
        price = crr_pricing(r=r, sigma=sigma, option=option, n=int(n_tree))
        st.write(f"**Prix CRR**: {price:.4f}")

