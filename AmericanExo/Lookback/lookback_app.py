import streamlit as st

from Lookback.barrier_call import barrier_call_option
from Lookback.european_call import european_call_option
from Lookback.lookback_call import lookback_call_option


st.title("Lookback, Barrier & European Options")

st.sidebar.header("Paramètres communs")
S0 = st.sidebar.number_input("S0 (spot)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("T (maturité en années)", value=1.0, min_value=0.01)
t0 = st.sidebar.number_input("t (temps courant)", value=0.0, min_value=0.0, max_value=float(T))
r = st.sidebar.number_input("Taux sans risque r", value=0.05)
sigma = st.sidebar.number_input("Volatilité σ", value=0.2, min_value=0.0001)

st.sidebar.header("Paramètres supplémentaires")
K = st.sidebar.number_input("K (strike, pour Euro & Barrière)", value=100.0, min_value=0.01)
B = st.sidebar.number_input("B (barrière up-and-out, pour Barrière)", value=120.0, min_value=0.01)

st.sidebar.header("Schéma numérique (PDE / MC)")
n_iters = st.sidebar.number_input("Itérations Monte Carlo", value=10_000, min_value=100)
n_t = st.sidebar.number_input("Pas de temps PDE (n_t)", value=200, min_value=10)
n_s = st.sidebar.number_input("Pas d'espace PDE (n_s)", value=200, min_value=10)

tab_eu, tab_barrier, tab_lookback = st.tabs(
    ["European call", "Barrier up-and-out call", "Lookback floating call"]
)

with tab_eu:
    st.subheader("European call option")
    euro = european_call_option(T=T, t=t0, S0=S0, K=K, r=r, sigma=sigma)

    method_eu = st.selectbox(
        "Méthode de pricing (Euro)",
        ["Exacte (BSM)", "Monte Carlo", "PDE Crank–Nicolson"],
        key="method_eu",
    )

    if st.button("Calculer (European)", key="btn_eu"):
        if method_eu == "Exacte (BSM)":
            price = euro.price_exact()
            st.write(f"**Prix exact**: {price:.6f}")
        elif method_eu == "Monte Carlo":
            price = euro.price_monte_carlo(int(n_iters))
            st.write(f"**Prix Monte Carlo**: {price:.6f}")
        else:
            euro.price_pde(int(n_t), int(n_s))
            price = euro.get_pde_result(S0)
            st.write(f"**Prix PDE**: {price:.6f}")

with tab_barrier:
    st.subheader("Barrier up-and-out call option")
    barrier = barrier_call_option(T=T, t=t0, S0=S0, K=K, B=B, r=r, sigma=sigma)

    method_barrier = st.selectbox(
        "Méthode de pricing (Barrière)",
        ["Exacte (fermée)", "Monte Carlo", "PDE Crank–Nicolson"],
        key="method_barrier",
    )

    if st.button("Calculer (Barrière)", key="btn_barrier"):
        if method_barrier == "Exacte (fermée)":
            price = barrier.price_exact()
            st.write(f"**Prix exact barrière**: {price:.6f}")
        elif method_barrier == "Monte Carlo":
            price = barrier.price_monte_carlo(int(n_iters))
            st.write(f"**Prix Monte Carlo barrière**: {price:.6f}")
        else:
            barrier.price_pde(int(n_t), int(n_s))
            price = barrier.get_pde_result(S0)
            st.write(f"**Prix PDE barrière**: {price:.6f}")

with tab_lookback:
    st.subheader("Lookback call option (floating strike)")
    lookback = lookback_call_option(T=T, t=t0, S0=S0, r=r, sigma=sigma)

    method_lb = st.selectbox(
        "Méthode de pricing (Lookback)",
        ["Exacte", "Monte Carlo", "PDE Crank–Nicolson"],
        key="method_lb",
    )

    if st.button("Calculer (Lookback)", key="btn_lb"):
        if method_lb == "Exacte":
            price = lookback.price_exact()
            st.write(f"**Prix exact lookback**: {price:.6f}")
        elif method_lb == "Monte Carlo":
            price = lookback.price_monte_carlo(int(n_iters))
            st.write(f"**Prix Monte Carlo lookback**: {price:.6f}")
        else:
            lookback.price_pde(int(n_t), int(n_s))
            # z = 1 pour t=0, voir doc de price_exact
            price = lookback.get_pde_result(z=1.0)
            st.write(f"**Prix PDE lookback**: {price:.6f}")

