"""
Application Streamlit consolidant tous les notebooks FYPY
Architecture: onglets pour chaque TP, sidebar pour paramètres communs
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import os

# Ajouter le chemin de fypy
sys.path.insert(0, os.path.dirname(__file__))

# Imports fypy avec gestion d'erreurs
try:
    from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate, InterpolatedDiscountCurve
    from fypy.termstructures.EquityForward import EquityForward
    from fypy.pricing.analytical.black_scholes import black_scholes_price
    from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
    from fypy.model.levy import VarianceGamma, BilateralGamma, BlackScholes, CMGY, KouJD, MertonJD, NIG
    from fypy.model.sv import Heston
    FYPY_CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"Erreur d'import fypy core: {e}")
    FYPY_CORE_AVAILABLE = False

# Imports optionnels
try:
    from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76
    IMPLIEDVOL_AVAILABLE = True
except ImportError:
    IMPLIEDVOL_AVAILABLE = False

try:
    from fypy.pricing.analytical.asian_approx import asian_price_approx
    ASIAN_AVAILABLE = True
except ImportError:
    ASIAN_AVAILABLE = False

try:
    from fypy.pricing.lattice.BinomialLattice import BinomialLattice
    LATTICE_AVAILABLE = True
except ImportError:
    LATTICE_AVAILABLE = False

try:
    from fypy.pricing.montecarlo.StochasticProcess import StochasticProcess
    from fypy.process.blackscholes import GeometricBrownianMotion
    MONTECARLO_AVAILABLE = True
except ImportError:
    MONTECARLO_AVAILABLE = False

try:
    from fypy.market.MarketSlice import MarketSlice
    from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
    from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator
    CALIBRATE_AVAILABLE = True
except ImportError:
    CALIBRATE_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="FYPY - Financial Python Library",
    page_icon="📈",
    layout="wide"
)

# Titre principal
st.title("📈 FYPY - Librairie de Pricing d'Options")
st.markdown("Application interactive consolidant tous les TPs FYPY")

# ============================================================================
# SIDEBAR - PARAMETRES COMMUNS
# ============================================================================
st.sidebar.header("⚙️ Paramètres Communs")

st.sidebar.subheader("Marché")
S0 = st.sidebar.number_input("Prix spot S₀", value=100.0, min_value=1.0, step=1.0)
r = st.sidebar.number_input("Taux sans risque r", value=0.03, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
q = st.sidebar.number_input("Dividende continu q", value=0.01, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")

st.sidebar.subheader("Option")
T = st.sidebar.slider("Maturité T (années)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
K = st.sidebar.number_input("Strike K", value=100.0, min_value=1.0, step=1.0)
is_call = st.sidebar.selectbox("Type d'option", ["Call", "Put"]) == "Call"

st.sidebar.subheader("Volatilité")
sigma = st.sidebar.slider("Volatilité σ", min_value=0.05, max_value=1.0, value=0.2, step=0.05)

# Construire les courbes communes
disc_curve = DiscountCurve_ConstRate(rate=r)
div_disc = DiscountCurve_ConstRate(rate=q)
fwd_curve = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

# ============================================================================
# ONGLETS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "TP1: Architecture",
    "TP2: Black-Scholes",
    "TP3: Lévy & Fourier",
    "TP4: Heston",
    "TP5: Exotiques",
    "TP6: Binomial",
    "TP7: Monte Carlo",
    "TP8: Vol Surfaces",
    "TP9: Calibration",
    "TP10: Dates & Data"
])

# ============================================================================
# TP1: ARCHITECTURE ET TERM STRUCTURES
# ============================================================================
with tab1:
    st.header("TP 1 - Architecture de fypy et courbes de taux")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courbe d'actualisation")
        T_range = np.linspace(0.0, 10.0, 100)
        D_T = disc_curve(T_range)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_range, D_T)
        ax.set_title(f"Facteur d'actualisation D(T) = exp(-r·T) avec r={r}")
        ax.set_xlabel("Échéance T (années)")
        ax.set_ylabel("D(T)")
        ax.grid(True)
        st.pyplot(fig)
        
        # Taux implicites
        T_test = np.array([0.5, 1.0, 5.0, 10.0])
        r_implied = disc_curve.implied_rate(T_test)
        st.write("**Taux implicites:**")
        for T_i, r_i in zip(T_test, r_implied):
            st.write(f"T = {T_i} ans → r(T) = {r_i:.4f}")
    
    with col2:
        st.subheader("Courbe forward d'equity")
        F_T = fwd_curve(T_range)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_range, F_T)
        ax.set_title(f"Forward F₀(T) = S₀·exp((r-q)·T)")
        ax.set_xlabel("Échéance T (années)")
        ax.set_ylabel("F₀(T)")
        ax.grid(True)
        st.pyplot(fig)
        
        st.write(f"**Forward à T={T}:** {fwd_curve(T):.2f}")
        st.write(f"**Théorique:** {S0 * np.exp((r - q) * T):.2f}")

# ============================================================================
# TP2: BLACK-SCHOLES ET MONTE CARLO
# ============================================================================
with tab2:
    st.header("TP 2 - Modèle de Black-Scholes et Monte Carlo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pricing Black-Scholes")
        
        # Prix BS
        prix_bs = black_scholes_price(
            S=S0, K=K, is_call=is_call, 
            sigma=sigma, T=T, r=r, q=q
        )
        
        st.metric("Prix Black-Scholes", f"{prix_bs:.4f}")
        
        # Greeks
        st.write("**Greeks (approximation numérique):**")
        dS = 0.01
        prix_up = black_scholes_price(S=S0+dS, K=K, is_call=is_call, sigma=sigma, T=T, r=r, q=q)
        prix_down = black_scholes_price(S=S0-dS, K=K, is_call=is_call, sigma=sigma, T=T, r=r, q=q)
        delta = (prix_up - prix_down) / (2*dS)
        gamma = (prix_up - 2*prix_bs + prix_down) / (dS**2)
        
        st.write(f"Delta: {delta:.4f}")
        st.write(f"Gamma: {gamma:.6f}")
        
        # Smile de volatilité
        st.subheader("Smile de volatilité implicite")
        strikes_range = np.linspace(0.7*S0, 1.3*S0, 50)
        prices = [black_scholes_price(S=S0, K=k, is_call=is_call, sigma=sigma, T=T, r=r, q=q) 
                  for k in strikes_range]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(strikes_range/S0, [sigma]*len(strikes_range))
        ax.set_title("Volatilité implicite (BS: flat)")
        ax.set_xlabel("K/S₀")
        ax.set_ylabel("Vol implicite")
        ax.grid(True)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Simulation Monte Carlo")
        
        n_paths = st.slider("Nombre de chemins", 1000, 100000, 10000, step=1000)
        
        # Monte Carlo pricing
        np.random.seed(42)
        Z = np.random.randn(n_paths)
        drift = (r - q - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T) * Z
        S_T = S0 * np.exp(drift + diffusion)
        
        if is_call:
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        
        prix_mc = np.exp(-r * T) * np.mean(payoff)
        err_std = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)
        
        st.metric("Prix Monte Carlo", f"{prix_mc:.4f}", f"±{1.96*err_std:.4f} (95%)")
        st.metric("Erreur vs BS", f"{abs(prix_mc - prix_bs):.4f}")
        
        # Histogramme des prix finaux
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(S_T, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(K, color='red', linestyle='--', label=f'Strike K={K}')
        ax.axvline(S0*np.exp((r-q)*T), color='green', linestyle='--', label='Forward')
        ax.set_title("Distribution de S_T (Monte Carlo)")
        ax.set_xlabel("Prix final S_T")
        ax.set_ylabel("Fréquence")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ============================================================================
# TP3: PROCESSUS DE LÉVY ET PRICING FOURIER
# ============================================================================
with tab3:
    st.header("TP 3 - Processus de Lévy et pricing par Fourier (PROJ)")
    
    st.markdown("""
    Les processus de Lévy permettent de capturer les sauts et l'asymétrie observés sur les marchés.
    Le pricing par méthode PROJ (Projection) utilise la transformée de Fourier.
    """)
    
    # Sélection du modèle
    levy_model = st.selectbox("Modèle de Lévy", 
                               ["Black-Scholes", "Variance Gamma", "Bilateral Gamma", "NIG", "Merton JD", "Kou JD"])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du modèle")
        
        if levy_model == "Black-Scholes":
            model = BlackScholes(sigma=sigma, forwardCurve=fwd_curve, discountCurve=disc_curve)
            st.write(f"σ = {sigma}")
            
        elif levy_model == "Variance Gamma":
            theta = st.slider("theta (asymétrie)", -0.5, 0.5, -0.1, 0.05)
            nu = st.slider("nu (kurtosis)", 0.1, 2.0, 0.6, 0.1)
            model = VarianceGamma(sigma=sigma, theta=theta, nu=nu, 
                                 forwardCurve=fwd_curve, discountCurve=disc_curve)
            
        elif levy_model == "Bilateral Gamma":
            alpha_p = st.slider("alpha_p", 0.5, 3.0, 1.18, 0.1)
            lambda_p = st.slider("lambda_p", 1.0, 20.0, 10.57, 0.5)
            alpha_m = st.slider("alpha_m", 0.5, 3.0, 1.44, 0.1)
            lambda_m = st.slider("lambda_m", 1.0, 20.0, 5.57, 0.5)
            model = BilateralGamma(alpha_p=alpha_p, lambda_p=lambda_p, 
                                  alpha_m=alpha_m, lambda_m=lambda_m,
                                  forwardCurve=fwd_curve, discountCurve=disc_curve)
            
        elif levy_model == "NIG":
            alpha = st.slider("alpha", 1.0, 20.0, 10.0, 1.0)
            beta = st.slider("beta", -10.0, 10.0, -3.0, 1.0)
            delta = st.slider("delta", 0.1, 2.0, 0.4, 0.1)
            model = NIG(alpha=alpha, beta=beta, delta=delta,
                       forwardCurve=fwd_curve, discountCurve=disc_curve)
            
        elif levy_model == "Merton JD":
            merton_lambda = st.slider("lambda (intensité)", 0.1, 5.0, 0.4, 0.1)
            mu_j = st.slider("mu_j (taille saut)", -0.5, 0.5, -0.1, 0.05)
            sig_j = st.slider("sig_j (vol saut)", 0.1, 0.5, 0.2, 0.05)
            model = MertonJD(sigma=sigma, lam=merton_lambda, mu_j=mu_j, sig_j=sig_j,
                           forwardCurve=fwd_curve, discountCurve=disc_curve)
            
        else:  # Kou JD
            kou_lambda = st.slider("lambda (intensité)", 0.1, 5.0, 1.0, 0.1)
            p_up = st.slider("p_up (prob saut haut)", 0.0, 1.0, 0.4, 0.05)
            eta_p = st.slider("eta_p", 1.0, 50.0, 20.0, 1.0)
            eta_m = st.slider("eta_m", 1.0, 50.0, 25.0, 1.0)
            model = KouJD(sigma=sigma, lam=kou_lambda, p_up=p_up, 
                         eta_p=eta_p, eta_m=eta_m,
                         forwardCurve=fwd_curve, discountCurve=disc_curve)
    
    with col2:
        st.subheader("Pricing et Smile de volatilité")
        
        # Paramètres PROJ
        N_proj = st.slider("N (grille PROJ)", 256, 2048, 1024, step=256)
        L = st.slider("L (support)", 6.0, 16.0, 10.0, step=1.0)
        
        # Pricer
        pricer = ProjEuropeanPricer(model=model, N=N_proj, L=L)
        
        # Grille de strikes
        strikes_range = np.linspace(0.6*S0, 1.4*S0, 40)
        is_calls = np.ones(len(strikes_range), dtype=bool) if is_call else np.zeros(len(strikes_range), dtype=bool)
        
        try:
            prices = pricer.price_strikes(T=T, strikes=strikes_range, is_calls=is_calls)
            
            # Prix pour le strike sélectionné
            idx_K = np.argmin(np.abs(strikes_range - K))
            prix_proj = prices[idx_K]
            st.metric(f"Prix {levy_model} (PROJ)", f"{prix_proj:.4f}")
            
            # Volatilité implicite
            if IMPLIEDVOL_AVAILABLE:
                ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd_curve)
                try:
                    vols_impl = ivc.imply_vols(strikes=strikes_range, prices=prices, 
                                              is_calls=is_calls, ttm=T)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Prix
                    ax1.plot(strikes_range/S0, prices, 'b-', linewidth=2)
                    ax1.axvline(K/S0, color='red', linestyle='--', alpha=0.7, label=f'K={K}')
                    ax1.set_xlabel("K/S₀")
                    ax1.set_ylabel("Prix")
                    ax1.set_title(f"Prix des options - {levy_model}")
                    ax1.grid(True)
                    ax1.legend()
                    
                    # Smile
                    ax2.plot(strikes_range/S0, vols_impl, 'g-', linewidth=2)
                    ax2.axhline(sigma, color='gray', linestyle='--', alpha=0.5, label=f'σ={sigma}')
                    ax2.axvline(K/S0, color='red', linestyle='--', alpha=0.7)
                    ax2.set_xlabel("K/S₀")
                    ax2.set_ylabel("Volatilité implicite")
                    ax2.set_title("Smile de volatilité")
                    ax2.grid(True)
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Impossible de calculer les vols implicites: {e}")
            else:
                # Si pas de vol implicite, afficher seulement les prix
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(strikes_range/S0, prices, 'b-', linewidth=2)
                ax.axvline(K/S0, color='red', linestyle='--', alpha=0.7, label=f'K={K}')
                ax.set_xlabel("K/S₀")
                ax.set_ylabel("Prix")
                ax.set_title(f"Prix des options - {levy_model}")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                st.info("Module de vol implicite non disponible")
                
        except Exception as e:
            st.error(f"Erreur lors du pricing: {e}")

# ============================================================================
# TP4: VOLATILITÉ STOCHASTIQUE (HESTON)
# ============================================================================
with tab4:
    st.header("TP 4 - Volatilité stochastique (Heston)")
    
    st.markdown("""
    Le modèle de Heston décrit une variance stochastique corrélée avec le sous-jacent.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres Heston")
        v0 = st.slider("v₀ (variance initiale)", 0.01, 0.5, 0.04, 0.01)
        theta_h = st.slider("θ (variance long terme)", 0.01, 0.5, 0.04, 0.01)
        kappa = st.slider("κ (vitesse retour moy.)", 0.1, 5.0, 1.5, 0.1)
        sigma_v = st.slider("σᵥ (vol de vol)", 0.1, 1.0, 0.3, 0.05)
        rho = st.slider("ρ (corrélation S-v)", -1.0, 1.0, -0.7, 0.05)
        
        st.write(f"**σ₀ = √v₀ = {np.sqrt(v0):.3f}**")
        st.write(f"**σ∞ = √θ = {np.sqrt(theta_h):.3f}**")
    
    with col2:
        st.subheader("Pricing et Smile Heston")
        
        # Modèle Heston
        model_heston = Heston(v0=v0, theta=theta_h, kappa=kappa, 
                             sigma_v=sigma_v, rho=rho,
                             forwardCurve=fwd_curve, discountCurve=disc_curve)
        
        # Pricer
        N_proj = 1024
        L = 12.0
        pricer_heston = ProjEuropeanPricer(model=model_heston, N=N_proj, L=L)
        
        # Grille de strikes
        strikes_range = np.linspace(0.6*S0, 1.4*S0, 40)
        is_calls = np.ones(len(strikes_range), dtype=bool) if is_call else np.zeros(len(strikes_range), dtype=bool)
        
        try:
            prices_heston = pricer_heston.price_strikes(T=T, strikes=strikes_range, is_calls=is_calls)
            
            # Prix pour le strike
            idx_K = np.argmin(np.abs(strikes_range - K))
            prix_heston = prices_heston[idx_K]
            st.metric("Prix Heston (PROJ)", f"{prix_heston:.4f}")
            
            # Comparaison avec BS à vol=sqrt(v0)
            model_bs_comp = BlackScholes(sigma=np.sqrt(v0), forwardCurve=fwd_curve, discountCurve=disc_curve)
            pricer_bs_comp = ProjEuropeanPricer(model=model_bs_comp, N=N_proj, L=L)
            prices_bs_comp = pricer_bs_comp.price_strikes(T=T, strikes=strikes_range, is_calls=is_calls)
            
            # Vols implicites
            if IMPLIEDVOL_AVAILABLE:
                ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd_curve)
                
                try:
                    vols_heston = ivc.imply_vols(strikes=strikes_range, prices=prices_heston, 
                                                is_calls=is_calls, ttm=T)
                    vols_bs_comp = ivc.imply_vols(strikes=strikes_range, prices=prices_bs_comp,
                                                 is_calls=is_calls, ttm=T)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Prix
                    ax1.plot(strikes_range/S0, prices_heston, 'b-', linewidth=2, label='Heston')
                    ax1.plot(strikes_range/S0, prices_bs_comp, 'g--', linewidth=2, label=f'BS σ={np.sqrt(v0):.2f}')
                    ax1.axvline(K/S0, color='red', linestyle='--', alpha=0.7)
                    ax1.set_xlabel("K/S₀")
                    ax1.set_ylabel("Prix")
                    ax1.set_title("Prix des options")
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Smile
                    ax2.plot(strikes_range/S0, vols_heston, 'b-', linewidth=2, label='Heston')
                    ax2.plot(strikes_range/S0, vols_bs_comp, 'g--', linewidth=2, label='BS')
                    ax2.axvline(K/S0, color='red', linestyle='--', alpha=0.7)
                    ax2.set_xlabel("K/S₀")
                    ax2.set_ylabel("Volatilité implicite")
                    ax2.set_title("Smile de volatilité")
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Impossible de calculer les vols implicites: {e}")
            else:
                # Si pas de module vol implicite, afficher seulement les prix
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(strikes_range/S0, prices_heston, 'b-', linewidth=2, label='Heston')
                ax.plot(strikes_range/S0, prices_bs_comp, 'g--', linewidth=2, label=f'BS σ={np.sqrt(v0):.2f}')
                ax.axvline(K/S0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel("K/S₀")
                ax.set_ylabel("Prix")
                ax.set_title("Prix des options")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                st.info("Module de vol implicite non disponible")
                
        except Exception as e:
            st.error(f"Erreur lors du pricing Heston: {e}")

# ============================================================================
# TP5: OPTIONS EXOTIQUES
# ============================================================================
with tab5:
    st.header("TP 5 - Pricing analytique d'options exotiques")
    
    if not ASIAN_AVAILABLE:
        st.warning("Module asiatique non disponible. Utilisation d'une approximation simple.")
    
    exotic_type = st.selectbox("Type d'option exotique", 
                               ["Asiatique (arithmétique)", "Asiatique (géométrique)"])
    
    if "Asiatique" in exotic_type:
        st.subheader("Option Asiatique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_observations = st.slider("Nombre d'observations", 10, 252, 52)
            st.write(f"Observations tous les {T*252/n_observations:.1f} jours")
            
            # Approximation simple pour asiatique arithmétique
            # Prix plus faible qu'européenne car moyenne réduit volatilité
            if ASIAN_AVAILABLE:
                st.markdown("**Approximation de Vorst** pour moyenne arithmétique" if "arithmétique" in exotic_type else "**Formule exacte** pour moyenne géométrique")
                try:
                    prix_asian = asian_price_approx(S=S0, K=K, is_call=is_call, T=T,
                                                   r=r, q=q, sigma=sigma, n=n_observations,
                                                   asian_type='arithmetic' if 'arithmétique' in exotic_type else 'geometric')
                    st.metric(f"Prix option asiatique", f"{prix_asian:.4f}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    prix_asian = None
            else:
                # Approximation simple basée sur réduction de volatilité
                sigma_asian = sigma / np.sqrt(3)  # Approximation simplifiée
                prix_asian = black_scholes_price(S=S0, K=K, is_call=is_call,
                                                sigma=sigma_asian, T=T, r=r, q=q)
                st.info("Utilisation d'une approximation simple (σ_asian ≈ σ/√3)")
                st.metric(f"Prix option asiatique (approx)", f"{prix_asian:.4f}")
            
            # Comparaison avec européenne
            prix_euro = black_scholes_price(S=S0, K=K, is_call=is_call, 
                                           sigma=sigma, T=T, r=r, q=q)
            st.metric("Prix européenne", f"{prix_euro:.4f}")
            if prix_asian:
                st.metric("Différence", f"{prix_asian - prix_euro:.4f}")
        
        with col2:
            st.subheader("Comparaison par strike")
            
            strikes_asian = np.linspace(0.7*S0, 1.3*S0, 30)
            prices_asian = []
            prices_euro = []
            
            for k in strikes_asian:
                # Prix européen
                p_euro = black_scholes_price(S=S0, K=k, is_call=is_call,
                                            sigma=sigma, T=T, r=r, q=q)
                prices_euro.append(p_euro)
                
                # Prix asiatique
                if ASIAN_AVAILABLE:
                    try:
                        p_asian = asian_price_approx(S=S0, K=k, is_call=is_call, T=T,
                                                    r=r, q=q, sigma=sigma, n=n_observations,
                                                    asian_type='arithmetic' if 'arithmétique' in exotic_type else 'geometric')
                        prices_asian.append(p_asian)
                    except:
                        prices_asian.append(np.nan)
                else:
                    # Approximation
                    sigma_asian = sigma / np.sqrt(3)
                    p_asian = black_scholes_price(S=S0, K=k, is_call=is_call,
                                                 sigma=sigma_asian, T=T, r=r, q=q)
                    prices_asian.append(p_asian)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(strikes_asian/S0, prices_asian, 'b-', linewidth=2, label='Asiatique')
            ax.plot(strikes_asian/S0, prices_euro, 'g--', linewidth=2, label='Européenne')
            ax.axvline(K/S0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel("K/S₀")
            ax.set_ylabel("Prix")
            ax.set_title(f"Comparaison {exotic_type}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# ============================================================================
# TP6: LATTICE BINOMIAL
# ============================================================================
with tab6:
    st.header("TP 6 - Arbre binomial vs Black-Scholes")
    
    if not LATTICE_AVAILABLE:
        st.warning("Module lattice binomial non disponible. Affichage de la formule BS uniquement.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres de l'arbre")
        n_steps = st.slider("Nombre de pas", 10, 500, 100, step=10)
        
        st.write(f"Δt = {T/n_steps:.4f} années")
        st.write(f"≈ {T*252/n_steps:.1f} jours par pas")
    
    with col2:
        st.subheader("Pricing")
        
        # Prix BS
        prix_bs = black_scholes_price(S=S0, K=K, is_call=is_call,
                                     sigma=sigma, T=T, r=r, q=q)
        st.metric("Prix Black-Scholes", f"{prix_bs:.6f}")
        
        # Prix binomial
        if LATTICE_AVAILABLE:
            try:
                # Implémentation simple de l'arbre binomial CRR
                dt = T / n_steps
                u = np.exp(sigma * np.sqrt(dt))
                d = 1 / u
                p = (np.exp((r - q) * dt) - d) / (u - d)
                
                # Construction de l'arbre
                prices = np.zeros((n_steps + 1, n_steps + 1))
                prices[0, 0] = S0
                
                for i in range(1, n_steps + 1):
                    prices[i, 0] = prices[i-1, 0] * u
                    for j in range(1, i + 1):
                        prices[i, j] = prices[i-1, j-1] * d
                
                # Payoff final
                values = np.zeros(n_steps + 1)
                for j in range(n_steps + 1):
                    S_T = prices[n_steps, j]
                    if is_call:
                        values[j] = max(S_T - K, 0)
                    else:
                        values[j] = max(K - S_T, 0)
                
                # Backward induction
                for i in range(n_steps - 1, -1, -1):
                    for j in range(i + 1):
                        values[j] = np.exp(-r * dt) * (p * values[j] + (1 - p) * values[j + 1])
                
                prix_binom = values[0]
                
                st.metric("Prix Binomial", f"{prix_binom:.6f}")
                st.metric("Erreur absolue", f"{abs(prix_binom - prix_bs):.6f}")
                st.metric("Erreur relative", f"{100*abs(prix_binom - prix_bs)/prix_bs:.4f}%")
                
            except Exception as e:
                st.error(f"Erreur: {e}")
                prix_binom = None
        else:
            st.info("Module binomial non disponible")
            prix_binom = None
    
    # Convergence
    if LATTICE_AVAILABLE and prix_binom is not None:
        st.subheader("Convergence de l'arbre binomial")
        
        steps_range = np.arange(10, 201, 10)
        prices_binom_conv = []
        
        with st.spinner("Calcul de la convergence..."):
            for n in steps_range:
                try:
                    dt = T / n
                    u = np.exp(sigma * np.sqrt(dt))
                    d = 1 / u
                    p = (np.exp((r - q) * dt) - d) / (u - d)
                    
                    prices = np.zeros((n + 1, n + 1))
                    prices[0, 0] = S0
                    
                    for i in range(1, n + 1):
                        prices[i, 0] = prices[i-1, 0] * u
                        for j in range(1, i + 1):
                            prices[i, j] = prices[i-1, j-1] * d
                    
                    values = np.zeros(n + 1)
                    for j in range(n + 1):
                        S_T = prices[n, j]
                        if is_call:
                            values[j] = max(S_T - K, 0)
                        else:
                            values[j] = max(K - S_T, 0)
                    
                    for i in range(n - 1, -1, -1):
                        for j in range(i + 1):
                            values[j] = np.exp(-r * dt) * (p * values[j] + (1 - p) * values[j + 1])
                    
                    prices_binom_conv.append(values[0])
                except:
                    prices_binom_conv.append(np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps_range, prices_binom_conv, 'b.-', label='Binomial')
        ax.axhline(prix_bs, color='red', linestyle='--', label='Black-Scholes')
        ax.set_xlabel("Nombre de pas")
        ax.set_ylabel("Prix")
        ax.set_title("Convergence du pricing binomial vers Black-Scholes")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ============================================================================
# TP7: MONTE CARLO GÉNÉRIQUE
# ============================================================================
with tab7:
    st.header("TP 7 - Monte Carlo générique et options de chemin")
    
    if not MONTECARLO_AVAILABLE:
        st.warning("Module Monte Carlo avancé non disponible. Utilisation d'une simulation simple.")
    
    st.markdown("Simulation du processus stochastique sous-jacent")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres simulation")
        n_paths_mc = st.slider("Nombre de chemins", 1000, 50000, 10000, step=1000)
        n_steps_mc = st.slider("Pas de temps", 50, 500, 252, step=50)
        seed_mc = st.number_input("Seed (0=aléatoire)", 0, 10000, 42, step=1)
        
        option_type_mc = st.selectbox("Type payoff", 
                                      ["Européenne", "Asiatique", "Lookback"])
    
    with col2:
        st.subheader("Simulation et pricing")
        
        if seed_mc > 0:
            np.random.seed(seed_mc)
        
        try:
            # Simulation simple GBM
            dt = T / n_steps_mc
            ttm_grid = np.linspace(0, T, n_steps_mc+1)
            
            # Initialisation
            paths = np.zeros((n_paths_mc, n_steps_mc+1))
            paths[:, 0] = S0
            
            # Simulation
            for i in range(1, n_steps_mc+1):
                Z = np.random.randn(n_paths_mc)
                drift = (r - q - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * Z
                paths[:, i] = paths[:, i-1] * np.exp(drift + diffusion)
            
            # Calcul du payoff selon le type
            if option_type_mc == "Européenne":
                S_T = paths[:, -1]
                if is_call:
                    payoffs = np.maximum(S_T - K, 0)
                else:
                    payoffs = np.maximum(K - S_T, 0)
                    
            elif option_type_mc == "Asiatique":
                S_avg = np.mean(paths, axis=1)
                if is_call:
                    payoffs = np.maximum(S_avg - K, 0)
                else:
                    payoffs = np.maximum(K - S_avg, 0)
                    
            else:  # Lookback
                if is_call:
                    S_max = np.max(paths, axis=1)
                    payoffs = S_max - K
                else:
                    S_min = np.min(paths, axis=1)
                    payoffs = K - S_min
            
            prix_mc = np.exp(-r*T) * np.mean(payoffs)
            err_mc = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths_mc)
            
            st.metric(f"Prix MC ({option_type_mc})", f"{prix_mc:.4f}", 
                     f"±{1.96*err_mc:.4f}")
            
            # Afficher quelques chemins
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Chemins
            n_display = min(100, n_paths_mc)
            for i in range(n_display):
                ax1.plot(ttm_grid, paths[i, :], alpha=0.1, color='blue')
            ax1.plot(ttm_grid, np.mean(paths, axis=0), 'r-', linewidth=2, label='Moyenne')
            ax1.axhline(K, color='green', linestyle='--', label=f'Strike K={K}')
            ax1.set_xlabel("Temps (années)")
            ax1.set_ylabel("S_t")
            ax1.set_title(f"Simulation de {n_display} chemins")
            ax1.legend()
            ax1.grid(True)
            
            # Distribution finale
            ax2.hist(paths[:, -1], bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(K, color='red', linestyle='--', label=f'Strike K={K}')
            ax2.set_xlabel("S_T")
            ax2.set_ylabel("Fréquence")
            ax2.set_title("Distribution du prix final")
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur: {e}")

# ============================================================================
# TP8: SURFACES DE VOLATILITÉ
# ============================================================================
with tab8:
    st.header("TP 8 - Surfaces de volatilité et objets de marché")
    
    st.markdown("""
    Visualisation des smiles de volatilité pour différentes maturités et modèles.
    """)
    
    # Sélection maturités
    maturities = st.multiselect(
        "Maturités (années)",
        [0.25, 0.5, 1.0, 2.0, 3.0],
        default=[0.5, 1.0, 2.0]
    )
    
    if len(maturities) == 0:
        st.warning("Sélectionnez au moins une maturité")
    else:
        model_surf = st.selectbox("Modèle", ["Black-Scholes", "Heston", "Variance Gamma"])
        
        # Créer le modèle
        if model_surf == "Black-Scholes":
            model = BlackScholes(sigma=sigma, forwardCurve=fwd_curve, discountCurve=disc_curve)
        elif model_surf == "Heston":
            model = Heston(v0=0.04, theta=0.04, kappa=1.5, sigma_v=0.3, rho=-0.7,
                          forwardCurve=fwd_curve, discountCurve=disc_curve)
        else:
            model = VarianceGamma(sigma=sigma, theta=-0.1, nu=0.6,
                                 forwardCurve=fwd_curve, discountCurve=disc_curve)
        
        # Grille de strikes relatifs
        moneyness = np.linspace(0.7, 1.3, 40)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd_curve) if IMPLIEDVOL_AVAILABLE else None
        
        for mat in sorted(maturities):
            strikes = moneyness * S0
            is_calls = np.ones(len(strikes), dtype=bool)
            
            try:
                pricer = ProjEuropeanPricer(model=model, N=1024, L=12.0)
                prices = pricer.price_strikes(T=mat, strikes=strikes, is_calls=is_calls)
                
                # Prix
                ax2.plot(moneyness, prices, label=f'T={mat}y', linewidth=2)
                
                # Vols implicites si disponible
                if ivc is not None:
                    try:
                        vols = ivc.imply_vols(strikes=strikes, prices=prices, 
                                             is_calls=is_calls, ttm=mat)
                        # Smile
                        ax1.plot(moneyness, vols, label=f'T={mat}y', linewidth=2)
                    except Exception as e:
                        st.warning(f"Erreur vol implicite pour T={mat}: {e}")
                
            except Exception as e:
                st.warning(f"Erreur pricing pour T={mat}: {e}")
        
        if ivc is not None:
            ax1.set_xlabel("Moneyness K/S₀")
            ax1.set_ylabel("Volatilité implicite")
            ax1.set_title(f"Smiles de volatilité - {model_surf}")
            ax1.legend()
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "Module vol implicite\nnon disponible", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title("Volatilité implicite")
        
        ax2.set_xlabel("Moneyness K/S₀")
        ax2.set_ylabel("Prix Call")
        ax2.set_title(f"Prix des calls - {model_surf}")
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)

# ============================================================================
# TP9: CALIBRATION AVANCÉE
# ============================================================================
with tab9:
    st.header("TP 9 - Calibration avancée: Lévy et SABR")
    
    if not CALIBRATE_AVAILABLE:
        st.warning("Modules de calibration non disponibles.")
        st.info("Les modules FourierModelCalibrator, SabrModelCalibrator ou MarketSlice ne sont pas accessibles.")
    else:
        st.markdown("""
        Calibration de modèles sur des données de marché simulées.
        """)
        
        calib_model = st.selectbox("Modèle à calibrer", 
                                   ["Variance Gamma", "Heston", "SABR"])
        
        # Génération de données de marché synthétiques
        st.subheader("1. Données de marché (simulées)")
        
        strikes_calib = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]) * S0
        T_calib = st.slider("Maturité pour calibration", 0.25, 3.0, 1.0, 0.25)
        
        # Générer des prix de marché avec un modèle
        if calib_model in ["Variance Gamma", "Heston"]:
            # Prix de marché = Heston avec paramètres fixés
            model_market = Heston(v0=0.05, theta=0.04, kappa=2.0, 
                                 sigma_v=0.4, rho=-0.6,
                                 forwardCurve=fwd_curve, discountCurve=disc_curve)
        else:  # SABR
            # Pour SABR on utilise des vols implicites directement
            st.info("SABR: calibration sur volatilités implicites")
            
        col1, col2 = st.columns(2)
        
        with col1:
            if calib_model != "SABR":
                pricer_market = ProjEuropeanPricer(model=model_market, N=1024, L=12.0)
                is_calls_calib = np.ones(len(strikes_calib), dtype=bool)
                market_prices = pricer_market.price_strikes(T=T_calib, 
                                                           strikes=strikes_calib,
                                                           is_calls=is_calls_calib)
                
                st.write("**Prix de marché (simulés):**")
                df_market = {"Strike": strikes_calib, "Prix": market_prices}
                st.dataframe(df_market)
        
        with col2:
            st.subheader("2. Calibration")
            
            if st.button("Lancer la calibration", type="primary"):
                with st.spinner("Calibration en cours..."):
                    try:
                        if calib_model == "Variance Gamma":
                            # Création du market slice
                            market_slice = MarketSlice(
                                ttm=T_calib,
                                strikes=strikes_calib,
                                call_prices=market_prices,
                                fwd=fwd_curve(T_calib),
                                disc=disc_curve(T_calib)
                            )
                            
                            # Paramètres initiaux
                            params_init = {'sigma': 0.2, 'theta': 0.0, 'nu': 0.5}
                            
                            # Calibrator
                            calibrator = FourierModelCalibrator(
                                model_class=VarianceGamma,
                                market_slice=market_slice,
                                forwardCurve=fwd_curve,
                                discountCurve=disc_curve
                            )
                            
                            result = calibrator.calibrate(params_init=params_init)
                            
                            st.success("Calibration terminée!")
                            st.write("**Paramètres calibrés:**")
                            st.json(result)
                            
                        elif calib_model == "Heston":
                            market_slice = MarketSlice(
                                ttm=T_calib,
                                strikes=strikes_calib,
                                call_prices=market_prices,
                                fwd=fwd_curve(T_calib),
                                disc=disc_curve(T_calib)
                            )
                            
                            params_init = {'v0': 0.04, 'theta': 0.04, 'kappa': 1.0,
                                          'sigma_v': 0.3, 'rho': -0.5}
                            
                            calibrator = FourierModelCalibrator(
                                model_class=Heston,
                                market_slice=market_slice,
                                forwardCurve=fwd_curve,
                                discountCurve=disc_curve
                            )
                            
                            result = calibrator.calibrate(params_init=params_init)
                            
                            st.success("Calibration terminée!")
                            st.write("**Paramètres calibrés:**")
                            st.json(result)
                            
                        else:  # SABR
                            st.info("Calibration SABR nécessite des vols implicites de marché")
                            # Simuler des vols implicites
                            if IMPLIEDVOL_AVAILABLE:
                                ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, 
                                                                  fwd_curve=fwd_curve)
                                market_vols = ivc.imply_vols(strikes=strikes_calib,
                                                            prices=market_prices,
                                                            is_calls=is_calls_calib,
                                                            ttm=T_calib)
                                
                                F = fwd_curve(T_calib)
                                params_init = {'alpha': 0.2, 'beta': 0.7, 'rho': -0.3, 'nu': 0.4}
                                
                                calibrator = SabrModelCalibrator(
                                    ttm=T_calib,
                                    forward=F,
                                    strikes=strikes_calib,
                                    vols=market_vols
                                )
                                
                                result = calibrator.calibrate(params_init=params_init)
                                
                                st.success("Calibration SABR terminée!")
                                st.write("**Paramètres calibrés:**")
                                st.json(result)
                            else:
                                st.error("Module de vol implicite nécessaire pour SABR")
                            
                    except Exception as e:
                        st.error(f"Erreur lors de la calibration: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# ============================================================================
# TP10: DATES ET DONNÉES
# ============================================================================
with tab10:
    st.header("TP 10 - Dates, day count et données Yahoo Finance")
    
    st.markdown("""
    Gestion des dates et chargement de données de marché depuis Yahoo Finance.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calculs de dates")
        
        from fypy.date.Date import Date
        from fypy.date.DayCounter import DayCounter_Act365, DayCounter_30_360
        
        # Dates
        date1_str = st.date_input("Date 1", value=None)
        date2_str = st.date_input("Date 2", value=None)
        
        if date1_str and date2_str:
            date1 = Date(date1_str.year, date1_str.month, date1_str.day)
            date2 = Date(date2_str.year, date2_str.month, date2_str.day)
            
            # Day counters
            dc_act365 = DayCounter_Act365()
            dc_30360 = DayCounter_30_360()
            
            year_frac_act365 = dc_act365.year_fraction(date1, date2)
            year_frac_30360 = dc_30360.year_fraction(date1, date2)
            
            st.write(f"**Date 1:** {date1}")
            st.write(f"**Date 2:** {date2}")
            st.write(f"**Fraction d'année (Act/365):** {year_frac_act365:.6f}")
            st.write(f"**Fraction d'année (30/360):** {year_frac_30360:.6f}")
            st.write(f"**Jours calendaires:** {(date2_str - date1_str).days}")
    
    with col2:
        st.subheader("Données Yahoo Finance")
        
        ticker = st.text_input("Ticker", value="AAPL")
        
        if st.button("Charger les données"):
            try:
                import yfinance as yf
                from datetime import datetime, timedelta
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                with st.spinner(f"Chargement des données pour {ticker}..."):
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty:
                        st.success(f"Données chargées: {len(data)} observations")
                        
                        # Graphique
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(data.index, data['Close'], linewidth=2)
                        ax.set_title(f"Prix de clôture - {ticker}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Prix")
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        # Stats
                        st.write("**Statistiques:**")
                        st.write(f"- Prix actuel: ${data['Close'].iloc[-1]:.2f}")
                        st.write(f"- Prix min (1an): ${data['Close'].min():.2f}")
                        st.write(f"- Prix max (1an): ${data['Close'].max():.2f}")
                        
                        # Volatilité historique
                        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                        vol_hist = returns.std() * np.sqrt(252)
                        st.write(f"- Volatilité historique (annualisée): {vol_hist:.2%}")
                        
                    else:
                        st.error("Aucune donnée disponible")
                        
            except ImportError:
                st.error("Module yfinance non installé. Installez-le avec: pip install yfinance")
            except Exception as e:
                st.error(f"Erreur: {e}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Application FYPY - Librairie de pricing d'options financières</p>
    <p>📚 Consolidation des TPs 1 à 10</p>
</div>
""", unsafe_allow_html=True)
