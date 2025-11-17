import numpy as np
import streamlit as st


class CrankNicolsonBS:
    """
    Solveur Crank–Nicolson pour la PDE de Black–Scholes en log(S).

    Typeflag:
        'Eu'  : option européenne
        'Am'  : option américaine (exercice possible à chaque date de grille)
        'Bmd' : option bermudéenne (exercice possible à certaines dates)
    cpflag:
        'c' : call
        'p' : put
    """

    def __init__(self, Typeflag, cpflag, S0, K, T, vol, r, d):
        self.Typeflag = Typeflag
        self.cpflag = cpflag
        self.S0 = S0
        self.K = K
        self.T = T
        self.vol = vol
        self.r = r
        self.d = d

    def CN_option_info(
        self,
        Typeflag=None,
        cpflag=None,
        S0=None,
        K=None,
        T=None,
        vol=None,
        r=None,
        d=None,
    ):
        """
        Résout la PDE et retourne (Price, Delta, Gamma, Theta).
        """

        # Mise à jour des paramètres éventuels
        Typeflag = Typeflag or self.Typeflag
        cpflag = cpflag or self.cpflag
        S0 = S0 or self.S0
        K = K or self.K
        T = T or self.T
        vol = vol or self.vol
        r = r or self.r
        d = d or self.d

        # Paramètres de la grille en x,t
        mu = r - d - 0.5 * vol * vol
        x_max = vol * np.sqrt(T) * 5
        N = 500
        dx = 2 * x_max / N
        X = np.linspace(-x_max, x_max, N + 1)
        n = np.arange(0, N + 1)

        J = 600
        dt = T / J

        # Coefficients Crank–Nicolson
        a = 0.25 * dt * ((vol**2) * (n**2) - mu * n)
        b = -0.5 * dt * ((vol**2) * (n**2) + r)
        c = 0.25 * dt * ((vol**2) * (n**2) + mu * n)

        main_diag_A = 1 - b - 2 * a
        upper_A = a + c
        lower_A = a - c

        main_diag_B = 1 + b + 2 * a
        upper_B = -a - c
        lower_B = -a + c

        A = np.zeros((N + 1, N + 1))
        B = np.zeros((N + 1, N + 1))

        np.fill_diagonal(A, main_diag_A)
        np.fill_diagonal(A[1:], lower_A[:-1])
        np.fill_diagonal(A[:, 1:], upper_A[:-1])

        np.fill_diagonal(B, main_diag_B)
        np.fill_diagonal(B[1:], lower_B[:-1])
        np.fill_diagonal(B[:, 1:], upper_B[:-1])

        Ainv = np.linalg.inv(A)

        # Payoff terminal à maturité
        if cpflag == "c":
            V = np.clip(S0 * np.exp(X) - K, 0, 1e10)
        elif cpflag == "p":
            V = np.clip(K - S0 * np.exp(X), 0, 1e10)
        else:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        V0 = V.copy()
        V1 = V.copy()

        # Backward en temps selon le type d’option
        if Typeflag == "Am":
            for j in range(J):
                if j == J - 1:
                    V1 = V.copy()
                V = B.dot(V)
                V = Ainv.dot(V)
                V = np.where(V > V0, V, V0)

        elif Typeflag == "Bmd":
            exercise_step = 10
            for j in range(J):
                if j == J - 1:
                    V1 = V.copy()
                V = B.dot(V)
                V = Ainv.dot(V)
                if j % exercise_step == 0:
                    V = np.where(V > V0, V, V0)

        elif Typeflag == "Eu":
            for j in range(J):
                if j == J - 1:
                    V1 = V.copy()
                V = B.dot(V)
                V = Ainv.dot(V)
                if cpflag == "c":
                    V[0] = 0.0
                    V[-1] = S0 * np.exp(x_max) - K * np.exp(-r * dt * j)
                else:
                    V[0] = K * np.exp(-r * dt * (J - j))
                    V[-1] = 0.0
        else:
            raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")

        # Extraction du prix et des grecs à S = S0
        n_mid = N // 2
        price = V[n_mid]

        Sp = S0 * np.exp(dx)
        Sm = S0 * np.exp(-dx)
        delta = (V[n_mid + 1] - V[n_mid - 1]) / (Sp - Sm)

        dVdSp = (V[n_mid + 1] - V[n_mid]) / (Sp - S0)
        dVdSm = (V[n_mid] - V[n_mid - 1]) / (S0 - Sm)
        gamma = (dVdSp - dVdSm) / ((Sp - Sm) / 2.0)

        theta = -(V[n_mid] - V1[n_mid]) / dt

        return float(price), float(delta), float(gamma), float(theta)


def CN_Barrier_option(Typeflag, cpflag, S0, K, Hu, Hd, T, vol, r, d):
    """
    Pricing d'une option barrière par Crank–Nicolson.
    """

    mu = r - d - 0.5 * vol * vol
    x_max = vol * np.sqrt(T) * 5
    N = 500
    dx = 2 * x_max / N
    X = np.linspace(-x_max, x_max, N + 1)
    n = np.arange(0, N + 1)

    J = 600
    dt = T / J

    a = 0.25 * dt * ((vol**2) * (n**2) - mu * n)
    b = -0.5 * dt * ((vol**2) * (n**2) + r)
    c = 0.25 * dt * ((vol**2) * (n**2) + mu * n)

    main_diag_A = 1 - b - 2 * a
    upper_A = a + c
    lower_A = a - c

    main_diag_B = 1 + b + 2 * a
    upper_B = -a - c
    lower_B = -a + c

    A = np.zeros((N + 1, N + 1))
    B = np.zeros((N + 1, N + 1))

    np.fill_diagonal(A, main_diag_A)
    np.fill_diagonal(A[1:], lower_A[:-1])
    np.fill_diagonal(A[:, 1:], upper_A[:-1])

    np.fill_diagonal(B, main_diag_B)
    np.fill_diagonal(B[1:], lower_B[:-1])
    np.fill_diagonal(B[:, 1:], upper_B[:-1])

    Ainv = np.linalg.inv(A)

    S_grid = S0 * np.exp(X)
    if cpflag == "c":
        V = np.clip(S_grid - K, 0, 1e10)
    elif cpflag == "p":
        V = np.clip(K - S_grid, 0, 1e10)
    else:
        raise ValueError("cpflag doit être 'c' ou 'p'.")

    if Typeflag == "UNO":
        V = np.where(S_grid < Hu, V, 0.0)
    elif Typeflag == "DNO":
        V = np.where((S_grid > Hd) & (S_grid < Hu), V, 0.0)
    else:
        raise ValueError("Typeflag doit être 'UNO' ou 'DNO'.")

    V1 = V.copy()

    for j in range(J):
        if j == J - 1:
            V1 = V.copy()

        V = B.dot(V)
        V = Ainv.dot(V)

        S_grid = S0 * np.exp(X)
        if Typeflag == "UNO":
            V = np.where(S_grid < Hu, V, 0.0)
        elif Typeflag == "DNO":
            V = np.where((S_grid > Hd) & (S_grid < Hu), V, 0.0)

    n_mid = N // 2

    price = V[n_mid]

    Sp = S0 * np.exp(dx)
    Sm = S0 * np.exp(-dx)

    delta = (V[n_mid + 1] - V[n_mid - 1]) / (Sp - Sm)

    dVdSp = (V[n_mid + 1] - V[n_mid]) / (Sp - S0)
    dVdSm = (V[n_mid] - V[n_mid - 1]) / (S0 - Sm)
    gamma = (dVdSp - dVdSm) / ((Sp - Sm) / 2.0)

    theta = -(V[n_mid] - V1[n_mid]) / dt

    return float(price), float(delta), float(gamma), float(theta)


st.title("Bermuda & Barrier Options (Crank–Nicolson)")

st.sidebar.header("Paramètres généraux")
S0 = st.sidebar.number_input("S0 (spot)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("K (strike)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("T (maturité en années)", value=1.0, min_value=0.01)
vol = st.sidebar.number_input("Volatilité", value=0.4, min_value=0.0001)
r = st.sidebar.number_input("Taux sans risque r", value=0.025)
d = st.sidebar.number_input("Dividende continu d", value=0.0175)

tab1, tab2 = st.tabs(["Bermuda / European / American", "Barrier options"])

with tab1:
    st.subheader("Option européenne / américaine / bermudéenne")
    typeflag = st.selectbox("Type d'option", ["Eu", "Am", "Bmd"])
    cpflag = st.selectbox("Call / Put", ["c", "p"])

    if st.button("Calculer (Bermuda/Eu/Am)"):
        model = CrankNicolsonBS(
            Typeflag=typeflag,
            cpflag=cpflag,
            S0=S0,
            K=K,
            T=T,
            vol=vol,
            r=r,
            d=d,
        )
        price, delta, gamma, theta = model.CN_option_info()

        st.write(f"**Prix**: {price:.4f}")
        st.write(f"**Delta**: {delta:.4f}")
        st.write(f"**Gamma**: {gamma:.4f}")
        st.write(f"**Theta**: {theta:.4f}")

with tab2:
    st.subheader("Options barrière (Up-and-out / Double knock-out)")
    barrier_type = st.selectbox("Type de barrière", ["UNO (Up-and-out)", "DNO (Double knock-out)"])
    barrier_flag = "UNO" if barrier_type.startswith("UNO") else "DNO"
    cpflag_barrier = st.selectbox("Call / Put (barrière)", ["c", "p"])
    Hu = st.number_input("Barrière haute Hu", value=120.0, min_value=0.01)
    Hd = st.number_input("Barrière basse Hd", value=0.0, min_value=0.0)

    if st.button("Calculer (barrière)"):
        price_b, delta_b, gamma_b, theta_b = CN_Barrier_option(
            Typeflag=barrier_flag,
            cpflag=cpflag_barrier,
            S0=S0,
            K=K,
            Hu=Hu,
            Hd=Hd,
            T=T,
            vol=vol,
            r=r,
            d=d,
        )

        st.write(f"**Prix**: {price_b:.4f}")
        st.write(f"**Delta**: {delta_b:.4f}")
        st.write(f"**Gamma**: {gamma_b:.4f}")
        st.write(f"**Theta**: {theta_b:.4f}")

