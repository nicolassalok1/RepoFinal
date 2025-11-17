def BTM(strike_type, option_type, S0, K, r, sigma, T, N):
    """
    Binomial Tree Model naïf pour option asiatique arithmétique.

    strike_type : "fixed" (strike K) ou "floating" (strike = moyenne / S_N)
    option_type : "C" (call) ou "P" (put)
    S0          : spot initial
    K           : strike
    r           : taux sans risque (continu)
    sigma       : volatilité
    T           : maturité
    N           : nombre de pas de temps
    """
    deltaT = T / N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    p = (np.exp(r * deltaT) - d) / (u - d)

    # On part de S0
    St = [S0]
    At = [S0]     # somme des prix le long du chemin
    strike = [K]  # strike répété (cas fixed strike)

    # Construction exhaustive des 2^N trajectoires
    for _ in range(N):
        # Nouveaux prix terminal pour tous les chemins :
        # - branche up : S * u
        # - branche down : S * d
        St = [s * u for s in St] + [s * d for s in St]

        # Duplique les chemins pour les deux branches (up et down)
        At = At + At
        strike = strike + strike

        # Ajoute le prix courant au cumul de chaque trajectoire
        for i in range(len(At)):
            At[i] = At[i] + St[i]

    # Moyenne arithmétique sur chaque trajectoire
    At = np.array(At) / (N + 1)
    St = np.array(St)
    strike = np.array(strike)

    # Payoff asiatique (strike fixe ou flottant)
    if strike_type == "fixed":
        if option_type == "C":
            payoff = np.maximum(At - strike, 0.0)
        else:
            payoff = np.maximum(strike - At, 0.0)
    else:
        # floating strike
        if option_type == "C":
            payoff = np.maximum(St - At, 0.0)
        else:
            payoff = np.maximum(At - St, 0.0)

    # Remontée backward sur l'arbre (probabilité neutre au risque)
    option_price = payoff.copy()
    for _ in range(N):
        length = len(option_price) // 2
        option_price = p * option_price[:length] + (1 - p) * option_price[length:]

    return option_price[0]
