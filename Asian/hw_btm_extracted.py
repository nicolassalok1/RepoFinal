def HW_BTM(strike_type, option_type, S0, K, r, sigma, T, step, M):
    """
    Schéma binomial de type Hull–White pour option asiatique arithmétique.

    strike_type : "fixed" ou "floating"
    option_type : "C" (call) ou "P" (put)
    S0          : spot
    K           : strike
    r           : taux
    sigma       : volatilité
    T           : maturité
    step        : nombre de pas de temps N
    M           : nombre de points pour discrétiser la moyenne A à chaque nœud

    Idée :
    - Pour chaque niveau (N, J), on approxime la moyenne par M valeurs
      allant de A_min à A_max.
    - À la remontée backward, on interpole les payoffs en fonction de A.
    """

    N = step
    deltaT = T / N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1.0 / u
    p = (np.exp(r * deltaT) - d) / (u - d)

    # On va construire At[N][J][k] = valeur de moyenne approx à l'état (N,J) et point k
    At = []
    strike = np.array([K] * M)

    # 1. Construction de la grille de moyennes At au temps final N
    for J in range(N + 1):
        # On calcule A_max et A_min possibles à (N,J) en utilisant deux chemins extrêmes
        # (toutes les hausses d'abord / toutes les baisses d'abord)
        path_up_then_down = np.array(
            [S0 * u**j * d**0 for j in range(N - J)] +
            [S0 * u**(N - J) * d**j for j in range(J + 1)]
        )
        A_max = path_up_then_down.mean()

        path_down_then_up = np.array(
            [S0 * d**j * u**0 for j in range(J + 1)] +
            [S0 * d**J * u**(j + 1) for j in range(N - J)]
        )
        A_min = path_down_then_up.mean()

        diff = A_max - A_min
        # M points régulièrement espacés entre A_max et A_min
        A_vals = [A_max - diff * k / (M - 1) for k in range(M)]
        At.append(A_vals)

    At = np.round(At, 4)
    St = np.array([S0 * u**(N - J) * d**J for J in range(N + 1)])  # S à l'échéance selon J

    # 2. Payoff à l'échéance pour chaque (N,J,k)
    # On utilise la même moyenne At pour toutes les J au début
    payoff = []

    for J in range(N + 1):
        A_vals = np.array(At[J])
        S_vals = np.array([S0 * u**(N - J) * d**J] * M)

        if strike_type == "fixed":
            if option_type == "C":
                pay = np.maximum(A_vals - strike, 0.0)
            else:
                pay = np.maximum(strike - A_vals, 0.0)
        else:
            if option_type == "C":
                pay = np.maximum(S_vals - A_vals, 0.0)
            else:
                pay = np.maximum(A_vals - S_vals, 0.0)

        payoff.append(pay)

    payoff = np.round(np.array(payoff), 4)

    # 3. Remontée backward avec interpolation sur A
    for n in range(N - 1, -1, -1):
        At_backward = []
        payoff_backward = []

        for J in range(n + 1):
            # Recalcule les A_min / A_max pour le niveau (n,J)
            path_up_then_down = np.array(
                [S0 * u**j * d**0 for j in range(n - J)] +
                [S0 * u**(n - J) * d**j for j in range(J + 1)]
            )
            A_max = path_up_then_down.mean()

            path_down_then_up = np.array(
                [S0 * d**j * u**0 for j in range(J + 1)] +
                [S0 * d**J * u**(j + 1) for j in range(n - J)]
            )
            A_min = path_down_then_up.mean()

            diff = A_max - A_min
            A_vals = np.array([A_max - diff * k / (M - 1) for k in range(M)])
            At_backward.append(A_vals)

        At_backward = np.round(np.array(At_backward), 4)

        # On va construire pour chaque (n,J,k) la valeur en fonction des états enfants (up, down)
        payoff_new = []

        for J in range(n + 1):
            A_vals = At_backward[J]
            pay_vals = np.zeros_like(A_vals)

            # Enfant "up" est (n+1, J), enfant "down" est (n+1, J+1)
            A_up = np.array(At[J])
            A_down = np.array(At[J + 1])
            pay_up = payoff[J]
            pay_down = payoff[J + 1]

            # Interpolation linéaire de pay_up(A) et pay_down(A) sur les nouvelles A_vals
            for k, A_k in enumerate(A_vals):
                # interpolation sur l'enfant up
                if A_k <= A_up[0]:
                    fu = pay_up[0]
                elif A_k >= A_up[-1]:
                    fu = pay_up[-1]
                else:
                    idx = np.searchsorted(A_up, A_k) - 1
                    x0, x1 = A_up[idx], A_up[idx+1]
                    y0, y1 = pay_up[idx], pay_up[idx+1]
                    fu = y0 + (y1 - y0) * (A_k - x0) / (x1 - x0)

                # interpolation sur l'enfant down
                if A_k <= A_down[0]:
                    fd = pay_down[0]
                elif A_k >= A_down[-1]:
                    fd = pay_down[-1]
                else:
                    idx = np.searchsorted(A_down, A_k) - 1
                    x0, x1 = A_down[idx], A_down[idx+1]
                    y0, y1 = pay_down[idx], pay_down[idx+1]
                    fd = y0 + (y1 - y0) * (A_k - x0) / (x1 - x0)

                # Valeur actualisée au nœud (n,J,A_k)
                pay_vals[k] = (p * fu + (1 - p) * fd) * np.exp(-r * deltaT)

            payoff_backward.append(pay_vals)

        # On remplace At et payoff par les valeurs du niveau précédent
        At = At_backward
        payoff = np.round(np.array(payoff_backward), 4)

    # Au temps 0, il reste une seule ligne J=0, on en prend la moyenne sur A
    option_price = payoff[0].mean()
    return option_price
