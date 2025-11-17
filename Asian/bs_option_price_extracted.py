def BS_option_price(t, St, K, T, r, sigma, opt_type):
    """
    Prix Black–Scholes d'une option européenne.

    opt_type : "call" ou "put"
    """
    tau = T - t
    if tau <= 0:
        return max(St - K, 0.0) if opt_type == "call" else max(K - St, 0.0)

    d1 = (np.log(St / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if opt_type == "call":
        price = St * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - St * norm.cdf(-d1)
    return price
