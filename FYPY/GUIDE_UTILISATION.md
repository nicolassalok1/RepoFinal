# 🚀 Guide de démarrage rapide - Application FYPY Streamlit

## Lancement rapide

```bash
cd /home/salok1/PythonProjects/RepoFinal/FYPY
./launch_app.sh
```

Ou directement:
```bash
streamlit run streamlit_fypy_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

## Structure de l'application

### Barre latérale (gauche) - Paramètres globaux
Ajustez les paramètres communs à tous les onglets:
- **Marché**: S₀, r (taux sans risque), q (dividende)
- **Option**: T (maturité), K (strike), Type (Call/Put)
- **Volatilité**: σ

Ces paramètres sont partagés et s'appliquent à tous les onglets.

### 10 Onglets - Un par TP

#### 📊 TP1: Architecture
- Courbes d'actualisation D(T)
- Courbes forward F(T)
- Taux implicites

#### 🎯 TP2: Black-Scholes
- Pricing analytique BS
- Greeks (Delta, Gamma)
- Simulation Monte Carlo simple
- Comparaison BS vs MC

#### 🎲 TP3: Lévy & Fourier
- Modèles: BS, Variance Gamma, Bilateral Gamma, NIG, Merton JD, Kou JD
- Pricing par méthode PROJ
- Smile de volatilité
- **Conseil**: Essayez VG avec theta=-0.15, nu=0.8 pour voir un skew prononcé

#### 📈 TP4: Heston
- Volatilité stochastique
- Paramètres: v₀, θ, κ, σᵥ, ρ
- Smile caractéristique
- **Conseil**: rho<0 crée un skew, σᵥ élevé crée du smile

#### 🌏 TP5: Exotiques
- Options asiatiques arithmétiques et géométriques
- Comparaison avec européennes
- Prix toujours inférieur à l'européenne (effet de moyenne)

#### 🌳 TP6: Binomial
- Arbre binomial CRR
- Convergence vers Black-Scholes
- **Conseil**: Observer la convergence avec 50, 100, 200 pas

#### 🎰 TP7: Monte Carlo
- Simulation de chemins GBM
- Options: Européenne, Asiatique, Lookback
- Visualisation des trajectoires
- **Conseil**: 10000 chemins donnent un bon compromis vitesse/précision

#### 📐 TP8: Vol Surfaces
- Smiles multi-maturités
- Modèles: BS (flat), Heston (skew), VG (smile)
- **Conseil**: Sélectionner maturités 0.5y, 1y, 2y pour voir l'évolution

#### 🔧 TP9: Calibration
- Calibration Variance Gamma, Heston, SABR
- Sur données de marché simulées
- **Note**: Processus peut prendre quelques secondes

#### 📅 TP10: Dates & Data
- Day count conventions
- Chargement Yahoo Finance
- Volatilité historique
- **Conseil**: Essayez AAPL, MSFT, GOOGL

## Scénarios d'utilisation

### Scénario 1: Comparer les modèles
1. Aller dans **TP3: Lévy & Fourier**
2. Noter le prix pour VG avec theta=-0.1
3. Aller dans **TP4: Heston**
4. Comparer les smiles de volatilité

### Scénario 2: Étudier la convergence
1. **TP6: Binomial** - Observer convergence avec n croissant
2. **TP7: Monte Carlo** - Observer précision avec + de chemins

### Scénario 3: Pricing complet
1. **TP10** - Charger données réelles (ex: AAPL)
2. Noter la volatilité historique
3. **TP2** - Utiliser cette vol pour pricer une option
4. **TP3** - Comparer avec un modèle à sauts

## Astuces

### Performance
- **Monte Carlo**: 10000 chemins = rapide, 50000 = précis mais lent
- **Binomial**: >200 pas peut être lent
- **PROJ**: N=1024 est un bon compromis

### Paramètres intéressants
- **VG Skew**: theta=-0.15, nu=0.8, sigma=0.25
- **Heston Smile**: v0=0.04, kappa=1.5, sigma_v=0.4, rho=-0.7
- **Options ATM**: K = S₀ = 100
- **OTM Call**: K = 110, S₀ = 100
- **ITM Put**: K = 110, S₀ = 100, type=Put

### Interprétation
- **Smile**: Vol plus élevée en OTM → primes d'assurance
- **Skew**: Vol décroît avec K → crash premium
- **Term structure**: Smile s'aplatit avec la maturité

## Dépannage

### L'application ne se lance pas
```bash
# Vérifier streamlit
streamlit --version

# Installer si nécessaire
pip install streamlit
```

### Erreurs d'import fypy
```bash
# S'assurer d'être dans le bon dossier
cd /home/salok1/PythonProjects/RepoFinal/FYPY
pwd  # Doit afficher .../FYPY
```

### Port déjà utilisé
```bash
# Utiliser un autre port
streamlit run streamlit_fypy_app.py --server.port 8502
```

### Graphiques ne s'affichent pas
- Rafraîchir la page (F5)
- Vérifier matplotlib: `pip install matplotlib`

## Raccourcis clavier (dans l'app)

- **R**: Rafraîchir/Rerun
- **C**: Clear cache
- **?**: Aide Streamlit

## Modules requis

✅ Core (obligatoire):
- streamlit
- numpy
- scipy
- matplotlib
- fypy (local)

⚠️ Optionnels (pour certaines fonctionnalités):
- yfinance (TP10 - données Yahoo)
- py_lets_be_rational (volatilité implicite)

## Performance

Temps d'exécution typiques sur machine moderne:
- **Black-Scholes**: Instantané
- **PROJ (N=1024)**: <1s
- **Monte Carlo (10k chemins)**: <1s
- **Binomial (100 pas)**: <1s
- **Calibration**: 5-15s

## Prochaines étapes

1. **Explorer chaque onglet** avec les paramètres par défaut
2. **Modifier la volatilité** dans la sidebar et observer les impacts
3. **Comparer les modèles** entre TP3 (Lévy) et TP4 (Heston)
4. **Tester des cas extrêmes** (vol très haute/basse, OTM profond, etc.)

## Support

Pour toute question sur:
- **Streamlit**: https://docs.streamlit.io
- **FYPY**: Consulter les notebooks originaux dans FYPY/
- **Finance**: Les TPs contiennent les explications théoriques

---

**Bon pricing! 📈💰**
