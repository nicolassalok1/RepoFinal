# Application Streamlit FYPY

## Description

Cette application Streamlit consolide tous les notebooks du dossier FYPY/ (TPs 1 à 10) en une interface interactive unique.

## Architecture

L'application est organisée en **10 onglets**, un pour chaque TP:

1. **TP1: Architecture** - Courbes de taux et forwards d'equity
2. **TP2: Black-Scholes** - Pricing BS et Monte Carlo simple
3. **TP3: Lévy & Fourier** - Processus de Lévy et méthode PROJ
4. **TP4: Heston** - Volatilité stochastique
5. **TP5: Exotiques** - Options asiatiques
6. **TP6: Binomial** - Arbre binomial et convergence
7. **TP7: Monte Carlo** - Simulations génériques de chemins
8. **TP8: Vol Surfaces** - Surfaces de volatilité
9. **TP9: Calibration** - Calibration de modèles (Lévy, Heston, SABR)
10. **TP10: Dates & Data** - Gestion des dates et Yahoo Finance

## Paramètres Communs (Sidebar)

La **barre latérale** permet de configurer les paramètres communs à tous les onglets:

### Marché
- **Prix spot S₀**: Prix actuel du sous-jacent
- **Taux sans risque r**: Taux d'intérêt continu
- **Dividende continu q**: Rendement du dividende

### Option
- **Maturité T**: Temps jusqu'à l'échéance (années)
- **Strike K**: Prix d'exercice
- **Type**: Call ou Put

### Volatilité
- **σ**: Volatilité (pour modèles Black-Scholes)

## Lancement

### Méthode 1: Ligne de commande

```bash
cd /home/salok1/PythonProjects/RepoFinal/FYPY
streamlit run streamlit_fypy_app.py
```

### Méthode 2: Python

```bash
cd /home/salok1/PythonProjects/RepoFinal/FYPY
python3 -m streamlit run streamlit_fypy_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`

## Fonctionnalités par onglet

### TP1: Architecture
- Visualisation des courbes d'actualisation
- Courbes forward d'equity
- Calcul des taux implicites

### TP2: Black-Scholes
- Pricing analytique Black-Scholes
- Calcul des Greeks (Delta, Gamma)
- Simulation Monte Carlo
- Comparaison BS vs MC
- Distribution des prix finaux

### TP3: Lévy & Fourier
- Choix du modèle: BS, VG, BG, NIG, Merton JD, Kou JD
- Paramètres spécifiques à chaque modèle
- Pricing par méthode PROJ
- Visualisation du smile de volatilité
- Comparaison des prix

### TP4: Heston
- Modèle à volatilité stochastique
- Paramètres: v₀, θ, κ, σᵥ, ρ
- Pricing PROJ
- Comparaison avec Black-Scholes
- Smile de volatilité caractéristique

### TP5: Exotiques
- Options asiatiques arithmétiques (approximation Vorst)
- Options asiatiques géométriques (formule exacte)
- Nombre d'observations configurable
- Comparaison avec européennes

### TP6: Binomial
- Arbre binomial Cox-Ross-Rubinstein
- Convergence vers Black-Scholes
- Nombre de pas configurable
- Erreur de pricing vs BS

### TP7: Monte Carlo
- Simulation de chemins stochastiques
- Options européennes, asiatiques, lookback
- Visualisation des trajectoires
- Distribution des prix finaux
- Intervalles de confiance

### TP8: Vol Surfaces
- Smiles de volatilité multi-maturités
- Choix du modèle: BS, Heston, VG
- Visualisation 2D (smile par maturité)
- Prix des calls par moneyness

### TP9: Calibration
- Calibration Variance Gamma
- Calibration Heston
- Calibration SABR
- Données de marché simulées
- Résultats de calibration

### TP10: Dates & Data
- Calculs de fractions d'année
- Day count conventions (Act/365, 30/360)
- Chargement de données Yahoo Finance
- Volatilité historique
- Graphiques de prix

## Dépendances

L'application nécessite:

- `streamlit`
- `numpy`
- `matplotlib`
- `scipy`
- `yfinance` (pour TP10)
- `fypy` (librairie locale)

## Structure du code

```
streamlit_fypy_app.py
├── Configuration page Streamlit
├── Sidebar: Paramètres communs
├── Création des courbes (disc_curve, fwd_curve)
├── Onglets (tabs)
│   ├── Tab 1: Architecture
│   ├── Tab 2: Black-Scholes
│   ├── Tab 3: Lévy & Fourier
│   ├── Tab 4: Heston
│   ├── Tab 5: Exotiques
│   ├── Tab 6: Binomial
│   ├── Tab 7: Monte Carlo
│   ├── Tab 8: Vol Surfaces
│   ├── Tab 9: Calibration
│   └── Tab 10: Dates & Data
└── Footer
```

## Utilisation

1. **Lancer l'application** avec la commande streamlit
2. **Ajuster les paramètres** dans la sidebar selon vos besoins
3. **Naviguer entre les onglets** pour explorer les différentes fonctionnalités
4. **Modifier les paramètres spécifiques** à chaque onglet si nécessaire
5. **Visualiser les graphiques** et résultats en temps réel

## Notes

- Les calculs sont effectués en temps réel lors du changement des paramètres
- Certains calculs (calibration, Monte Carlo avec beaucoup de chemins) peuvent prendre quelques secondes
- Les graphiques sont interactifs (zoom, pan) grâce à matplotlib
- L'application conserve l'état entre les changements d'onglets

## Exemple d'utilisation

1. Aller dans **TP3: Lévy & Fourier**
2. Sélectionner le modèle "Variance Gamma"
3. Ajuster theta=-0.15, nu=0.8
4. Observer le smile de volatilité asymétrique (skew)
5. Comparer avec Black-Scholes (smile plat)

## Troubleshooting

### Erreur d'import fypy
Vérifier que vous êtes dans le bon dossier:
```bash
cd /home/salok1/PythonProjects/RepoFinal/FYPY
```

### Port déjà utilisé
Utiliser un port différent:
```bash
streamlit run streamlit_fypy_app.py --server.port 8502
```

### Module manquant
Installer les dépendances:
```bash
pip install streamlit numpy matplotlib scipy yfinance
```

## Contact

Pour toute question ou suggestion d'amélioration, n'hésitez pas à contribuer!
