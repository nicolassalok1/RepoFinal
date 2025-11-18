#!/bin/bash

# Script de lancement de l'application Streamlit FYPY

echo "🚀 Lancement de l'application FYPY Streamlit..."
echo ""
echo "📍 Répertoire: $(pwd)"
echo "📊 Application: streamlit_fypy_app.py"
echo ""
echo "⚙️  Configuration:"
echo "   - Port: 8501 (par défaut)"
echo "   - URL: http://localhost:8501"
echo ""
echo "💡 Conseil: Utilisez Ctrl+C pour arrêter l'application"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Vérifier que streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit n'est pas installé!"
    echo "📦 Installation avec: pip install streamlit"
    exit 1
fi

# Vérifier que le fichier existe
if [ ! -f "streamlit_fypy_app.py" ]; then
    echo "❌ Fichier streamlit_fypy_app.py introuvable!"
    echo "📂 Assurez-vous d'être dans le dossier FYPY/"
    exit 1
fi

# Lancer l'application
streamlit run streamlit_fypy_app.py
