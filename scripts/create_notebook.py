#!/usr/bin/env python3
# scripts/create_notebook.py
"""
Script pour créer automatiquement le notebook d'exploration de données.
"""

import json
from pathlib import Path

# Contenu du notebook
notebook_content = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}


# Fonction helper pour créer une cellule
def create_cell(cell_type, source, metadata=None):
    """Crée une cellule de notebook."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }

    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    return cell


# Ajouter les cellules
cells = [
    # Titre
    create_cell("markdown", [
        "# 📊 Exploration des Données Crypto - Phase 1\n",
        "\n",
        "Ce notebook explore les données historiques collectées depuis Binance.\n",
        "\n",
        "## Objectifs\n",
        "1. Charger et visualiser les données collectées\n",
        "2. Analyser les statistiques descriptives\n",
        "3. Détecter les valeurs manquantes et anomalies\n",
        "4. Visualiser les tendances de prix et volumes\n",
        "5. Analyser les corrélations entre cryptos\n",
        "6. Calculer les rendements et volatilité"
    ]),

    # Imports
    create_cell("code", [
        "# Imports\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.graph_objects as go\n",
        "import plotly.express as px\n",
        "from plotly.subplots import make_subplots\n",
        "from pathlib import Path\n",
        "import warnings\n",
        "from datetime import datetime, timedelta\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Configuration des graphiques\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline\n",
        "\n",
        "# Configuration pandas\n",
        "pd.set_option('display.max_columns', None)\n",
        "pd.set_option('display.max_rows', 100)\n",
        "pd.set_option('display.float_format', '{:.2f}'.format)\n",
        "\n",
        "print(\"✅ Imports réussis!\")\n",
        "print(f\"📅 Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")"
    ]),

    # Section 1
    create_cell("markdown", ["## 1. 📁 Chargement des Données"]),

    create_cell("code", [
        "# Chemin des données\n",
        "data_path = Path('../data/raw/historical')\n",
        "\n",
        "# Lister tous les fichiers CSV\n",
        "csv_files = list(data_path.glob('*.csv'))\n",
        "\n",
        "print(f\"📂 Fichiers CSV trouvés: {len(csv_files)}\")\n",
        "print(\"\\n📋 Liste des fichiers CSV:\")\n",
        "for f in csv_files:\n",
        "    size_kb = f.stat().st_size / 1024\n",
        "    print(f\"  • {f.name} ({size_kb:.2f} KB)\")"
    ]),

    create_cell("code", [
        "# Fonction pour charger tous les fichiers\n",
        "def load_all_crypto_data(data_path):\n",
        "    crypto_data = {}\n",
        "    for csv_file in data_path.glob('*.csv'):\n",
        "        try:\n",
        "            df = pd.read_csv(csv_file)\n",
        "            df['timestamp'] = pd.to_datetime(df['timestamp'])\n",
        "            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else csv_file.stem\n",
        "            crypto_data[symbol] = df\n",
        "            print(f\"✓ {symbol}: {len(df)} lignes chargées\")\n",
        "        except Exception as e:\n",
        "            print(f\"✗ Erreur avec {csv_file.name}: {e}\")\n",
        "    return crypto_data\n",
        "\n",
        "crypto_data = load_all_crypto_data(data_path)\n",
        "print(f\"\\n✅ {len(crypto_data)} cryptos chargées avec succès!\")"
    ]),

    # Section 2
    create_cell("markdown", ["## 2. 🔍 Aperçu des Données"]),

    create_cell("code", [
        "# Afficher un aperçu\n",
        "for symbol, df in crypto_data.items():\n",
        "    print(f\"\\n{'='*70}\")\n",
        "    print(f\"📊 {symbol}\")\n",
        "    print(f\"{'='*70}\")\n",
        "    print(f\"Période: {df['timestamp'].min()} à {df['timestamp'].max()}\")\n",
        "    print(f\"Bougies: {len(df)}\")\n",
        "    display(df.head())\n",
        "    display(df.describe())"
    ]),

    # Section 3
    create_cell("markdown", ["## 3. 📊 Visualisation des Prix"]),

    create_cell("code", [
        "# Graphique des prix\n",
        "fig = go.Figure()\n",
        "for symbol, df in crypto_data.items():\n",
        "    fig.add_trace(go.Scatter(\n",
        "        x=df['timestamp'],\n",
        "        y=df['close'],\n",
        "        mode='lines',\n",
        "        name=symbol\n",
        "    ))\n",
        "\n",
        "fig.update_layout(\n",
        "    title='📈 Évolution des Prix',\n",
        "    xaxis_title='Date',\n",
        "    yaxis_title='Prix (USD)',\n",
        "    height=600,\n",
        "    template='plotly_dark'\n",
        ")\n",
        "fig.show()"
    ]),

    # Section finale
    create_cell("markdown", [
        "## 🎯 Prochaines Étapes\n",
        "\n",
        "1. Feature Engineering - Indicateurs techniques\n",
        "2. Base de données - TimescaleDB\n",
        "3. Modélisation ML\n",
        "4. Prédictions"
    ])
]

notebook_content["cells"] = cells

# Créer le dossier notebooks s'il n'existe pas
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

# Sauvegarder le notebook
output_file = notebooks_dir / "01_data_exploration.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook créé avec succès: {output_file}")
print(f"\n🚀 Pour l'utiliser:")
print(f"   jupyter notebook {output_file}")
print(f"   # ou")
print(f"   jupyter lab")