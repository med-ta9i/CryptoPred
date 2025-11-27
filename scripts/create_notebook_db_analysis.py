#!/usr/bin/env python3
# scripts/create_notebook_db_analysis.py
"""
Script pour créer le notebook d'analyse des données depuis la base de données.
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
            "codemirror_mode": {"name": "ipython", "version": 3},
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


# Cellules du notebook
cells = [
    # Titre
    create_cell("markdown", [
        "# 💾 Analyse des Données depuis TimescaleDB\n",
        "\n",
        "Ce notebook analyse les données crypto stockées dans la base de données TimescaleDB.\n",
        "\n",
        "## Objectifs\n",
        "1. Se connecter à la base de données\n",
        "2. Explorer les données avec SQL\n",
        "3. Analyser les statistiques avancées\n",
        "4. Calculer les indicateurs techniques\n",
        "5. Visualiser les données avec Plotly\n",
        "6. Comparer les performances des cryptos"
    ]),

    # Imports
    create_cell("code", [
        "# Imports\n",
        "import sys\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.graph_objects as go\n",
        "import plotly.express as px\n",
        "from plotly.subplots import make_subplots\n",
        "import psycopg2\n",
        "from datetime import datetime, timedelta\n",
        "import warnings\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Ajouter le dossier racine au path\n",
        "sys.path.insert(0, str(Path.cwd().parent))\n",
        "\n",
        "from src.utils.config import get_config\n",
        "\n",
        "# Configuration\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "%matplotlib inline\n",
        "\n",
        "pd.set_option('display.max_columns', None)\n",
        "pd.set_option('display.float_format', '{:.2f}'.format)\n",
        "\n",
        "print(\"✅ Imports réussis!\")\n",
        "print(f\"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")"
    ]),

    # Connexion
    create_cell("markdown", ["## 1. 🔌 Connexion à la Base de Données"]),

    create_cell("code", [
        "# Charger la configuration\n",
        "config = get_config()\n",
        "\n",
        "# Connexion à PostgreSQL\n",
        "def get_connection():\n",
        "    return psycopg2.connect(\n",
        "        host=config.database.host,\n",
        "        port=config.database.port,\n",
        "        database=config.database.name,\n",
        "        user=config.database.user,\n",
        "        password=config.database.password\n",
        "    )\n",
        "\n",
        "# Fonction helper pour exécuter des requêtes\n",
        "def query_to_df(query, params=None):\n",
        "    \"\"\"Exécute une requête SQL et retourne un DataFrame.\"\"\"\n",
        "    conn = get_connection()\n",
        "    try:\n",
        "        df = pd.read_sql_query(query, conn, params=params)\n",
        "        return df\n",
        "    finally:\n",
        "        conn.close()\n",
        "\n",
        "# Test de connexion\n",
        "try:\n",
        "    conn = get_connection()\n",
        "    print(\"✅ Connexion à TimescaleDB réussie!\")\n",
        "    print(f\"📊 Base: {config.database.name}\")\n",
        "    print(f\"🏠 Host: {config.database.host}\")\n",
        "    conn.close()\n",
        "except Exception as e:\n",
        "    print(f\"❌ Erreur de connexion: {e}\")"
    ]),

    # Vue d'ensemble
    create_cell("markdown", ["## 2. 📊 Vue d'Ensemble des Données"]),

    create_cell("code", [
        "# Statistiques générales\n",
        "stats_query = \"\"\"\n",
        "SELECT \n",
        "    COUNT(DISTINCT symbol) as nb_cryptos,\n",
        "    COUNT(*) as total_candles,\n",
        "    MIN(time) as first_date,\n",
        "    MAX(time) as last_date,\n",
        "    pg_size_pretty(pg_total_relation_size('ohlcv_data')) as table_size\n",
        "FROM ohlcv_data;\n",
        "\"\"\"\n",
        "\n",
        "stats = query_to_df(stats_query)\n",
        "print(\"=\"*70)\n",
        "print(\"📈 STATISTIQUES GÉNÉRALES\")\n",
        "print(\"=\"*70)\n",
        "display(stats)\n",
        "\n",
        "# Durée de la période\n",
        "duration = stats['last_date'].iloc[0] - stats['first_date'].iloc[0]\n",
        "print(f\"\\n⏱️  Période couverte: {duration.days} jours\")"
    ]),

    create_cell("code", [
        "# Données par crypto\n",
        "crypto_stats_query = \"\"\"\n",
        "SELECT \n",
        "    symbol,\n",
        "    COUNT(*) as nb_candles,\n",
        "    MIN(time) as first_date,\n",
        "    MAX(time) as last_date,\n",
        "    ROUND(AVG(close)::numeric, 2) as avg_price,\n",
        "    ROUND(MIN(low)::numeric, 2) as min_price,\n",
        "    ROUND(MAX(high)::numeric, 2) as max_price,\n",
        "    ROUND(SUM(volume)::numeric, 2) as total_volume\n",
        "FROM ohlcv_data\n",
        "GROUP BY symbol\n",
        "ORDER BY total_volume DESC;\n",
        "\"\"\"\n",
        "\n",
        "crypto_stats = query_to_df(crypto_stats_query)\n",
        "print(\"\\n📊 STATISTIQUES PAR CRYPTO\")\n",
        "print(\"=\"*70)\n",
        "display(crypto_stats)"
    ]),

    # Requêtes avancées
    create_cell("markdown", ["## 3. 🔍 Requêtes SQL Avancées avec TimescaleDB"]),

    create_cell("code", [
        "# Utiliser time_bucket pour agréger par jour\n",
        "daily_query = \"\"\"\n",
        "SELECT \n",
        "    time_bucket('1 day', time) AS day,\n",
        "    symbol,\n",
        "    FIRST(open, time) as day_open,\n",
        "    MAX(high) as day_high,\n",
        "    MIN(low) as day_low,\n",
        "    LAST(close, time) as day_close,\n",
        "    SUM(volume) as day_volume\n",
        "FROM ohlcv_data\n",
        "GROUP BY day, symbol\n",
        "ORDER BY day DESC, symbol\n",
        "LIMIT 50;\n",
        "\"\"\"\n",
        "\n",
        "daily_data = query_to_df(daily_query)\n",
        "print(\"📅 DONNÉES QUOTIDIENNES (time_bucket)\")\n",
        "print(\"=\"*70)\n",
        "display(daily_data.head(10))"
    ]),

    create_cell("code", [
        "# Performance des dernières 24h\n",
        "performance_24h_query = \"\"\"\n",
        "WITH latest AS (\n",
        "    SELECT DISTINCT ON (symbol)\n",
        "        symbol,\n",
        "        close as current_price,\n",
        "        time as current_time\n",
        "    FROM ohlcv_data\n",
        "    ORDER BY symbol, time DESC\n",
        "),\n",
        "day_ago AS (\n",
        "    SELECT DISTINCT ON (symbol)\n",
        "        symbol,\n",
        "        close as price_24h_ago\n",
        "    FROM ohlcv_data\n",
        "    WHERE time <= NOW() - INTERVAL '24 hours'\n",
        "    ORDER BY symbol, time DESC\n",
        ")\n",
        "SELECT \n",
        "    l.symbol,\n",
        "    ROUND(l.current_price::numeric, 2) as current_price,\n",
        "    ROUND(d.price_24h_ago::numeric, 2) as price_24h_ago,\n",
        "    ROUND(((l.current_price - d.price_24h_ago) / d.price_24h_ago * 100)::numeric, 2) as change_24h_pct\n",
        "FROM latest l\n",
        "JOIN day_ago d ON l.symbol = d.symbol\n",
        "ORDER BY change_24h_pct DESC;\n",
        "\"\"\"\n",
        "\n",
        "performance = query_to_df(performance_24h_query)\n",
        "print(\"\\n📈 PERFORMANCE 24H\")\n",
        "print(\"=\"*70)\n",
        "display(performance)"
    ]),

    # Chargement des données pour analyse
    create_cell("markdown", ["## 4. 📥 Chargement des Données pour Analyse"]),

    create_cell("code", [
        "# Charger toutes les données\n",
        "def load_crypto_data(symbol=None, limit=None):\n",
        "    \"\"\"Charge les données OHLCV pour une ou toutes les cryptos.\"\"\"\n",
        "    query = \"SELECT * FROM ohlcv_data\"\n",
        "    \n",
        "    if symbol:\n",
        "        query += f\" WHERE symbol = '{symbol}'\"\n",
        "    \n",
        "    query += \" ORDER BY time\"\n",
        "    \n",
        "    if limit:\n",
        "        query += f\" LIMIT {limit}\"\n",
        "    \n",
        "    df = query_to_df(query)\n",
        "    df['time'] = pd.to_datetime(df['time'])\n",
        "    return df\n",
        "\n",
        "# Charger les données\n",
        "all_data = load_crypto_data()\n",
        "print(f\"✅ {len(all_data)} bougies chargées\")\n",
        "print(f\"📊 Cryptos: {all_data['symbol'].unique()}\")\n",
        "print(f\"📅 Période: {all_data['time'].min()} à {all_data['time'].max()}\")\n",
        "\n",
        "display(all_data.head())"
    ]),

    # Visualisations
    create_cell("markdown", ["## 5. 📊 Visualisations Interactives"]),

    create_cell("code", [
        "# Graphique des prix\n",
        "fig = go.Figure()\n",
        "\n",
        "for symbol in all_data['symbol'].unique():\n",
        "    df_symbol = all_data[all_data['symbol'] == symbol]\n",
        "    \n",
        "    fig.add_trace(go.Scatter(\n",
        "        x=df_symbol['time'],\n",
        "        y=df_symbol['close'],\n",
        "        mode='lines',\n",
        "        name=symbol,\n",
        "        hovertemplate='<b>%{fullData.name}</b><br>' +\n",
        "                      'Date: %{x}<br>' +\n",
        "                      'Prix: $%{y:.2f}<br>' +\n",
        "                      '<extra></extra>'\n",
        "    ))\n",
        "\n",
        "fig.update_layout(\n",
        "    title='📈 Évolution des Prix (depuis TimescaleDB)',\n",
        "    xaxis_title='Date',\n",
        "    yaxis_title='Prix (USD)',\n",
        "    height=600,\n",
        "    hovermode='x unified',\n",
        "    template='plotly_dark'\n",
        ")\n",
        "\n",
        "fig.show()"
    ]),

    create_cell("code", [
        "# Graphique en chandelier pour chaque crypto\n",
        "for symbol in all_data['symbol'].unique():\n",
        "    df_symbol = all_data[all_data['symbol'] == symbol].copy()\n",
        "    \n",
        "    fig = go.Figure(data=[go.Candlestick(\n",
        "        x=df_symbol['time'],\n",
        "        open=df_symbol['open'],\n",
        "        high=df_symbol['high'],\n",
        "        low=df_symbol['low'],\n",
        "        close=df_symbol['close'],\n",
        "        name=symbol\n",
        "    )])\n",
        "    \n",
        "    fig.update_layout(\n",
        "        title=f'🕯️ Chandelier - {symbol}',\n",
        "        xaxis_title='Date',\n",
        "        yaxis_title='Prix (USD)',\n",
        "        height=500,\n",
        "        template='plotly_dark',\n",
        "        xaxis_rangeslider_visible=False\n",
        "    )\n",
        "    \n",
        "    fig.show()"
    ]),

    # Analyse des volumes
    create_cell("markdown", ["## 6. 📊 Analyse des Volumes"]),

    create_cell("code", [
        "# Volume analysis avec SQL\n",
        "volume_query = \"\"\"\n",
        "SELECT \n",
        "    time,\n",
        "    symbol,\n",
        "    volume,\n",
        "    AVG(volume) OVER (\n",
        "        PARTITION BY symbol \n",
        "        ORDER BY time \n",
        "        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW\n",
        "    ) as volume_sma_24\n",
        "FROM ohlcv_data\n",
        "ORDER BY time;\n",
        "\"\"\"\n",
        "\n",
        "volume_data = query_to_df(volume_query)\n",
        "\n",
        "# Graphique des volumes\n",
        "for symbol in volume_data['symbol'].unique():\n",
        "    df_vol = volume_data[volume_data['symbol'] == symbol]\n",
        "    \n",
        "    fig = make_subplots(\n",
        "        rows=2, cols=1,\n",
        "        row_heights=[0.7, 0.3],\n",
        "        subplot_titles=(f'{symbol} - Prix', 'Volume'),\n",
        "        vertical_spacing=0.1\n",
        "    )\n",
        "    \n",
        "    # Prix\n",
        "    df_price = all_data[all_data['symbol'] == symbol]\n",
        "    fig.add_trace(\n",
        "        go.Scatter(x=df_price['time'], y=df_price['close'], name='Prix'),\n",
        "        row=1, col=1\n",
        "    )\n",
        "    \n",
        "    # Volume\n",
        "    fig.add_trace(\n",
        "        go.Bar(x=df_vol['time'], y=df_vol['volume'], name='Volume', marker_color='lightblue'),\n",
        "        row=2, col=1\n",
        "    )\n",
        "    \n",
        "    # Volume SMA\n",
        "    fig.add_trace(\n",
        "        go.Scatter(x=df_vol['time'], y=df_vol['volume_sma_24'], name='Volume SMA 24', line=dict(color='red')),\n",
        "        row=2, col=1\n",
        "    )\n",
        "    \n",
        "    fig.update_layout(height=600, template='plotly_dark', showlegend=True)\n",
        "    fig.show()"
    ]),

    # Indicateurs techniques
    create_cell("markdown", ["## 7. 📈 Calcul des Indicateurs Techniques avec SQL"]),

    create_cell("code", [
        "# Calculer les moyennes mobiles avec SQL (Window Functions)\n",
        "indicators_query = \"\"\"\n",
        "SELECT \n",
        "    time,\n",
        "    symbol,\n",
        "    close,\n",
        "    AVG(close) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as sma_7,\n",
        "    AVG(close) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as sma_25,\n",
        "    AVG(close) OVER (PARTITION BY symbol ORDER BY time ROWS BETWEEN 98 PRECEDING AND CURRENT ROW) as sma_99\n",
        "FROM ohlcv_data\n",
        "ORDER BY symbol, time;\n",
        "\"\"\"\n",
        "\n",
        "indicators = query_to_df(indicators_query)\n",
        "print(\"✅ Indicateurs calculés\")\n",
        "display(indicators.tail(10))"
    ]),

    create_cell("code", [
        "# Visualiser les moyennes mobiles\n",
        "for symbol in indicators['symbol'].unique():\n",
        "    df_ind = indicators[indicators['symbol'] == symbol].copy()\n",
        "    \n",
        "    fig = go.Figure()\n",
        "    \n",
        "    fig.add_trace(go.Scatter(x=df_ind['time'], y=df_ind['close'], name='Prix', line=dict(width=2)))\n",
        "    fig.add_trace(go.Scatter(x=df_ind['time'], y=df_ind['sma_7'], name='SMA 7', line=dict(dash='dash')))\n",
        "    fig.add_trace(go.Scatter(x=df_ind['time'], y=df_ind['sma_25'], name='SMA 25', line=dict(dash='dash')))\n",
        "    fig.add_trace(go.Scatter(x=df_ind['time'], y=df_ind['sma_99'], name='SMA 99', line=dict(dash='dot')))\n",
        "    \n",
        "    fig.update_layout(\n",
        "        title=f'📊 {symbol} - Prix et Moyennes Mobiles',\n",
        "        xaxis_title='Date',\n",
        "        yaxis_title='Prix (USD)',\n",
        "        height=600,\n",
        "        template='plotly_dark',\n",
        "        hovermode='x unified'\n",
        "    )\n",
        "    \n",
        "    fig.show()"
    ]),

    # Rendements
    create_cell("markdown", ["## 8. 💹 Analyse des Rendements"]),

    create_cell("code", [
        "# Calculer les rendements avec SQL\n",
        "returns_query = \"\"\"\n",
        "SELECT \n",
        "    time,\n",
        "    symbol,\n",
        "    close,\n",
        "    LAG(close) OVER (PARTITION BY symbol ORDER BY time) as prev_close,\n",
        "    (close - LAG(close) OVER (PARTITION BY symbol ORDER BY time)) / \n",
        "        LAG(close) OVER (PARTITION BY symbol ORDER BY time) * 100 as returns_pct\n",
        "FROM ohlcv_data\n",
        "ORDER BY symbol, time;\n",
        "\"\"\"\n",
        "\n",
        "returns_data = query_to_df(returns_query)\n",
        "returns_data = returns_data.dropna()\n",
        "\n",
        "print(\"📈 STATISTIQUES DES RENDEMENTS\")\n",
        "print(\"=\"*70)\n",
        "stats_returns = returns_data.groupby('symbol')['returns_pct'].agg([\n",
        "    ('Moyenne', 'mean'),\n",
        "    ('Médiane', 'median'),\n",
        "    ('Écart-type', 'std'),\n",
        "    ('Min', 'min'),\n",
        "    ('Max', 'max')\n",
        "]).round(4)\n",
        "\n",
        "display(stats_returns)"
    ]),

    create_cell("code", [
        "# Distribution des rendements\n",
        "for symbol in returns_data['symbol'].unique():\n",
        "    df_ret = returns_data[returns_data['symbol'] == symbol]\n",
        "    \n",
        "    fig = go.Figure()\n",
        "    fig.add_trace(go.Histogram(\n",
        "        x=df_ret['returns_pct'],\n",
        "        nbinsx=50,\n",
        "        name=symbol\n",
        "    ))\n",
        "    \n",
        "    fig.update_layout(\n",
        "        title=f'📊 {symbol} - Distribution des Rendements',\n",
        "        xaxis_title='Rendement (%)',\n",
        "        yaxis_title='Fréquence',\n",
        "        height=400,\n",
        "        template='plotly_dark'\n",
        "    )\n",
        "    \n",
        "    fig.show()"
    ]),

    # Corrélations
    create_cell("markdown", ["## 9. 🔗 Analyse des Corrélations"]),

    create_cell("code", [
        "# Créer une matrice de corrélation\n",
        "if len(all_data['symbol'].unique()) > 1:\n",
        "    # Pivot pour avoir les prix en colonnes\n",
        "    price_matrix = all_data.pivot(index='time', columns='symbol', values='close')\n",
        "    \n",
        "    # Calculer les rendements\n",
        "    returns_matrix = price_matrix.pct_change().dropna()\n",
        "    \n",
        "    # Corrélation\n",
        "    corr = returns_matrix.corr()\n",
        "    \n",
        "    # Heatmap\n",
        "    fig = go.Figure(data=go.Heatmap(\n",
        "        z=corr.values,\n",
        "        x=corr.columns,\n",
        "        y=corr.index,\n",
        "        colorscale='RdBu',\n",
        "        zmid=0,\n",
        "        text=corr.values,\n",
        "        texttemplate='%{text:.2f}',\n",
        "        textfont={\"size\": 12}\n",
        "    ))\n",
        "    \n",
        "    fig.update_layout(\n",
        "        title='🔗 Matrice de Corrélation des Rendements',\n",
        "        height=500,\n",
        "        template='plotly_dark'\n",
        "    )\n",
        "    \n",
        "    fig.show()\n",
        "else:\n",
        "    print(\"⚠️  Besoin d'au moins 2 cryptos pour calculer les corrélations\")"
    ]),

    # Résumé final
    create_cell("markdown", ["## 10. 📝 Résumé et Export"]),

    create_cell("code", [
        "# Résumé final\n",
        "summary_query = \"\"\"\n",
        "SELECT \n",
        "    symbol,\n",
        "    COUNT(*) as nb_candles,\n",
        "    ROUND(FIRST(close, time)::numeric, 2) as first_price,\n",
        "    ROUND(LAST(close, time)::numeric, 2) as last_price,\n",
        "    ROUND(((LAST(close, time) - FIRST(close, time)) / FIRST(close, time) * 100)::numeric, 2) as total_change_pct,\n",
        "    ROUND(MIN(low)::numeric, 2) as min_price,\n",
        "    ROUND(MAX(high)::numeric, 2) as max_price,\n",
        "    ROUND(AVG(volume)::numeric, 2) as avg_volume,\n",
        "    ROUND(SUM(volume)::numeric, 2) as total_volume\n",
        "FROM ohlcv_data\n",
        "GROUP BY symbol\n",
        "ORDER BY total_volume DESC;\n",
        "\"\"\"\n",
        "\n",
        "summary = query_to_df(summary_query)\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(\"📊 RÉSUMÉ COMPLET\")\n",
        "print(\"=\"*70)\n",
        "display(summary)\n",
        "\n",
        "print(\"\\n✅ Analyse terminée!\")"
    ]),

    create_cell("code", [
        "# Sauvegarder le résumé\n",
        "output_dir = Path('../output')\n",
        "output_dir.mkdir(exist_ok=True)\n",
        "\n",
        "summary.to_csv(output_dir / 'database_summary.csv', index=False)\n",
        "print(f\"💾 Résumé sauvegardé dans: {output_dir / 'database_summary.csv'}\")"
    ]),

    # Conclusion
    create_cell("markdown", [
        "## 🎯 Prochaines Étapes\n",
        "\n",
        "1. **Feature Engineering** - Calculer et stocker les indicateurs dans la table `features`\n",
        "2. **Modélisation ML** - Entraîner les premiers modèles\n",
        "3. **API** - Créer une API REST pour accéder aux données\n",
        "4. **Dashboard** - Créer un dashboard Streamlit temps réel\n",
        "\n",
        "---\n",
        "\n",
        "**📝 Notes:**\n",
        "- Toutes les analyses utilisent TimescaleDB\n",
        "- Les window functions SQL sont optimisées\n",
        "- Les données sont prêtes pour le ML"
    ])
]

notebook_content["cells"] = cells

# Créer le notebook
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

output_file = notebooks_dir / "02_database_analysis.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook créé: {output_file}")
print(f"\n🚀 Pour l'utiliser:")
print(f"   jupyter notebook {output_file}")
print(f"   # ou")
print(f"   jupyter lab")