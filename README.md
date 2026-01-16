# 🚀 Crypto Price Prediction

Système avancé de prédiction de prix de cryptomonnaies utilisant Machine Learning et Deep Learning.

## 📋 Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Documentation](#documentation)

## ✨ Fonctionnalités

- 📊 Collecte de données en temps réel depuis Binance
- 🔄 Pipeline ETL automatisé
- 🤖 Modèles ML/DL (Random Forest, XGBoost, LSTM)
- 📈 Dashboard interactif avec Streamlit
- 🔌 API REST pour les prédictions
- 💾 Stockage optimisé avec TimescaleDB
- ⚡ Cache Redis pour performance
- 📉 Analyse technique avancée
- 🎯 Prédictions multi-horizon

## 🏗️ Architecture

```
[Binance API] → [ETL Pipeline] → [TimescaleDB]
                                       ↓
                               [Feature Engineering]
                                       ↓
                                [ML/DL Models]
                                       ↓
                      [API] ← [Predictions] → [Dashboard]
```

## 🛠️ Technologies

- **Backend**: Python 3.9+, FastAPI
- **ML/DL**: Scikit-learn, TensorFlow, XGBoost
- **Big Data**: PySpark, Dask
- **Database**: PostgreSQL/TimescaleDB, Redis
- **Dashboard**: Streamlit, Plotly
- **Orchestration**: Apache Airflow
- **Containerization**: Docker

## 📦 Installation

### Prérequis

- Python 3.9+
- PostgreSQL 14+
- Redis 6+
- (Optionnel) Docker & Docker Compose

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/votre-username/crypto-price-prediction.git
cd crypto-price-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Copier le fichier de configuration
cp .env.example .env
# Éditer .env avec vos clés API

# Initialiser la base de données
python scripts/setup_database.py

# Collecter les données initiales
python scripts/collect_data.py
```

## 🚀 Utilisation

### Collecter des données

```bash
python scripts/collect_data.py --symbols BTC ETH --days 365
```

### Entraîner un modèle

```bash
python scripts/train_model.py --model xgboost --symbol BTC
```

### Lancer l'API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Lancer le Dashboard

```bash
streamlit run dashboard/app.py
```

## 📁 Structure du projet

Voir [docs/structure.md](docs/structure.md) pour la structure complète.

## 📚 Documentation

- [Guide d'installation](docs/installation.md)
- [Documentation API](docs/api_documentation.md)
- [Guide utilisateur](docs/user_guide.md)
- [Architecture système](docs/architecture.md)

## 👨‍💻 Auteur

M'hamed Taki 

