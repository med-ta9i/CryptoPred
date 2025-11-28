#!/usr/bin/env python3
# scripts/train_model.py
"""
Script pour entraîner les modèles de prédiction de prix crypto.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.models.traditional_models import ModelFactory

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Affiche le banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🤖 ENTRAÎNEMENT DES MODÈLES ML - CRYPTO PRED 🤖      ║
    ║                                                              ║
    ║            Prédiction de prix avec Machine Learning          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_connection(config):
    """Crée une connexion à la base de données."""
    return psycopg2.connect(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password=config.database.password
    )


def load_features_from_db(conn, symbol, horizon=1):
    """
    Charge les features depuis la base de données.

    Args:
        conn: Connexion PostgreSQL
        symbol: Symbole de la crypto
        horizon: Horizon de prédiction (en heures)

    Returns:
        X (features), y (target), données complètes
    """
    logger.info(f"📥 Chargement des features pour {symbol}...")

    # Requête pour charger toutes les features
    query = """
            SELECT f.time, \
                   o.close, \
                   f.sma_7, \
                   f.sma_25, \
                   f.sma_99, \
                   f.ema_7, \
                   f.ema_25, \
                   f.ema_99, \
                   f.rsi_14, \
                   f.macd, \
                   f.macd_signal, \
                   f.macd_histogram, \
                   f.bb_upper, \
                   f.bb_middle, \
                   f.bb_lower, \
                   f.bb_width, \
                   f.returns, \
                   f.log_returns, \
                   f.volatility_24h, \
                   f.volume_sma_7, \
                   f.volume_ratio
            FROM features f
                     JOIN ohlcv_data o ON f.time = o.time AND f.symbol = o.symbol AND f.timeframe = o.timeframe
            WHERE f.symbol = %s
            ORDER BY f.time \
            """

    df = pd.read_sql_query(query, conn, params=[symbol])
    df['time'] = pd.to_datetime(df['time'])

    logger.info(f"✅ {len(df)} lignes chargées")

    if len(df) == 0:
        raise ValueError(f"Aucune donnée trouvée pour {symbol}")

    # Créer la target (prix futur)
    df['target'] = df['close'].shift(-horizon)

    # Supprimer les NaN
    df = df.dropna()

    # Séparer features et target
    feature_cols = [col for col in df.columns if col not in ['time', 'close', 'target']]
    X = df[feature_cols]
    y = df['target']

    logger.info(f"📊 Features: {len(feature_cols)} colonnes")
    logger.info(f"🎯 Target: prédire le prix dans {horizon}h")
    logger.info(f"📈 Samples: {len(X)}")

    return X, y, df


def prepare_train_test_split(X, y, test_size=0.2, val_size=0.1):
    """
    Sépare les données en train/val/test de manière temporelle.

    Args:
        X: Features
        y: Target
        test_size: Proportion du test set
        val_size: Proportion du validation set

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("✂️  Séparation des données...")

    # Split temporel (important pour les séries temporelles)
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    logger.info(f"  Train: {len(X_train)} samples ({len(X_train) / n * 100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val) / n * 100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test) / n * 100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Normalise les features.

    Args:
        X_train, X_val, X_test: Features

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    logger.info("📏 Normalisation des features...")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def plot_predictions(y_true, y_pred, title, save_path=None):
    """
    Affiche les prédictions vs valeurs réelles.

    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        title: Titre du graphique
        save_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Graphique 1: Série temporelle
    axes[0].plot(y_true.values, label='Réel', alpha=0.7)
    axes[0].plot(y_pred, label='Prédit', alpha=0.7)
    axes[0].set_title(f'{title} - Série Temporelle')
    axes[0].set_xlabel('Temps')
    axes[0].set_ylabel('Prix')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphique 2: Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1].set_title(f'{title} - Prédictions vs Réalité')
    axes[1].set_xlabel('Prix Réel')
    axes[1].set_ylabel('Prix Prédit')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Graphique sauvegardé: {save_path}")

    plt.close()


def plot_feature_importance(model, top_n=15, save_path=None):
    """
    Affiche l'importance des features.

    Args:
        model: Modèle entraîné
        top_n: Nombre de features à afficher
        save_path: Chemin de sauvegarde
    """
    importance_df = model.get_feature_importance()

    if importance_df is None:
        logger.warning("Ce modèle ne supporte pas l'importance des features")
        return

    # Top N features
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features les Plus Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Feature importance sauvegardée: {save_path}")

    plt.close()


def display_results(results, X_test, y_test):
    """
    Affiche les résultats de tous les modèles.

    Args:
        results: Résultats des modèles
        X_test: Features de test
        y_test: Target de test
    """
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("=" * 70)

    # Créer un DataFrame de comparaison
    comparison = []
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison.append({
            'Modèle': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE (%)': metrics['mape'],
            'R²': metrics['r2'],
            'Directional Acc (%)': metrics.get('directional_accuracy', 0)
        })

    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('RMSE')

    print("\n📈 Comparaison des Modèles:")
    print(df_comparison.to_string(index=False))

    # Meilleur modèle
    best_model_name = df_comparison.iloc[0]['Modèle']
    print(f"\n🏆 Meilleur Modèle: {best_model_name}")
    print(f"   RMSE: {df_comparison.iloc[0]['RMSE']:.4f}")
    print(f"   MAE:  {df_comparison.iloc[0]['MAE']:.4f}")
    print(f"   R²:   {df_comparison.iloc[0]['R²']:.4f}")

    print("\n" + "=" * 70)


def main():
    """Fonction principale."""
    print_banner()

    parser = argparse.ArgumentParser(description="Entraînement des modèles ML")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbole (défaut: BTC/USDT)')
    parser.add_argument('--horizon', type=int, default=1, help='Horizon de prédiction en heures (défaut: 1)')
    parser.add_argument('--models', nargs='+', default=['linear', 'random_forest', 'xgboost'],
                        help='Modèles à entraîner')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion du test set (défaut: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Proportion du validation set (défaut: 0.1)')
    parser.add_argument('--save', action='store_true', help='Sauvegarder les modèles')

    args = parser.parse_args()

    try:
        # Configuration
        config = get_config()

        logger.info(f"\n⚙️  Configuration:")
        logger.info(f"  Symbole: {args.symbol}")
        logger.info(f"  Horizon: {args.horizon}h")
        logger.info(f"  Modèles: {args.models}")
        logger.info(f"  Test size: {args.test_size}")
        logger.info(f"  Val size: {args.val_size}")

        # Connexion à la base
        logger.info("\n📡 Connexion à la base de données...")
        conn = get_connection(config)

        # Charger les données
        X, y, df_full = load_features_from_db(conn, args.symbol, args.horizon)
        conn.close()

        # Séparer les données
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
            X, y, args.test_size, args.val_size
        )

        # Normaliser
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test
        )

        # Entraîner les modèles
        logger.info(f"\n🤖 Entraînement des modèles: {args.models}")
        results = ModelFactory.train_multiple_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            model_types=args.models
        )

        # Évaluer sur le test set
        logger.info("\n📊 Évaluation sur le test set...")
        for model_name, result in results.items():
            model = result['model']
            test_metrics = model.evaluate(X_test_scaled, y_test)
            result['test_metrics'] = test_metrics
            logger.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.4f}")

        # Afficher les résultats
        display_results(results, X_test_scaled, y_test)

        # Créer les visualisations
        output_dir = Path(f"output/models/{args.symbol.replace('/', '_')}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name, result in results.items():
            model = result['model']

            # Prédictions sur le test set
            y_pred = model.predict(X_test_scaled)

            # Plot prédictions
            plot_predictions(
                y_test, y_pred,
                f"{model_name} - {args.symbol}",
                save_path=output_dir / f"{model_name}_predictions.png"
            )

            # Plot feature importance
            plot_feature_importance(
                model,
                save_path=output_dir / f"{model_name}_feature_importance.png"
            )

        # Sauvegarder les modèles
        if args.save:
            logger.info("\n💾 Sauvegarde des modèles...")
            models_dir = Path(f"models/production/{args.symbol.replace('/', '_')}")

            for model_name, result in results.items():
                model = result['model']
                model.save(models_dir / model_name)
                logger.info(f"  ✅ {model_name} sauvegardé")

        # Succès !
        print("\n" + "🎉" * 35)
        logger.info("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS! 🎉")
        print("🎉" * 35)

        print("\n📚 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("1. Visualiser les résultats:")
        print(f"   ls {output_dir}")
        print()
        print("2. Créer un notebook pour analyser les prédictions:")
        print("   jupyter notebook notebooks/04_model_analysis.ipynb")
        print()
        print("3. Créer l'API pour faire des prédictions:")
        print("   python scripts/create_api.py")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Entraînement interrompu")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()