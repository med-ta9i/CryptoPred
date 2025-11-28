#!/usr/bin/env python3
# scripts/calculate_features.py
"""
Script pour calculer les features et les stocker dans la base de données.
"""

import sys
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from tqdm import tqdm
from datetime import datetime

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.preprocessing.feature_engineering import FeatureEngineering

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Affiche le banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🔧 CALCUL DES FEATURES - FEATURE ENGINEERING 🔧      ║
    ║                                                              ║
    ║          Calcul et stockage des indicateurs techniques       ║
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


def load_ohlcv_data(conn, symbol=None, timeframe='1h'):
    """
    Charge les données OHLCV depuis la base.

    Args:
        conn: Connexion PostgreSQL
        symbol: Symbole spécifique ou None pour tous
        timeframe: Intervalle de temps

    Returns:
        DataFrame avec les données
    """
    query = """
            SELECT time, symbol, open, high, low, close, volume, timeframe
            FROM ohlcv_data
            WHERE timeframe = %s \
            """
    params = [timeframe]

    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)

    query += " ORDER BY symbol, time"

    df = pd.read_sql_query(query, conn, params=params)
    df['time'] = pd.to_datetime(df['time'])

    return df


def save_features_to_db(conn, features_df, symbol, timeframe='1h'):
    """
    Sauvegarde les features dans la base de données.

    Args:
        conn: Connexion PostgreSQL
        features_df: DataFrame avec les features
        symbol: Symbole de la crypto
        timeframe: Intervalle de temps

    Returns:
        Nombre de lignes insérées
    """
    # Colonnes à insérer (correspondant au schéma de la table features)
    feature_columns = [
        'sma_7', 'sma_25', 'sma_99',
        'ema_7', 'ema_25', 'ema_99',
        'rsi_14',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'returns', 'log_returns',
        'volatility_24h',
        'volume_sma_7', 'volume_ratio'
    ]

    # Préparer les données
    records = []
    for _, row in features_df.iterrows():
        # Vérifier que toutes les colonnes nécessaires existent
        if pd.isna(row['time']):
            continue

        record = [
            row['time'],
            symbol,
            timeframe
        ]

        # Ajouter les features (avec gestion des NaN)
        for col in feature_columns:
            if col == 'volatility_24h':
                # Utiliser volatility_7 comme proxy si disponible
                val = row.get('volatility_7', None)
            else:
                val = row.get(col, None)

            # Convertir NaN en None pour PostgreSQL
            if pd.isna(val):
                record.append(None)
            else:
                record.append(float(val))

        records.append(tuple(record))

    # Supprimer les enregistrements avec trop de NaN
    records = [r for r in records if sum(1 for x in r if x is None) < len(feature_columns) / 2]

    if not records:
        logger.warning(f"Aucune feature valide pour {symbol}")
        return 0

    # Insertion dans la base
    cursor = conn.cursor()

    insert_query = f"""
        INSERT INTO features 
            (time, symbol, timeframe, {', '.join(feature_columns)})
        VALUES %s
        ON CONFLICT (time, symbol, timeframe) 
        DO UPDATE SET
            {', '.join([f'{col} = EXCLUDED.{col}' for col in feature_columns])}
    """

    execute_values(cursor, insert_query, records, page_size=1000)
    conn.commit()

    inserted_count = cursor.rowcount
    cursor.close()

    return inserted_count


def calculate_and_store_features(config, symbol=None, timeframe='1h'):
    """
    Calcule et stocke les features pour une ou toutes les cryptos.

    Args:
        config: Configuration de l'application
        symbol: Symbole spécifique ou None pour tous
        timeframe: Intervalle de temps

    Returns:
        Dictionnaire avec les statistiques
    """
    logger.info("📡 Connexion à la base de données...")
    conn = get_connection(config)

    try:
        # Charger les données
        logger.info(f"📥 Chargement des données OHLCV (timeframe: {timeframe})...")
        df = load_ohlcv_data(conn, symbol, timeframe)

        if df.empty:
            logger.error("❌ Aucune donnée trouvée")
            return {}

        logger.info(f"✅ {len(df)} bougies chargées")

        # Obtenir la liste des symboles
        symbols = df['symbol'].unique()
        logger.info(f"💰 Traitement de {len(symbols)} crypto(s)")

        # Initialiser le feature engineering
        fe = FeatureEngineering()

        # Statistiques
        stats = {
            'total_processed': 0,
            'total_features': 0,
            'symbols': {}
        }

        # Traiter chaque crypto
        for sym in tqdm(symbols, desc="Calcul des features"):
            try:
                # Filtrer les données pour ce symbole
                df_symbol = df[df['symbol'] == sym].copy()

                logger.info(f"\n🔧 Traitement de {sym} ({len(df_symbol)} bougies)...")

                # Calculer les features
                df_features = fe.calculate_all_features(df_symbol)

                # Sauvegarder dans la base
                inserted = save_features_to_db(conn, df_features, sym, timeframe)

                logger.info(f"  ✅ {inserted} features insérées pour {sym}")

                stats['total_processed'] += len(df_features)
                stats['total_features'] += inserted
                stats['symbols'][sym] = {
                    'rows': len(df_features),
                    'features_inserted': inserted,
                    'feature_columns': len(df_features.columns) - 6  # Exclure colonnes OHLCV
                }

            except Exception as e:
                logger.error(f"  ❌ Erreur pour {sym}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return stats

    finally:
        conn.close()


def display_results(stats):
    """
    Affiche les résultats du calcul des features.

    Args:
        stats: Dictionnaire de statistiques
    """
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DU CALCUL DES FEATURES")
    print("=" * 70)

    if not stats:
        print("❌ Aucune feature calculée")
        return

    print(f"\n✅ Cryptos traitées: {len(stats['symbols'])}")
    print(f"📈 Total de lignes traitées: {stats['total_processed']:,}")
    print(f"💾 Total de features insérées: {stats['total_features']:,}")

    print(f"\n📊 Détails par crypto:")
    print("-" * 70)

    for symbol, info in stats['symbols'].items():
        print(f"\n  {symbol}:")
        print(f"    • Lignes: {info['rows']:,}")
        print(f"    • Features insérées: {info['features_inserted']:,}")
        print(f"    • Colonnes de features: {info['feature_columns']}")

    print("\n" + "=" * 70)


def verify_features_in_db(config):
    """
    Vérifie les features dans la base de données.

    Args:
        config: Configuration de l'application
    """
    conn = get_connection(config)

    try:
        cursor = conn.cursor()

        # Statistiques générales
        cursor.execute("""
                       SELECT COUNT(*)               as total_rows,
                              COUNT(DISTINCT symbol) as nb_symbols,
                              MIN(time)              as first_date,
                              MAX(time)              as last_date
                       FROM features
                       """)
        stats = cursor.fetchone()

        print("\n" + "=" * 70)
        print("🔍 VÉRIFICATION DES FEATURES DANS LA BASE")
        print("=" * 70)

        if stats[0] == 0:
            print("\n⚠️  Aucune feature trouvée dans la base")
        else:
            print(f"\n✅ Total de lignes: {stats[0]:,}")
            print(f"💰 Cryptos: {stats[1]}")
            print(f"📅 Période: {stats[2]} à {stats[3]}")

            # Détails par crypto
            cursor.execute("""
                           SELECT symbol,
                                  COUNT(*)      as nb_rows,
                                  COUNT(sma_7)  as sma_count,
                                  COUNT(rsi_14) as rsi_count,
                                  COUNT(macd)   as macd_count
                           FROM features
                           GROUP BY symbol
                           ORDER BY symbol
                           """)

            results = cursor.fetchall()
            print(f"\n📊 Détails par crypto:")
            print("-" * 70)
            for row in results:
                symbol, nb_rows, sma, rsi, macd = row
                print(f"\n  {symbol}:")
                print(f"    • Lignes: {nb_rows:,}")
                print(f"    • SMA calculées: {sma:,}")
                print(f"    • RSI calculées: {rsi:,}")
                print(f"    • MACD calculées: {macd:,}")

        cursor.close()
        print("\n" + "=" * 70)

    finally:
        conn.close()


def main():
    """Fonction principale."""
    print_banner()

    import argparse

    parser = argparse.ArgumentParser(description="Calcul des features pour les cryptos")
    parser.add_argument('--symbol', type=str, help='Symbole spécifique (ex: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Intervalle (défaut: 1h)')
    parser.add_argument('--verify-only', action='store_true', help='Uniquement vérifier les features existantes')

    args = parser.parse_args()

    try:
        # Charger la configuration
        config = get_config()

        # Mode vérification uniquement
        if args.verify_only:
            verify_features_in_db(config)
            return

        # Calculer et stocker les features
        logger.info("\n🚀 Démarrage du calcul des features...")

        if args.symbol:
            logger.info(f"📊 Traitement de {args.symbol} uniquement")
        else:
            logger.info("📊 Traitement de toutes les cryptos")

        stats = calculate_and_store_features(config, args.symbol, args.timeframe)

        # Afficher les résultats
        display_results(stats)

        # Vérifier dans la base
        verify_features_in_db(config)

        # Succès !
        print("\n" + "🎉" * 35)
        logger.info("🎉 CALCUL DES FEATURES TERMINÉ! 🎉")
        print("🎉" * 35)

        print("\n📚 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("1. Vérifier les features:")
        print("   psql -U taki -d crypto_prediction")
        print("   SELECT * FROM features LIMIT 10;")
        print()
        print("2. Créer un notebook pour analyser les features:")
        print("   jupyter notebook notebooks/03_feature_analysis.ipynb")
        print()
        print("3. Entraîner les modèles ML:")
        print("   python scripts/train_model.py")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Opération annulée")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()