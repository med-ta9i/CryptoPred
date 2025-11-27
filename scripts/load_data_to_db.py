#!/usr/bin/env python3
# scripts/load_data_to_db.py
"""
Script pour charger les données CSV collectées dans la base de données TimescaleDB.
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
    ║        📊 CHARGEMENT DES DONNÉES DANS TIMESCALEDB 📊        ║
    ║                                                              ║
    ║              Import des données historiques CSV              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_connection(config):
    """
    Crée une connexion à la base de données.

    Args:
        config: Configuration de l'application

    Returns:
        Connection PostgreSQL
    """
    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database=config.database.name,
            user=config.database.user,
            password=config.database.password
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        raise


def ensure_crypto_exists(conn, symbol, base_currency, quote_currency):
    """
    S'assure que la crypto existe dans crypto_metadata.

    Args:
        conn: Connexion PostgreSQL
        symbol: Symbole (ex: BTC/USDT)
        base_currency: Devise de base
        quote_currency: Devise de cotation
    """
    cursor = conn.cursor()

    try:
        # Vérifier si existe
        cursor.execute(
            "SELECT id FROM crypto_metadata WHERE symbol = %s",
            (symbol,)
        )

        if cursor.fetchone() is None:
            # Insérer si n'existe pas
            cursor.execute("""
                           INSERT INTO crypto_metadata (symbol, base_currency, quote_currency)
                           VALUES (%s, %s, %s) ON CONFLICT (symbol) DO NOTHING
                           """, (symbol, base_currency, quote_currency))
            conn.commit()
            logger.info(f"  ✓ Crypto {symbol} ajoutée dans crypto_metadata")

    finally:
        cursor.close()


def load_csv_to_db(csv_file, conn, timeframe='1h'):
    """
    Charge un fichier CSV dans la base de données.

    Args:
        csv_file: Chemin du fichier CSV
        conn: Connexion PostgreSQL
        timeframe: Intervalle de temps

    Returns:
        Nombre de lignes insérées
    """
    try:
        # Lire le CSV
        logger.info(f"📖 Lecture du fichier: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Vérifier les colonnes nécessaires
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ Colonnes manquantes dans {csv_file.name}")
            return 0

        # Convertir timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extraire le symbole
        symbol = df['symbol'].iloc[0]

        # Extraire les devises
        if '/' in symbol:
            base_currency, quote_currency = symbol.split('/')
        else:
            base_currency, quote_currency = symbol, 'USDT'

        # S'assurer que la crypto existe
        ensure_crypto_exists(conn, symbol, base_currency, quote_currency)

        # Préparer les données pour l'insertion
        records = []
        for _, row in df.iterrows():
            record = (
                row['timestamp'],
                symbol,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row.get('quote_volume', 0)),
                int(row.get('trades_count', 0)),
                timeframe
            )
            records.append(record)

        # Insertion batch avec ON CONFLICT
        cursor = conn.cursor()

        insert_query = """
                       INSERT INTO ohlcv_data
                       (time, symbol, open, high, low, close, volume, quote_volume, trades_count, timeframe)
                       VALUES %s ON CONFLICT (time, symbol, timeframe) DO NOTHING \
                       """

        # Utiliser execute_values pour l'insertion batch (plus rapide)
        execute_values(cursor, insert_query, records, page_size=1000)
        conn.commit()

        inserted_count = cursor.rowcount
        cursor.close()

        logger.info(f"  ✅ {inserted_count} bougies insérées pour {symbol}")

        return inserted_count

    except Exception as e:
        logger.error(f"  ❌ Erreur lors du chargement de {csv_file.name}: {e}")
        conn.rollback()
        return 0


def get_database_stats(conn):
    """
    Récupère les statistiques de la base de données.

    Args:
        conn: Connexion PostgreSQL

    Returns:
        dict: Statistiques
    """
    cursor = conn.cursor()
    stats = {}

    try:
        # Nombre de cryptos
        cursor.execute("SELECT COUNT(*) FROM crypto_metadata")
        stats['crypto_count'] = cursor.fetchone()[0]

        # Nombre total de bougies
        cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
        stats['total_candles'] = cursor.fetchone()[0]

        # Bougies par crypto
        cursor.execute("""
                       SELECT symbol, COUNT(*) as count
                       FROM ohlcv_data
                       GROUP BY symbol
                       ORDER BY count DESC
                       """)
        stats['candles_per_crypto'] = cursor.fetchall()

        # Période couverte
        cursor.execute("""
                       SELECT MIN(time) as first_date,
                              MAX(time) as last_date
                       FROM ohlcv_data
                       """)
        result = cursor.fetchone()
        stats['first_date'] = result[0]
        stats['last_date'] = result[1]

        # Taille de la base
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('ohlcv_data'))
        """)
        stats['ohlcv_size'] = cursor.fetchone()[0]

    finally:
        cursor.close()

    return stats


def display_stats(stats):
    """
    Affiche les statistiques de la base.

    Args:
        stats: Dictionnaire de statistiques
    """
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES DE LA BASE DE DONNÉES")
    print("=" * 70)

    print(f"\n💰 Cryptomonnaies: {stats['crypto_count']}")
    print(f"📈 Total de bougies: {stats['total_candles']:,}")

    if stats['first_date'] and stats['last_date']:
        print(f"📅 Période: {stats['first_date']} à {stats['last_date']}")
        duration = stats['last_date'] - stats['first_date']
        print(f"⏱️  Durée: {duration.days} jours")

    print(f"💾 Taille table ohlcv_data: {stats['ohlcv_size']}")

    if stats['candles_per_crypto']:
        print(f"\n📊 Bougies par crypto:")
        for symbol, count in stats['candles_per_crypto']:
            print(f"   • {symbol}: {count:,} bougies")

    print("\n" + "=" * 70 + "\n")


def main():
    """Fonction principale."""
    print_banner()

    try:
        # Charger la configuration
        config = get_config()

        # Chemin des données
        data_path = Path("data/raw/historical")

        if not data_path.exists():
            logger.error(f"❌ Dossier introuvable: {data_path}")
            logger.info("💡 Collectez d'abord des données avec:")
            logger.info("   python scripts/collect_data.py --symbols BTC/USDT ETH/USDT --days 30")
            sys.exit(1)

        # Lister les fichiers CSV
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            logger.error(f"❌ Aucun fichier CSV trouvé dans {data_path}")
            sys.exit(1)

        logger.info(f"📂 {len(csv_files)} fichiers CSV trouvés")

        # Connexion à la base
        logger.info("📡 Connexion à la base de données...")
        conn = get_connection(config)
        logger.info("✅ Connexion établie")

        # Charger chaque fichier
        total_inserted = 0
        successful_files = 0

        print("\n🚀 Chargement des données...\n")

        for csv_file in tqdm(csv_files, desc="Chargement"):
            inserted = load_csv_to_db(csv_file, conn)
            if inserted > 0:
                total_inserted += inserted
                successful_files += 1

        # Afficher les résultats
        print("\n" + "=" * 70)
        print("📊 RÉSULTAT DU CHARGEMENT")
        print("=" * 70)
        print(f"✅ Fichiers traités avec succès: {successful_files}/{len(csv_files)}")
        print(f"📈 Total de bougies insérées: {total_inserted:,}")
        print("=" * 70)

        # Afficher les statistiques de la base
        stats = get_database_stats(conn)
        display_stats(stats)

        # Fermer la connexion
        conn.close()

        # Succès !
        print("🎉" * 35)
        logger.info("🎉 CHARGEMENT TERMINÉ AVEC SUCCÈS! 🎉")
        print("🎉" * 35)

        print("\n📚 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("1. Vérifier les données:")
        print("   psql -U taki -d crypto_prediction")
        print("   SELECT * FROM ohlcv_data LIMIT 10;")
        print()
        print("2. Calculer les features:")
        print("   python scripts/calculate_features.py")
        print()
        print("3. Lancer le notebook d'analyse:")
        print("   jupyter notebook notebooks/02_database_analysis.ipynb")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()