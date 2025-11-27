#!/usr/bin/env python3
# scripts/init_database.py
"""
Script d'initialisation de la base de données PostgreSQL/TimescaleDB.
Crée les tables, hypertables et charge le schéma initial.
"""

import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

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
    ║        💾 INITIALISATION BASE DE DONNÉES TIMESCALEDB 💾     ║
    ║                                                              ║
    ║             Création du schéma et des tables                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def test_connection(config):
    """
    Test la connexion à PostgreSQL.

    Args:
        config: Configuration de l'application

    Returns:
        bool: True si connexion réussie
    """
    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database="postgres",  # Se connecter à la base par défaut
            user=config.database.user,
            password=config.database.password
        )
        conn.close()
        logger.info("✅ Connexion à PostgreSQL réussie")
        return True
    except Exception as e:
        logger.error(f"❌ Échec de connexion à PostgreSQL: {e}")
        logger.error("\n💡 Vérifiez que:")
        logger.error("  1. PostgreSQL est bien démarré")
        logger.error("  2. Les identifiants dans .env sont corrects")
        logger.error("  3. L'utilisateur existe et a les bonnes permissions")
        return False


def database_exists(config):
    """
    Vérifie si la base de données existe.

    Args:
        config: Configuration de l'application

    Returns:
        bool: True si la base existe
    """
    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database="postgres",
            user=config.database.user,
            password=config.database.password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (config.database.name,)
        )
        exists = cursor.fetchone() is not None

        cursor.close()
        conn.close()

        return exists
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la base: {e}")
        return False


def create_database(config):
    """
    Crée la base de données si elle n'existe pas.

    Args:
        config: Configuration de l'application
    """
    try:
        if database_exists(config):
            logger.info(f"✓ La base de données '{config.database.name}' existe déjà")
            return True

        logger.info(f"Création de la base de données '{config.database.name}'...")

        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database="postgres",
            user=config.database.user,
            password=config.database.password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE {config.database.name}")

        cursor.close()
        conn.close()

        logger.info(f"✅ Base de données '{config.database.name}' créée avec succès")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de la base: {e}")
        return False


def execute_schema(config):
    """
    Exécute le fichier schema.sql.

    Args:
        config: Configuration de l'application
    """
    try:
        # Chemin du fichier schema.sql
        schema_file = Path(__file__).parent.parent / "src" / "database" / "schema.sql"

        if not schema_file.exists():
            logger.error(f"❌ Fichier schema.sql introuvable: {schema_file}")
            return False

        logger.info(f"📄 Lecture du fichier: {schema_file}")

        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        logger.info("🔧 Exécution du schéma...")

        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database=config.database.name,
            user=config.database.user,
            password=config.database.password
        )
        cursor = conn.cursor()

        # Exécuter le schéma
        cursor.execute(schema_sql)
        conn.commit()

        cursor.close()
        conn.close()

        logger.info("✅ Schéma créé avec succès")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution du schéma: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_installation(config):
    """
    Vérifie que tout est bien installé.

    Args:
        config: Configuration de l'application
    """
    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database=config.database.name,
            user=config.database.user,
            password=config.database.password
        )
        cursor = conn.cursor()

        print("\n" + "=" * 70)
        print("🔍 VÉRIFICATION DE L'INSTALLATION")
        print("=" * 70)

        # Vérifier TimescaleDB
        cursor.execute("""
                       SELECT default_version, installed_version
                       FROM pg_available_extensions
                       WHERE name = 'timescaledb'
                       """)
        result = cursor.fetchone()
        if result:
            logger.info(f"✅ TimescaleDB installé: version {result[1]}")
        else:
            logger.warning("⚠️  TimescaleDB n'est pas installé")

        # Lister les tables
        cursor.execute("""
                       SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema = 'public'
                       ORDER BY table_name
                       """)
        tables = cursor.fetchall()
        logger.info(f"\n📊 Tables créées ({len(tables)}):")
        for table in tables:
            logger.info(f"   • {table[0]}")

        # Lister les hypertables
        cursor.execute("""
                       SELECT hypertable_name
                       FROM timescaledb_information.hypertables
                       """)
        hypertables = cursor.fetchall()
        if hypertables:
            logger.info(f"\n⏰ Hypertables TimescaleDB ({len(hypertables)}):")
            for ht in hypertables:
                logger.info(f"   • {ht[0]}")

        # Vérifier les cryptos insérées
        cursor.execute("SELECT COUNT(*) FROM crypto_metadata")
        crypto_count = cursor.fetchone()[0]
        logger.info(f"\n💰 Cryptomonnaies enregistrées: {crypto_count}")

        if crypto_count > 0:
            cursor.execute("SELECT symbol, name FROM crypto_metadata ORDER BY symbol LIMIT 10")
            cryptos = cursor.fetchall()
            logger.info("\n📋 Exemples de cryptos:")
            for symbol, name in cryptos:
                logger.info(f"   • {symbol}: {name}")

        # Taille de la base
        cursor.execute("""
            SELECT pg_size_pretty(pg_database_size(%s))
        """, (config.database.name,))
        size = cursor.fetchone()[0]
        logger.info(f"\n💾 Taille de la base de données: {size}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 70)
        logger.info("✅ Vérification terminée avec succès")
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification: {e}")
        return False


def display_connection_info(config):
    """
    Affiche les informations de connexion.

    Args:
        config: Configuration de l'application
    """
    print("\n📝 INFORMATIONS DE CONNEXION")
    print("=" * 70)
    print(f"Host:     {config.database.host}")
    print(f"Port:     {config.database.port}")
    print(f"Database: {config.database.name}")
    print(f"User:     {config.database.user}")
    print(f"URL:      {config.database.url.replace(config.database.password, '***')}")
    print("=" * 70 + "\n")


def main():
    """Fonction principale."""
    print_banner()

    try:
        # Charger la configuration
        logger.info("⚙️  Chargement de la configuration...")
        config = get_config()

        display_connection_info(config)

        # Étape 1: Test de connexion
        logger.info("📡 Étape 1/4: Test de connexion à PostgreSQL...")
        if not test_connection(config):
            logger.error("\n💡 Pour installer PostgreSQL et TimescaleDB:")
            logger.error("   Consultez: docs/database_installation.md")
            sys.exit(1)

        # Étape 2: Créer la base de données
        logger.info("\n🏗️  Étape 2/4: Création de la base de données...")
        if not create_database(config):
            sys.exit(1)

        # Étape 3: Exécuter le schéma
        logger.info("\n📊 Étape 3/4: Création des tables et hypertables...")
        if not execute_schema(config):
            sys.exit(1)

        # Étape 4: Vérification
        logger.info("\n🔍 Étape 4/4: Vérification de l'installation...")
        if not verify_installation(config):
            sys.exit(1)

        # Succès !
        print("\n" + "🎉" * 35)
        logger.info("🎉 INSTALLATION TERMINÉE AVEC SUCCÈS! 🎉")
        print("🎉" * 35)

        print("\n📚 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("1. Charger des données dans la base:")
        print("   python scripts/load_data_to_db.py")
        print()
        print("2. Vérifier les données:")
        print("   psql -h localhost -U crypto_user -d crypto_prediction")
        print()
        print("3. Lancer l'API:")
        print("   uvicorn src.api.app:app --reload")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Opération annulée par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()