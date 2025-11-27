#!/usr/bin/env python3
# scripts/collect_data.py
"""
Script de collecte de données historiques depuis Binance.
Permet de collecter facilement des données pour plusieurs cryptos.

Utilisation:
    python scripts/collect_data.py --top 20 --days 730
    python scripts/collect_data.py --symbols BTC/USDT ETH/USDT --days 365
    python scripts/collect_data.py --all --days 30 --timeframe 4h
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors import BinanceCollector


def setup_logging(log_level: str = "INFO") -> None:
    """Configure le système de logging."""
    log_dir = Path("logs/data_collection")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Logs sauvegardés dans: {log_file}")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Collecte de données historiques depuis Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Collecter les top 20 cryptos sur 2 ans
  python scripts/collect_data.py --top 20 --days 730

  # Collecter des cryptos spécifiques sur 1 an
  python scripts/collect_data.py --symbols BTC/USDT ETH/USDT BNB/USDT --days 365

  # Collecter avec un intervalle de 4h
  python scripts/collect_data.py --top 10 --days 90 --timeframe 4h

  # Collecter toutes les cryptos disponibles (attention!)
  python scripts/collect_data.py --all --days 30 --timeframe 1d
        """
    )

    # Groupe pour le choix des symboles
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--top',
        type=int,
        help='Nombre de cryptos top par volume (ex: --top 20)'
    )
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='Liste de symboles spécifiques (ex: --symbols BTC/USDT ETH/USDT)'
    )
    symbol_group.add_argument(
        '--all',
        action='store_true',
        help='Collecter TOUTES les cryptos disponibles (peut prendre longtemps!)'
    )

    # Paramètres de collecte
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Nombre de jours d\'historique à collecter (défaut: 365)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
        help='Intervalle de temps (défaut: 1h)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/historical',
        help='Dossier de sortie (défaut: data/raw/historical)'
    )

    parser.add_argument(
        '--quote',
        type=str,
        default='USDT',
        choices=['USDT', 'BTC', 'ETH', 'BNB'],
        help='Devise de cotation (défaut: USDT)'
    )

    parser.add_argument(
        '--min-volume',
        type=float,
        default=1000000,
        help='Volume minimum 24h en $ (défaut: 1000000)'
    )

    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Pause entre requêtes en secondes (défaut: 1.0)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Niveau de logging (défaut: INFO)'
    )

    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Utiliser le testnet Binance'
    )

    return parser.parse_args()


def print_banner():
    """Affiche le banner du script."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🚀 CRYPTO PRICE PREDICTION - DATA COLLECTOR 🚀       ║
    ║                                                              ║
    ║              Collecte de données depuis Binance              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_collection_summary(results: dict, args: argparse.Namespace):
    """Affiche un résumé de la collecte."""
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE LA COLLECTE")
    print("=" * 70)

    print(f"\n📅 Période collectée: {args.days} jours")
    print(f"⏱️  Intervalle: {args.timeframe}")
    print(f"💰 Devise: {args.quote}")
    print(f"\n✅ Cryptos collectées avec succès: {len(results)}")

    if results:
        print(f"\n📈 Détails par crypto:")
        print("-" * 70)

        total_candles = 0
        total_size = 0

        for symbol, df in results.items():
            candles = len(df)
            total_candles += candles

            # Estimation de la taille en MB
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            total_size += size_mb

            print(f"\n  {symbol}:")
            print(f"    • Bougies: {candles:,}")
            print(f"    • Période: {df['timestamp'].min()} à {df['timestamp'].max()}")
            print(f"    • Prix min/max: ${df['low'].min():.2f} / ${df['high'].max():.2f}")
            print(f"    • Volume total: ${df['volume'].sum():,.0f}")
            print(f"    • Taille: {size_mb:.2f} MB")

        print("\n" + "-" * 70)
        print(f"\n📊 TOTAUX:")
        print(f"  • Total bougies: {total_candles:,}")
        print(f"  • Taille totale: {total_size:.2f} MB")
        print(f"  • Fichiers sauvegardés dans: {args.output}")

    print("\n" + "=" * 70)


def main():
    """Fonction principale."""
    # Affichage du banner
    print_banner()

    # Parse des arguments
    args = parse_arguments()

    # Configuration du logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Affichage de la configuration
    print("\n⚙️  Configuration de la collecte:")
    print(f"  • Mode: ", end="")
    if args.top:
        print(f"Top {args.top} cryptos")
    elif args.symbols:
        print(f"{len(args.symbols)} symboles spécifiques")
    elif args.all:
        print("TOUTES les cryptos disponibles")

    print(f"  • Historique: {args.days} jours")
    print(f"  • Intervalle: {args.timeframe}")
    print(f"  • Devise: {args.quote}")
    print(f"  • Volume min: ${args.min_volume:,.0f}")
    print(f"  • Sortie: {args.output}")
    print(f"  • Testnet: {'Oui' if args.testnet else 'Non'}")
    print()

    try:
        # Initialisation du collecteur
        logger.info("Initialisation du collecteur Binance...")
        collector = BinanceCollector(
            testnet=args.testnet,
            rate_limit_pause=args.rate_limit
        )

        # Détermination des symboles à collecter
        symbols: List[str] = []

        if args.top:
            logger.info(f"Récupération des top {args.top} cryptos...")
            symbols = collector.get_top_symbols(
                top_n=args.top,
                quote_currency=args.quote
            )

        elif args.symbols:
            symbols = args.symbols
            logger.info(f"Collecte de {len(symbols)} symboles spécifiques")

        elif args.all:
            logger.info("Récupération de TOUTES les cryptos disponibles...")
            symbols = collector.get_available_symbols(
                quote_currency=args.quote,
                min_volume_24h=args.min_volume
            )

            # Confirmation pour éviter les accidents
            response = input(f"\n⚠️  Vous allez collecter {len(symbols)} cryptos. Continuer? [y/N] ")
            if response.lower() != 'y':
                logger.info("Collecte annulée par l'utilisateur")
                return

        if not symbols:
            logger.error("Aucun symbole à collecter!")
            return

        logger.info(f"\n📋 Symboles à collecter: {', '.join(symbols[:10])}")
        if len(symbols) > 10:
            logger.info(f"   ... et {len(symbols) - 10} autres")

        # Calcul des dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

        # Collecte des données
        logger.info(f"\n🚀 Démarrage de la collecte...")
        logger.info(f"   De {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}")
        print()

        results = collector.collect_multiple_symbols(
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            save_path=Path(args.output)
        )

        # Affichage du résumé
        print_collection_summary(results, args)

        # Génération d'un rapport JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'results': {
                'total_symbols': len(results),
                'symbols': list(results.keys()),
                'total_candles': sum(len(df) for df in results.values())
            }
        }

        report_file = Path(args.output) / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✅ Collecte terminée avec succès!")
        logger.info(f"📄 Rapport sauvegardé: {report_file}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Collecte interrompue par l'utilisateur")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Erreur lors de la collecte: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()