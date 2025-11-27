# src/data/collectors.py
"""
Module de collecte de données depuis l'API Binance.
Permet de récupérer les données historiques et en temps réel.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)


class BinanceCollector:
    """
    Collecteur de données depuis l'API Binance.

    Fonctionnalités :
    - Récupération de données historiques OHLCV
    - Support de plusieurs cryptos simultanément
    - Gestion automatique des limites de rate
    - Retry automatique en cas d'erreur
    - Sauvegarde progressive des données
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = False,
            rate_limit_pause: float = 1.0
    ):
        """
        Initialise le collecteur Binance.

        Args:
            api_key: Clé API Binance (optionnel pour données publiques)
            api_secret: Secret API Binance (optionnel pour données publiques)
            testnet: Utiliser le testnet au lieu du réseau principal
            rate_limit_pause: Pause en secondes entre les requêtes
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.rate_limit_pause = rate_limit_pause

        # Initialisation de l'exchange
        self.exchange = self._initialize_exchange()

        logger.info(f"BinanceCollector initialisé (testnet={testnet})")

    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialise la connexion à l'exchange Binance.

        Returns:
            Instance de l'exchange ccxt
        """
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }

        if self.api_key and self.api_secret:
            config['apiKey'] = self.api_key
            config['secret'] = self.api_secret

        if self.testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            }

        exchange = ccxt.binance(config)

        try:
            exchange.load_markets()
            logger.info(f"Connecté à Binance. {len(exchange.markets)} paires disponibles.")
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à Binance: {e}")
            raise

        return exchange

    def get_available_symbols(
            self,
            quote_currency: str = 'USDT',
            min_volume_24h: float = 1000000.0
    ) -> List[str]:
        """
        Récupère la liste des cryptos disponibles.

        Args:
            quote_currency: Devise de cotation (ex: USDT, BTC)
            min_volume_24h: Volume minimum sur 24h pour filtrer

        Returns:
            Liste des symboles (ex: ['BTC/USDT', 'ETH/USDT'])
        """
        try:
            # Récupérer les tickers pour avoir les volumes
            tickers = self.exchange.fetch_tickers()

            # Filtrer par devise de cotation et volume
            symbols = []
            for symbol, ticker in tickers.items():
                if quote_currency in symbol:
                    volume_24h = ticker.get('quoteVolume', 0)
                    if volume_24h and volume_24h >= min_volume_24h:
                        symbols.append(symbol)

            # Trier par volume décroissant
            symbols_with_volume = [
                (symbol, tickers[symbol].get('quoteVolume', 0))
                for symbol in symbols
            ]
            symbols_with_volume.sort(key=lambda x: x[1], reverse=True)

            sorted_symbols = [s[0] for s in symbols_with_volume]

            logger.info(f"{len(sorted_symbols)} symboles trouvés avec volume >= ${min_volume_24h:,.0f}")

            return sorted_symbols

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des symboles: {e}")
            raise

    def get_top_symbols(
            self,
            top_n: int = 20,
            quote_currency: str = 'USDT'
    ) -> List[str]:
        """
        Récupère les top N cryptos par volume.

        Args:
            top_n: Nombre de cryptos à retourner
            quote_currency: Devise de cotation

        Returns:
            Liste des top symboles
        """
        symbols = self.get_available_symbols(quote_currency=quote_currency)
        return symbols[:top_n]

    def get_historical_data(
            self,
            symbol: str,
            timeframe: str = '1h',
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 1000
    ) -> pd.DataFrame:
        """
        Récupère les données historiques OHLCV pour un symbole.

        Args:
            symbol: Symbole (ex: 'BTC/USDT')
            timeframe: Intervalle (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            start_date: Date de début (par défaut: 1 an avant)
            end_date: Date de fin (par défaut: maintenant)
            limit: Nombre max de bougies par requête

        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        # Dates par défaut
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        # Conversion en timestamps (millisecondes)
        since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        all_candles = []
        current_since = since

        logger.info(f"Collecte de {symbol} depuis {start_date} jusqu'à {end_date}")

        try:
            with tqdm(desc=f"Collecte {symbol}", unit="batch") as pbar:
                while current_since < end_ts:
                    try:
                        # Récupération des données
                        candles = self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=current_since,
                            limit=limit
                        )

                        if not candles:
                            break

                        all_candles.extend(candles)

                        # Mise à jour du timestamp de départ pour la prochaine requête
                        current_since = candles[-1][0] + 1

                        pbar.update(1)

                        # Pause pour respecter le rate limit
                        time.sleep(self.rate_limit_pause)

                    except ccxt.RateLimitExceeded:
                        logger.warning(f"Rate limit atteint pour {symbol}. Pause de 60s...")
                        time.sleep(60)
                        continue

                    except Exception as e:
                        logger.error(f"Erreur lors de la collecte de {symbol}: {e}")
                        time.sleep(5)
                        continue

            # Conversion en DataFrame
            if not all_candles:
                logger.warning(f"Aucune donnée collectée pour {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Conversion du timestamp en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Suppression des doublons
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            # Tri par date
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Ajout du symbole
            df['symbol'] = symbol

            logger.info(f"✓ {len(df)} bougies collectées pour {symbol}")

            return df

        except Exception as e:
            logger.error(f"Erreur critique lors de la collecte de {symbol}: {e}")
            raise

    def collect_multiple_symbols(
            self,
            symbols: List[str],
            timeframe: str = '1h',
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            save_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collecte les données pour plusieurs symboles.

        Args:
            symbols: Liste des symboles à collecter
            timeframe: Intervalle de temps
            start_date: Date de début
            end_date: Date de fin
            save_path: Chemin où sauvegarder les données (optionnel)

        Returns:
            Dictionnaire {symbol: DataFrame}
        """
        results = {}
        errors = []

        logger.info(f"Collecte de {len(symbols)} symboles...")

        for symbol in symbols:
            try:
                df = self.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    results[symbol] = df

                    # Sauvegarde si chemin fourni
                    if save_path:
                        self.save_to_csv(df, symbol, save_path, timeframe)

            except Exception as e:
                logger.error(f"Échec de collecte pour {symbol}: {e}")
                errors.append((symbol, str(e)))
                continue

        # Rapport final
        logger.info(f"\n{'=' * 50}")
        logger.info(f"RAPPORT DE COLLECTE")
        logger.info(f"{'=' * 50}")
        logger.info(f"✓ Succès: {len(results)}/{len(symbols)}")
        logger.info(f"✗ Échecs: {len(errors)}/{len(symbols)}")

        if errors:
            logger.warning("\nErreurs:")
            for symbol, error in errors:
                logger.warning(f"  - {symbol}: {error}")

        return results

    def save_to_csv(
            self,
            df: pd.DataFrame,
            symbol: str,
            save_path: Path,
            timeframe: str = '1h'
    ) -> None:
        """
        Sauvegarde les données dans un fichier CSV.

        Args:
            df: DataFrame à sauvegarder
            symbol: Nom du symbole
            save_path: Chemin du dossier de sauvegarde
            timeframe: Intervalle de temps
        """
        # Création du dossier si nécessaire
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Nom du fichier
        safe_symbol = symbol.replace('/', '_')
        filename = f"{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = save_path / filename

        # Sauvegarde
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Données sauvegardées: {filepath}")

    def get_market_summary(self, symbols: List[str]) -> pd.DataFrame:
        """
        Récupère un résumé du marché pour les symboles donnés.

        Args:
            symbols: Liste des symboles

        Returns:
            DataFrame avec les statistiques de marché
        """
        summary_data = []

        for symbol in tqdm(symbols, desc="Récupération des stats"):
            try:
                ticker = self.exchange.fetch_ticker(symbol)

                summary_data.append({
                    'symbol': symbol,
                    'last_price': ticker.get('last'),
                    'volume_24h': ticker.get('quoteVolume'),
                    'change_24h_pct': ticker.get('percentage'),
                    'high_24h': ticker.get('high'),
                    'low_24h': ticker.get('low'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'timestamp': pd.to_datetime(ticker.get('timestamp'), unit='ms')
                })

                time.sleep(self.rate_limit_pause)

            except Exception as e:
                logger.error(f"Erreur pour {symbol}: {e}")
                continue

        df = pd.DataFrame(summary_data)
        return df.sort_values('volume_24h', ascending=False).reset_index(drop=True)


# Fonction utilitaire pour utilisation rapide
def quick_collect(
        symbols: Optional[List[str]] = None,
        top_n: int = 10,
        days: int = 365,
        timeframe: str = '1h',
        save_path: str = 'data/raw/historical'
) -> Dict[str, pd.DataFrame]:
    """
    Fonction rapide pour collecter des données.

    Args:
        symbols: Liste de symboles (si None, prend les top N)
        top_n: Nombre de top cryptos si symbols=None
        days: Nombre de jours d'historique
        timeframe: Intervalle de temps
        save_path: Chemin de sauvegarde

    Returns:
        Dictionnaire des DataFrames collectés
    """
    collector = BinanceCollector()

    # Récupération des symboles
    if symbols is None:
        symbols = collector.get_top_symbols(top_n=top_n)

    # Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Collecte
    results = collector.collect_multiple_symbols(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        save_path=Path(save_path)
    )

    return results


if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Exemple d'utilisation
    print("🚀 Démarrage de la collecte de données Binance\n")

    # Collecte rapide des top 5 cryptos sur 30 jours
    results = quick_collect(
        top_n=5,
        days=30,
        timeframe='1h',
        save_path='data/raw/historical'
    )

    # Affichage des résultats
    print(f"\n📊 Résumé de la collecte:")
    for symbol, df in results.items():
        print(f"\n{symbol}:")
        print(f"  - Période: {df['timestamp'].min()} à {df['timestamp'].max()}")
        print(f"  - Nombre de bougies: {len(df)}")
        print(f"  - Prix min: ${df['low'].min():.2f}")
        print(f"  - Prix max: ${df['high'].max():.2f}")
        print(f"  - Volume total: ${df['volume'].sum():,.0f}")