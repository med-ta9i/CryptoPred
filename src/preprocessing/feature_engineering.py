# src/preprocessing/feature_engineering.py
"""
Module de feature engineering pour les données crypto.
Calcule les indicateurs techniques et les features pour le ML.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Classe pour calculer les indicateurs techniques.
    """

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'close', periods: List[int] = [7, 25, 99]) -> pd.DataFrame:
        """
        Calcule les moyennes mobiles simples (SMA).

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            periods: Liste des périodes

        Returns:
            DataFrame avec les SMA ajoutées
        """
        df = df.copy()
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()

        logger.debug(f"SMA calculées pour les périodes: {periods}")
        return df

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'close', periods: List[int] = [7, 25, 99]) -> pd.DataFrame:
        """
        Calcule les moyennes mobiles exponentielles (EMA).

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            periods: Liste des périodes

        Returns:
            DataFrame avec les EMA ajoutées
        """
        df = df.copy()
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()

        logger.debug(f"EMA calculées pour les périodes: {periods}")
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
        """
        Calcule le RSI (Relative Strength Index).

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            period: Période pour le calcul

        Returns:
            DataFrame avec RSI ajouté
        """
        df = df.copy()

        # Calcul des variations
        delta = df[column].diff()

        # Séparer gains et pertes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Moyennes exponentielles
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        # RSI
        rs = avg_gain / avg_loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        logger.debug(f"RSI calculé avec période {period}")
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, column: str = 'close',
                       fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calcule le MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            fast: Période rapide
            slow: Période lente
            signal: Période du signal

        Returns:
            DataFrame avec MACD ajouté
        """
        df = df.copy()

        # EMA rapide et lente
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()

        # MACD
        df['macd'] = ema_fast - ema_slow

        # Signal
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

        # Histogramme
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        logger.debug(f"MACD calculé (fast={fast}, slow={slow}, signal={signal})")
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'close',
                                  period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """
        Calcule les bandes de Bollinger.

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            period: Période pour la moyenne
            std: Nombre d'écarts-types

        Returns:
            DataFrame avec Bollinger Bands ajoutées
        """
        df = df.copy()

        # Moyenne mobile
        df['bb_middle'] = df[column].rolling(window=period).mean()

        # Écart-type
        rolling_std = df[column].rolling(window=period).std()

        # Bandes supérieure et inférieure
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std)

        # Largeur des bandes (indicateur de volatilité)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        logger.debug(f"Bollinger Bands calculées (period={period}, std={std})")
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calcule l'ATR (Average True Range) - mesure de volatilité.

        Args:
            df: DataFrame avec les données (doit avoir high, low, close)
            period: Période pour le calcul

        Returns:
            DataFrame avec ATR ajouté
        """
        df = df.copy()

        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR (moyenne mobile du TR)
        df[f'atr_{period}'] = tr.ewm(span=period, adjust=False).mean()

        logger.debug(f"ATR calculé avec période {period}")
        return df

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
        """
        Calcule l'oscillateur stochastique.

        Args:
            df: DataFrame avec les données
            period: Période pour le calcul
            smooth: Période de lissage

        Returns:
            DataFrame avec stochastique ajouté
        """
        df = df.copy()

        # Stochastique %K
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        # Stochastique %D (lissage de %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

        logger.debug(f"Stochastique calculé (period={period}, smooth={smooth})")
        return df


class FeatureEngineering:
    """
    Classe principale pour le feature engineering.
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features basées sur les prix.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()

        # Rendements
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Rendements cumulés
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

        # Variations de prix
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100

        # Range
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = (df['high'] - df['low']) / df['low'] * 100

        logger.debug("Price features calculées")
        return df

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features basées sur le volume.

        Args:
            df: DataFrame avec les données

        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()

        # Moyenne mobile du volume
        df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_sma_25'] = df['volume'].rolling(window=25).mean()

        # Ratio volume actuel / moyenne
        df['volume_ratio'] = df['volume'] / df['volume_sma_25']

        # Volume pondéré par le prix
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        logger.debug("Volume features calculées")
        return df

    def calculate_volatility_features(self, df: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Calcule les features de volatilité.

        Args:
            df: DataFrame avec les données
            windows: Fenêtres temporelles

        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()

        for window in windows:
            # Volatilité des rendements
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

            # Volatilité des prix
            df[f'price_volatility_{window}'] = df['close'].rolling(window=window).std()

        logger.debug(f"Volatility features calculées pour les fenêtres: {windows}")
        return df

    def calculate_temporal_features(self, df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
        """
        Calcule les features temporelles.

        Args:
            df: DataFrame avec les données
            time_column: Nom de la colonne temporelle

        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()

        # S'assurer que la colonne est datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])

        # Features temporelles
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['day_of_month'] = df[time_column].dt.day
        df['month'] = df[time_column].dt.month
        df['quarter'] = df[time_column].dt.quarter
        df['year'] = df[time_column].dt.year

        # Indicateurs cycliques (sin/cos pour capturer la nature cyclique)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        logger.debug("Temporal features calculées")
        return df

    def calculate_lag_features(self, df: pd.DataFrame, column: str = 'close',
                               lags: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Calcule les lag features (valeurs passées).

        Args:
            df: DataFrame avec les données
            column: Colonne à utiliser
            lags: Liste des lags

        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()

        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)

        logger.debug(f"Lag features calculées pour les lags: {lags}")
        return df

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les features.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            DataFrame avec toutes les features
        """
        logger.info("Début du calcul de toutes les features...")

        df = df.copy()

        # Trier par temps
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)

        # 1. Features de prix
        df = self.calculate_price_features(df)

        # 2. Indicateurs techniques
        df = self.indicators.calculate_sma(df, periods=[7, 25, 99])
        df = self.indicators.calculate_ema(df, periods=[7, 25, 99])
        df = self.indicators.calculate_rsi(df, period=14)
        df = self.indicators.calculate_macd(df)
        df = self.indicators.calculate_bollinger_bands(df)
        df = self.indicators.calculate_atr(df)
        df = self.indicators.calculate_stochastic(df)

        # 3. Features de volume
        df = self.calculate_volume_features(df)

        # 4. Features de volatilité
        df = self.calculate_volatility_features(df)

        # 5. Features temporelles
        if 'time' in df.columns:
            df = self.calculate_temporal_features(df)

        # 6. Lag features
        df = self.calculate_lag_features(df, column='close', lags=[1, 3, 6, 12, 24])

        logger.info(f"✅ Toutes les features calculées. Total de colonnes: {len(df.columns)}")

        return df

    def get_feature_names(self, exclude_original: bool = True) -> List[str]:
        """
        Retourne la liste des noms de features calculées.

        Args:
            exclude_original: Exclure les colonnes OHLCV originales

        Returns:
            Liste des noms de features
        """
        # Liste de toutes les features possibles
        features = []

        # Price features
        features.extend(['returns', 'log_returns', 'cumulative_returns',
                         'price_change', 'price_change_pct', 'daily_range', 'daily_range_pct'])

        # Technical indicators
        features.extend([f'sma_{p}' for p in [7, 25, 99]])
        features.extend([f'ema_{p}' for p in [7, 25, 99]])
        features.extend(['rsi_14', 'macd', 'macd_signal', 'macd_histogram'])
        features.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width'])
        features.extend(['atr_14', 'stoch_k', 'stoch_d'])

        # Volume features
        features.extend(['volume_sma_7', 'volume_sma_25', 'volume_ratio', 'vwap'])

        # Volatility features
        features.extend([f'volatility_{w}' for w in [7, 14, 30]])
        features.extend([f'price_volatility_{w}' for w in [7, 14, 30]])

        # Temporal features
        features.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                         'is_weekend'])

        # Lag features
        features.extend([f'close_lag_{lag}' for lag in [1, 3, 6, 12, 24]])

        return features


# Fonctions utilitaires
def prepare_features_for_ml(df: pd.DataFrame, target_column: str = 'close',
                            horizon: int = 1) -> tuple:
    """
    Prépare les features pour le machine learning.

    Args:
        df: DataFrame avec toutes les features
        target_column: Colonne cible à prédire
        horizon: Horizon de prédiction (nombre de périodes)

    Returns:
        X (features), y (target)
    """
    df = df.copy()

    # Créer la target (prix futur)
    df[f'target_{horizon}'] = df[target_column].shift(-horizon)

    # Supprimer les lignes avec NaN
    df = df.dropna()

    # Colonnes à exclure
    exclude_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                    'timeframe', 'created_at', f'target_{horizon}']

    # Features
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[f'target_{horizon}']

    logger.info(f"Features préparées: {len(feature_cols)} features, {len(df)} samples")

    return X, y


if __name__ == "__main__":
    # Test du module
    logging.basicConfig(level=logging.INFO)

    # Créer des données de test
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df_test = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.rand(1000) * 1000000
    })

    # Calculer les features
    fe = FeatureEngineering()
    df_features = fe.calculate_all_features(df_test)

    print(f"\n✅ Test réussi!")
    print(f"📊 Colonnes originales: {len(df_test.columns)}")
    print(f"📈 Colonnes avec features: {len(df_features.columns)}")
    print(f"🎯 Features ajoutées: {len(df_features.columns) - len(df_test.columns)}")
    print(f"\n📋 Liste des features:")
    print(fe.get_feature_names())