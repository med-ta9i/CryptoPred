# src/models/base_model.py
"""
Classe de base abstraite pour tous les modèles de prédiction.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Classe abstraite de base pour tous les modèles de prédiction.
    """

    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialise le modèle.

        Args:
            model_name: Nom du modèle
            model_params: Paramètres du modèle
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.is_trained = False
        self.metadata = {
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        self.metrics = {}
        self.feature_names = []
        self.scaler = None

    @abstractmethod
    def build_model(self):
        """Construit le modèle. À implémenter par les sous-classes."""
        pass

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entraîne le modèle.

        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions.

        Args:
            X: Features

        Returns:
            Prédictions
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Évalue le modèle.

        Args:
            X: Features
            y: Target réel

        Returns:
            Dictionnaire de métriques
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")

        y_pred = self.predict(X)

        # Calculer les métriques
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2': r2_score(y, y_pred)
        }

        # Directional accuracy (pour le trading)
        y_series = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y)
        pred_series = pd.Series(y_pred).reset_index(drop=True)
        actual_direction = np.sign(y_series.diff())
        pred_direction = np.sign(pred_series.diff())
        # Ignorer les NaN du premier élément
        valid_mask = ~(actual_direction.isna() | pred_direction.isna())
        if valid_mask.sum() > 0:
            metrics['directional_accuracy'] = np.mean(actual_direction[valid_mask] == pred_direction[valid_mask]) * 100
        else:
            metrics['directional_accuracy'] = 0.0

        self.metrics.update(metrics)

        logger.info(f"Métriques d'évaluation: {metrics}")

        return metrics

    def save(self, path: Path, save_model: bool = True):
        """
        Sauvegarde le modèle.

        Args:
            path: Chemin de sauvegarde
            save_model: Sauvegarder le modèle aussi
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les métadonnées
        metadata_file = path / f"{self.model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Sauvegarder les métriques
        metrics_file = path / f"{self.model_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Sauvegarder le modèle
        if save_model and self.model is not None:
            model_file = path / f"{self.model_name}_model.joblib"
            joblib.dump(self.model, model_file)
            logger.info(f"Modèle sauvegardé: {model_file}")

        # Sauvegarder le scaler si présent
        if self.scaler is not None:
            scaler_file = path / f"{self.model_name}_scaler.joblib"
            joblib.dump(self.scaler, scaler_file)

        logger.info(f"Modèle et métadonnées sauvegardés dans: {path}")

    def load(self, path: Path):
        """
        Charge le modèle.

        Args:
            path: Chemin du modèle
        """
        path = Path(path)

        # Charger les métadonnées
        metadata_file = path / f"{self.model_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)

        # Charger les métriques
        metrics_file = path / f"{self.model_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)

        # Charger le modèle
        model_file = path / f"{self.model_name}_model.joblib"
        if model_file.exists():
            self.model = joblib.load(model_file)
            self.is_trained = True
            logger.info(f"Modèle chargé: {model_file}")

        # Charger le scaler
        scaler_file = path / f"{self.model_name}_scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)

        logger.info(f"Modèle chargé depuis: {path}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Retourne l'importance des features si disponible.

        Returns:
            DataFrame avec l'importance des features
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Ce modèle ne supporte pas l'importance des features")
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def predict_with_confidence(self, X: pd.DataFrame,
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fait des prédictions avec intervalles de confiance.

        Args:
            X: Features
            confidence_level: Niveau de confiance

        Returns:
            (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(X)

        # Calculer les intervalles basés sur l'erreur historique
        if 'rmse' in self.metrics:
            margin = self.metrics['rmse'] * 1.96  # 95% de confiance
            lower_bound = predictions - margin
            upper_bound = predictions + margin
        else:
            # Par défaut, ±5%
            margin = predictions * 0.05
            lower_bound = predictions - margin
            upper_bound = predictions + margin

        return predictions, lower_bound, upper_bound

    def __str__(self):
        """Représentation string du modèle."""
        return f"{self.model_name}(trained={self.is_trained}, params={self.model_params})"

    def __repr__(self):
        """Représentation du modèle."""
        return self.__str__()


class SklearnModelWrapper(BaseModel):
    """
    Wrapper pour les modèles scikit-learn.
    """

    def __init__(self, model_name: str, model_class, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialise le wrapper.

        Args:
            model_name: Nom du modèle
            model_class: Classe du modèle sklearn
            model_params: Paramètres du modèle
        """
        super().__init__(model_name, model_params)
        self.model_class = model_class
        self.build_model()

    def build_model(self):
        """Construit le modèle sklearn."""
        self.model = self.model_class(**self.model_params)
        logger.info(f"Modèle {self.model_name} construit avec params: {self.model_params}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entraîne le modèle sklearn.

        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (non utilisé pour sklearn classique)
            y_val: Target de validation (non utilisé pour sklearn classique)
        """
        logger.info(f"Entraînement de {self.model_name}...")

        # Sauvegarder les noms de features
        self.feature_names = list(X_train.columns)

        # Entraîner le modèle
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        self.is_trained = True
        self.metadata['training_time'] = training_time
        self.metadata['n_features'] = len(self.feature_names)
        self.metadata['n_samples'] = len(X_train)

        logger.info(f"✅ {self.model_name} entraîné en {training_time:.2f}s")

        # Évaluer sur le train set
        train_metrics = self.evaluate(X_train, y_train)
        logger.info(f"Métriques train: RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}")

        # Évaluer sur le validation set si fourni
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Métriques validation: RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions.

        Args:
            X: Features

        Returns:
            Prédictions
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        return self.model.predict(X)


if __name__ == "__main__":
    # Test du module
    logging.basicConfig(level=logging.INFO)

    # Créer des données de test
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Données aléatoires
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(X['feature_0'] * 2 + X['feature_1'] * 3 + np.random.randn(1000) * 0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test du wrapper
    model = SklearnModelWrapper(
        model_name="linear_regression_test",
        model_class=LinearRegression,
        model_params={}
    )

    # Entraîner
    model.fit(X_train, y_train)

    # Évaluer
    metrics = model.evaluate(X_test, y_test)
    print(f"\n✅ Test réussi!")
    print(f"📊 Métriques: {metrics}")

    # Sauvegarder
    model.save(Path("models/test"))
    print(f"💾 Modèle sauvegardé")