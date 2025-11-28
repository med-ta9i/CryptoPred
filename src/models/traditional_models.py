# src/models/traditional_models.py
"""
Modèles de machine learning traditionnels pour la prédiction de prix crypto.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_model import SklearnModelWrapper

logger = logging.getLogger(__name__)


class LinearRegressionModel(SklearnModelWrapper):
    """
    Modèle de régression linéaire (baseline).
    """

    def __init__(self, regularization: str = 'none', alpha: float = 1.0):
        """
        Initialise le modèle de régression linéaire.

        Args:
            regularization: Type de régularisation ('none', 'ridge', 'lasso')
            alpha: Paramètre de régularisation
        """
        if regularization == 'ridge':
            model_class = Ridge
            model_params = {'alpha': alpha}
        elif regularization == 'lasso':
            model_class = Lasso
            model_params = {'alpha': alpha}
        else:
            model_class = LinearRegression
            model_params = {}

        super().__init__(
            model_name=f"linear_regression_{regularization}",
            model_class=model_class,
            model_params=model_params
        )


class RandomForestModel(SklearnModelWrapper):
    """
    Modèle Random Forest pour la prédiction.
    """

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, random_state: int = 42, **kwargs):
        """
        Initialise le modèle Random Forest.

        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale
            min_samples_split: Nombre minimum d'échantillons pour split
            random_state: Seed aléatoire
            **kwargs: Autres paramètres
        """
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            'n_jobs': -1,
            **kwargs
        }

        super().__init__(
            model_name="random_forest",
            model_class=RandomForestRegressor,
            model_params=model_params
        )


class XGBoostModel(SklearnModelWrapper):
    """
    Modèle XGBoost pour la prédiction.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """
        Initialise le modèle XGBoost.

        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale
            learning_rate: Taux d'apprentissage
            random_state: Seed aléatoire
            **kwargs: Autres paramètres
        """
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            **kwargs
        }

        super().__init__(
            model_name="xgboost",
            model_class=xgb.XGBRegressor,
            model_params=model_params
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entraîne XGBoost avec early stopping si validation set fourni.

        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation
            y_val: Target de validation
        """
        logger.info(f"Entraînement de {self.model_name}...")

        self.feature_names = list(X_train.columns)

        # Early stopping si validation set fourni
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True
        logger.info(f"✅ {self.model_name} entraîné")

        # Évaluer
        train_metrics = self.evaluate(X_train, y_train)
        logger.info(f"Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}")

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")


class LightGBMModel(SklearnModelWrapper):
    """
    Modèle LightGBM pour la prédiction.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = -1,
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """
        Initialise le modèle LightGBM.

        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale (-1 = pas de limite)
            learning_rate: Taux d'apprentissage
            random_state: Seed aléatoire
            **kwargs: Autres paramètres
        """
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'objective': 'regression',
            'n_jobs': -1,
            'verbose': -1,
            **kwargs
        }

        super().__init__(
            model_name="lightgbm",
            model_class=lgb.LGBMRegressor,
            model_params=model_params
        )


class ModelFactory:
    """
    Factory pour créer des modèles facilement.
    """

    @staticmethod
    def create_model(model_type: str, **kwargs):
        """
        Crée un modèle selon le type spécifié.

        Args:
            model_type: Type de modèle ('linear', 'random_forest', 'xgboost', 'lightgbm')
            **kwargs: Paramètres du modèle

        Returns:
            Instance du modèle
        """
        models = {
            'linear': LinearRegressionModel,
            'linear_ridge': lambda **k: LinearRegressionModel(regularization='ridge', **k),
            'linear_lasso': lambda **k: LinearRegressionModel(regularization='lasso', **k),
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel
        }

        if model_type not in models:
            raise ValueError(f"Type de modèle inconnu: {model_type}. Choisir parmi {list(models.keys())}")

        model = models[model_type](**kwargs)
        logger.info(f"Modèle créé: {model_type}")

        return model

    @staticmethod
    def train_multiple_models(X_train, y_train, X_val=None, y_val=None,
                              model_types: list = None) -> Dict[str, Any]:
        """
        Entraîne plusieurs modèles et compare leurs performances.

        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation
            y_val: Target de validation
            model_types: Liste des types de modèles à entraîner

        Returns:
            Dictionnaire avec les modèles et leurs performances
        """
        if model_types is None:
            model_types = ['linear', 'random_forest', 'xgboost']

        results = {}

        for model_type in model_types:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Entraînement: {model_type}")
            logger.info(f"{'=' * 70}")

            try:
                # Créer et entraîner le modèle
                model = ModelFactory.create_model(model_type)
                model.fit(X_train, y_train, X_val, y_val)

                # Évaluer
                if X_val is not None and y_val is not None:
                    val_metrics = model.evaluate(X_val, y_val)
                    results[model_type] = {
                        'model': model,
                        'metrics': val_metrics
                    }
                else:
                    train_metrics = model.evaluate(X_train, y_train)
                    results[model_type] = {
                        'model': model,
                        'metrics': train_metrics
                    }

            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_type}: {e}")
                continue

        # Trouver le meilleur modèle
        if results:
            best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
            logger.info(f"\n🏆 Meilleur modèle: {best_model_name}")
            logger.info(f"   RMSE: {results[best_model_name]['metrics']['rmse']:.4f}")

        return results


if __name__ == "__main__":
    # Test des modèles
    logging.basicConfig(level=logging.INFO)

    from sklearn.model_selection import train_test_split

    # Créer des données de test
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(X.sum(axis=1) + np.random.randn(1000) * 0.5)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("📊 Test des modèles traditionnels")
    print("=" * 70)

    # Entraîner plusieurs modèles
    results = ModelFactory.train_multiple_models(
        X_train, y_train,
        X_val, y_val,
        model_types=['linear', 'random_forest', 'xgboost']
    )

    # Afficher les résultats
    print("\n📈 COMPARAISON DES MODÈLES")
    print("=" * 70)
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")

    print("\n✅ Tests réussis!")