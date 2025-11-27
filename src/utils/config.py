# src/utils/config.py
"""
Module de gestion de la configuration du projet.
Charge les variables d'environnement et valide les configurations.
Compatible avec Pydantic V2.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import yaml

# Charger les variables d'environnement
load_dotenv()


class BinanceConfig(BaseModel):
    """Configuration pour l'API Binance."""
    api_key: Optional[str] = Field(default=None, description="Clé API Binance")
    api_secret: Optional[str] = Field(default=None, description="Secret API Binance")
    testnet: bool = Field(default=False, description="Utiliser le testnet")
    rate_limit_pause: float = Field(default=1.0, description="Pause entre requêtes (secondes)")


class DatabaseConfig(BaseModel):
    """Configuration pour la base de données PostgreSQL/TimescaleDB."""
    host: str = Field(default="localhost", description="Hôte de la base de données")
    port: int = Field(default=5432, description="Port de la base de données")
    name: str = Field(default="crypto_prediction", description="Nom de la base de données")
    user: str = Field(default="postgres", description="Utilisateur de la base de données")
    password: str = Field(description="Mot de passe de la base de données")

    @property
    def url(self) -> str:
        """Génère l'URL de connexion PostgreSQL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """Génère l'URL de connexion asynchrone."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseModel):
    """Configuration pour Redis."""
    host: str = Field(default="localhost", description="Hôte Redis")
    port: int = Field(default=6379, description="Port Redis")
    password: Optional[str] = Field(default=None, description="Mot de passe Redis")
    db: int = Field(default=0, description="Numéro de base de données Redis")

    @property
    def url(self) -> str:
        """Génère l'URL de connexion Redis."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class APIConfig(BaseModel):
    """Configuration pour l'API REST."""
    host: str = Field(default="0.0.0.0", description="Hôte de l'API")
    port: int = Field(default=8000, description="Port de l'API")
    secret_key: str = Field(description="Clé secrète pour JWT")
    workers: int = Field(default=4, description="Nombre de workers")
    rate_limit: str = Field(default="100/minute", description="Limite de requêtes")

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("La clé secrète doit faire au moins 32 caractères")
        return v


class DataConfig(BaseModel):
    """Configuration pour les données."""
    symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        description="Symboles à collecter"
    )
    intervals: List[str] = Field(
        default=["1h", "4h", "1d"],
        description="Intervalles de temps"
    )
    lookback_days: int = Field(default=730, description="Jours d'historique")
    update_frequency_minutes: int = Field(default=60, description="Fréquence de mise à jour")


class ModelConfig(BaseModel):
    """Configuration pour les modèles ML."""
    train_size: float = Field(default=0.7, description="Proportion de données d'entraînement")
    validation_size: float = Field(default=0.15, description="Proportion de données de validation")
    test_size: float = Field(default=0.15, description="Proportion de données de test")
    random_state: int = Field(default=42, description="Seed aléatoire")
    prediction_horizons: List[int] = Field(
        default=[1, 6, 24, 168],
        description="Horizons de prédiction en heures"
    )
    enabled_models: List[str] = Field(
        default=["linear_regression", "random_forest", "xgboost", "lstm"],
        description="Modèles activés"
    )

    @field_validator('train_size', 'validation_size', 'test_size')
    @classmethod
    def validate_splits(cls, v):
        if not 0 < v < 1:
            raise ValueError("Les proportions doivent être entre 0 et 1")
        return v


class LoggingConfig(BaseModel):
    """Configuration pour le logging."""
    level: str = Field(default="INFO", description="Niveau de log")
    file_path: str = Field(default="logs/", description="Chemin des fichiers de log")

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Niveau de log invalide. Choisir parmi {valid_levels}")
        return v.upper()


class Config(BaseSettings):
    """Configuration globale de l'application."""

    # Métadonnées du projet
    project_name: str = Field(default="Crypto Price Prediction")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    timezone: str = Field(default="UTC")

    # Sous-configurations
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig(password="changeme"))
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api: APIConfig = Field(
        default_factory=lambda: APIConfig(secret_key="changeme_minimum_32_characters_required_for_security"))
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Chemins du projet
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Configuration Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra='ignore'
    )

    @property
    def data_dir(self) -> Path:
        """Chemin du dossier data."""
        return self.base_dir / "data"

    @property
    def models_dir(self) -> Path:
        """Chemin du dossier models."""
        return self.base_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Chemin du dossier logs."""
        return self.base_dir / "logs"

    def model_post_init(self, __context):
        """Appelé après l'initialisation du modèle."""
        self._load_env_overrides()
        self._load_yaml_config()
        self._create_directories()

    def _load_env_overrides(self):
        """Charge et override avec les variables d'environnement."""
        # Binance
        if os.getenv('BINANCE_API_KEY'):
            self.binance.api_key = os.getenv('BINANCE_API_KEY')
        if os.getenv('BINANCE_API_SECRET'):
            self.binance.api_secret = os.getenv('BINANCE_API_SECRET')
        if os.getenv('BINANCE_TESTNET'):
            self.binance.testnet = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'

        # Database
        if os.getenv('DB_HOST'):
            self.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self.database.name = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self.database.user = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.database.password = os.getenv('DB_PASSWORD')

        # Redis
        if os.getenv('REDIS_HOST'):
            self.redis.host = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            self.redis.port = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            self.redis.password = os.getenv('REDIS_PASSWORD')
        if os.getenv('REDIS_DB'):
            self.redis.db = int(os.getenv('REDIS_DB'))

        # API
        if os.getenv('API_HOST'):
            self.api.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.port = int(os.getenv('API_PORT'))
        if os.getenv('API_SECRET_KEY'):
            self.api.secret_key = os.getenv('API_SECRET_KEY')
        if os.getenv('API_WORKERS'):
            self.api.workers = int(os.getenv('API_WORKERS'))

        # Logging
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE_PATH'):
            self.logging.file_path = os.getenv('LOG_FILE_PATH')

        # Debug
        if os.getenv('DEBUG'):
            self.debug = os.getenv('DEBUG', 'False').lower() == 'true'

    def _load_yaml_config(self):
        """Charge la configuration depuis config.yaml si disponible."""
        yaml_path = self.base_dir / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Mise à jour avec les valeurs du YAML (priorité aux variables d'env)
            if 'data' in yaml_config and yaml_config['data']:
                if 'symbols' in yaml_config['data']:
                    self.data.symbols = yaml_config['data']['symbols']
                if 'intervals' in yaml_config['data']:
                    self.data.intervals = yaml_config['data']['intervals']
                if 'lookback_days' in yaml_config['data']:
                    self.data.lookback_days = yaml_config['data']['lookback_days']

            if 'models' in yaml_config and yaml_config['models']:
                if 'prediction_horizons' in yaml_config['models']:
                    self.models.prediction_horizons = yaml_config['models']['prediction_horizons']
                if 'enabled_models' in yaml_config['models']:
                    self.models.enabled_models = yaml_config['models']['enabled_models']

    def _create_directories(self):
        """Crée les dossiers nécessaires s'ils n'existent pas."""
        directories = [
            self.data_dir / "raw" / "historical",
            self.data_dir / "raw" / "realtime",
            self.data_dir / "processed",
            self.data_dir / "features",
            self.models_dir / "checkpoints",
            self.models_dir / "production",
            self.logs_dir / "data_collection",
            self.logs_dir / "model_training",
            self.logs_dir / "api",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_required_configs(self) -> bool:
        """
        Valide que toutes les configurations requises sont présentes.

        Returns:
            True si valide, sinon lève une exception
        """
        # Ne pas valider en mode développement
        if self.database.password == "changeme":
            print("⚠️  Utilisation de la configuration par défaut. Configurez .env pour la production.")
            return True

        return True

    def display_config(self):
        """Affiche la configuration actuelle (sans les secrets)."""
        print("\n" + "=" * 60)
        print("⚙️  CONFIGURATION DU PROJET")
        print("=" * 60)

        print(f"\n📦 Projet: {self.project_name} v{self.version}")
        print(f"🐛 Mode Debug: {self.debug}")
        print(f"🌍 Timezone: {self.timezone}")

        print(f"\n💰 Binance:")
        print(f"  • API Key configurée: {'✓' if self.binance.api_key else '✗'}")
        print(f"  • Testnet: {self.binance.testnet}")

        print(f"\n💾 Base de données:")
        print(f"  • Host: {self.database.host}:{self.database.port}")
        print(f"  • Database: {self.database.name}")
        print(f"  • User: {self.database.user}")

        print(f"\n🔴 Redis:")
        print(f"  • Host: {self.redis.host}:{self.redis.port}")
        print(f"  • DB: {self.redis.db}")

        print(f"\n🌐 API:")
        print(f"  • Endpoint: http://{self.api.host}:{self.api.port}")
        print(f"  • Workers: {self.api.workers}")
        print(f"  • Rate Limit: {self.api.rate_limit}")

        print(f"\n📊 Données:")
        print(f"  • Symboles: {len(self.data.symbols)} cryptos")
        print(f"  • Intervalles: {', '.join(self.data.intervals)}")
        print(f"  • Historique: {self.data.lookback_days} jours")

        print(f"\n🤖 Modèles:")
        print(f"  • Train/Val/Test: {self.models.train_size}/{self.models.validation_size}/{self.models.test_size}")
        print(f"  • Horizons: {self.models.prediction_horizons}")
        print(f"  • Modèles actifs: {len(self.models.enabled_models)}")

        print(f"\n📝 Logging:")
        print(f"  • Niveau: {self.logging.level}")
        print(f"  • Dossier: {self.logging.file_path}")

        print("\n" + "=" * 60 + "\n")


# Instance globale de configuration
try:
    config = Config()
except Exception as e:
    print(f"⚠️  Erreur lors du chargement de la configuration: {e}")
    print("📝 Assurez-vous d'avoir créé un fichier .env avec toutes les variables nécessaires")
    print("💡 Copiez .env.example vers .env et remplissez les valeurs")
    raise


# Fonction utilitaire pour obtenir la config
def get_config() -> Config:
    """
    Retourne l'instance de configuration globale.

    Returns:
        Instance de Config
    """
    return config


if __name__ == "__main__":
    # Test de la configuration
    try:
        cfg = get_config()
        cfg.validate_required_configs()
        cfg.display_config()
        print("✅ Configuration valide!")
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        import traceback

        traceback.print_exc()