-- src/database/schema.sql
-- Schéma de base de données pour le projet Crypto Price Prediction
-- Compatible avec PostgreSQL et TimescaleDB

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- Activer l'extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE: crypto_metadata
-- Métadonnées des cryptomonnaies
-- ============================================================================

CREATE TABLE IF NOT EXISTS crypto_metadata (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100),
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    exchange VARCHAR(50) DEFAULT 'binance',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_symbol UNIQUE(symbol)
);

-- Index pour les recherches fréquentes
CREATE INDEX idx_crypto_symbol ON crypto_metadata(symbol);
CREATE INDEX idx_crypto_active ON crypto_metadata(is_active);

-- Commentaires
COMMENT ON TABLE crypto_metadata IS 'Métadonnées des cryptomonnaies suivies';
COMMENT ON COLUMN crypto_metadata.symbol IS 'Symbole de la paire (ex: BTC/USDT)';
COMMENT ON COLUMN crypto_metadata.base_currency IS 'Devise de base (ex: BTC)';
COMMENT ON COLUMN crypto_metadata.quote_currency IS 'Devise de cotation (ex: USDT)';

-- ============================================================================
-- TABLE: ohlcv_data (Hypertable TimescaleDB)
-- Données OHLCV (Open, High, Low, Close, Volume)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(30, 8) NOT NULL,
    quote_volume NUMERIC(30, 8),
    trades_count INTEGER,
    timeframe VARCHAR(10) NOT NULL DEFAULT '1h',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Contraintes
    CONSTRAINT ohlcv_time_symbol_timeframe_key UNIQUE(time, symbol, timeframe),
    CONSTRAINT check_prices CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close),
    CONSTRAINT check_volume CHECK (volume >= 0),

    -- Clé étrangère vers crypto_metadata
    CONSTRAINT fk_symbol FOREIGN KEY (symbol) REFERENCES crypto_metadata(symbol) ON DELETE CASCADE
);

-- Convertir en hypertable TimescaleDB (optimisé pour séries temporelles)
SELECT create_hypertable('ohlcv_data', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index pour optimiser les requêtes
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv_data (symbol, time DESC);
CREATE INDEX idx_ohlcv_timeframe ON ohlcv_data (timeframe, time DESC);
CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe, time DESC);

-- Commentaires
COMMENT ON TABLE ohlcv_data IS 'Données historiques OHLCV des cryptomonnaies';
COMMENT ON COLUMN ohlcv_data.time IS 'Timestamp de la bougie';
COMMENT ON COLUMN ohlcv_data.timeframe IS 'Intervalle de temps (1m, 5m, 1h, 4h, 1d)';

-- ============================================================================
-- TABLE: features
-- Features calculées pour le machine learning
-- ============================================================================

CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL DEFAULT '1h',

    -- Moyennes mobiles
    sma_7 NUMERIC(20, 8),
    sma_25 NUMERIC(20, 8),
    sma_99 NUMERIC(20, 8),
    ema_7 NUMERIC(20, 8),
    ema_25 NUMERIC(20, 8),
    ema_99 NUMERIC(20, 8),

    -- Indicateurs techniques
    rsi_14 NUMERIC(10, 4),
    macd NUMERIC(20, 8),
    macd_signal NUMERIC(20, 8),
    macd_histogram NUMERIC(20, 8),

    -- Bollinger Bands
    bb_upper NUMERIC(20, 8),
    bb_middle NUMERIC(20, 8),
    bb_lower NUMERIC(20, 8),
    bb_width NUMERIC(20, 8),

    -- Rendements
    returns NUMERIC(15, 8),
    log_returns NUMERIC(15, 8),

    -- Volatilité
    volatility_24h NUMERIC(15, 8),

    -- Volume features
    volume_sma_7 NUMERIC(30, 8),
    volume_ratio NUMERIC(10, 4),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT features_time_symbol_timeframe_key UNIQUE(time, symbol, timeframe),
    CONSTRAINT fk_features_symbol FOREIGN KEY (symbol) REFERENCES crypto_metadata(symbol) ON DELETE CASCADE
);

-- Convertir en hypertable
SELECT create_hypertable('features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index
CREATE INDEX idx_features_symbol_time ON features (symbol, time DESC);
CREATE INDEX idx_features_timeframe ON features (timeframe, time DESC);

COMMENT ON TABLE features IS 'Features calculées pour les modèles ML';

-- ============================================================================
-- TABLE: predictions
-- Prédictions des modèles
-- ============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_time TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Prédictions
    predicted_price NUMERIC(20, 8) NOT NULL,
    actual_price NUMERIC(20, 8),

    -- Métadonnées de prédiction
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    horizon_hours INTEGER NOT NULL,
    confidence_lower NUMERIC(20, 8),
    confidence_upper NUMERIC(20, 8),

    -- Erreurs (calculées après que le prix réel soit connu)
    absolute_error NUMERIC(20, 8),
    percentage_error NUMERIC(10, 4),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_predictions_symbol FOREIGN KEY (symbol) REFERENCES crypto_metadata(symbol) ON DELETE CASCADE
);

ALTER TABLE predictions DROP CONSTRAINT predictions_pkey;

ALTER TABLE predictions
ADD PRIMARY KEY (symbol, timeframe, target_time, prediction_time);



-- Convertir en hypertable
SELECT create_hypertable('predictions', 'prediction_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index
CREATE INDEX idx_predictions_symbol_target ON predictions (symbol, target_time DESC);
CREATE INDEX idx_predictions_model ON predictions (model_name, prediction_time DESC);
CREATE INDEX idx_predictions_symbol_model ON predictions (symbol, model_name, prediction_time DESC);

COMMENT ON TABLE predictions IS 'Prédictions des modèles ML';
COMMENT ON COLUMN predictions.prediction_time IS 'Moment où la prédiction a été faite';
COMMENT ON COLUMN predictions.target_time IS 'Moment futur pour lequel on prédit';
COMMENT ON COLUMN predictions.horizon_hours IS 'Horizon de prédiction en heures';

-- ============================================================================
-- TABLE: model_performance
-- Métriques de performance des modèles
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    evaluation_date DATE NOT NULL,

    -- Métriques de régression
    rmse NUMERIC(20, 8),
    mae NUMERIC(20, 8),
    mape NUMERIC(10, 4),
    r2_score NUMERIC(10, 6),

    -- Métriques spécifiques trading
    directional_accuracy NUMERIC(10, 4),
    profit_factor NUMERIC(10, 4),

    -- Informations d'entraînement
    train_samples INTEGER,
    test_samples INTEGER,
    training_duration_seconds INTEGER,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_performance_symbol FOREIGN KEY (symbol) REFERENCES crypto_metadata(symbol) ON DELETE CASCADE
);

-- Index
CREATE INDEX idx_model_performance_name ON model_performance (model_name, evaluation_date DESC);
CREATE INDEX idx_model_performance_symbol ON model_performance (symbol, evaluation_date DESC);

COMMENT ON TABLE model_performance IS 'Métriques de performance des modèles ML';

-- ============================================================================
-- TABLE: trading_signals
-- Signaux de trading générés
-- ============================================================================

CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),

    -- Détails du signal
    price NUMERIC(20, 8) NOT NULL,
    confidence NUMERIC(5, 4),
    model_name VARCHAR(50),

    -- Stop loss et take profit
    stop_loss NUMERIC(20, 8),
    take_profit NUMERIC(20, 8),

    -- Statut
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'executed', 'cancelled', 'expired')),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_signals_symbol FOREIGN KEY (symbol) REFERENCES crypto_metadata(symbol) ON DELETE CASCADE
);

-- ALTER TABLE trading_signals DROP CONSTRAINT trading_signals_pkey;

CREATE INDEX IF NOT EXISTS idx_trading_signals_id ON trading_signals(id);



-- Convertir en hypertable
SELECT create_hypertable('trading_signals', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index
CREATE INDEX idx_trading_signals_symbol ON trading_signals (symbol, time DESC);
CREATE INDEX idx_trading_signals_status ON trading_signals (status, time DESC);

COMMENT ON TABLE trading_signals IS 'Signaux de trading générés par les modèles';

-- ============================================================================
-- VUES MATÉRIALISÉES
-- Pour optimiser les requêtes fréquentes
-- ============================================================================

-- Vue pour les derniers prix
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_prices AS
SELECT DISTINCT ON (symbol, timeframe)
    symbol,
    timeframe,
    time,
    close as price,
    volume,
    created_at
FROM ohlcv_data
ORDER BY symbol, timeframe, time DESC;

CREATE UNIQUE INDEX idx_latest_prices ON latest_prices(symbol, timeframe);

-- Vue pour les statistiques quotidiennes
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    timeframe,
    FIRST(open, time) as day_open,
    MAX(high) as day_high,
    MIN(low) as day_low,
    LAST(close, time) as day_close,
    SUM(volume) as day_volume,
    COUNT(*) as candles_count
FROM ohlcv_data
GROUP BY day, symbol, timeframe
ORDER BY day DESC, symbol;

CREATE INDEX idx_daily_stats ON daily_stats(symbol, day DESC);

-- ============================================================================
-- FONCTIONS UTILITAIRES
-- ============================================================================

-- Fonction pour mettre à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger pour crypto_metadata
CREATE TRIGGER update_crypto_metadata_updated_at
    BEFORE UPDATE ON crypto_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger pour trading_signals
CREATE TRIGGER update_trading_signals_updated_at
    BEFORE UPDATE ON trading_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- POLITIQUES DE RÉTENTION (TimescaleDB)
-- ============================================================================

-- Garder les données OHLCV détaillées pendant 1 an
SELECT add_retention_policy('ohlcv_data', INTERVAL '1 year', if_not_exists => TRUE);

-- Garder les prédictions pendant 6 mois
SELECT add_retention_policy('predictions', INTERVAL '6 months', if_not_exists => TRUE);

-- Garder les signaux de trading pendant 3 mois
SELECT add_retention_policy('trading_signals', INTERVAL '3 months', if_not_exists => TRUE);

-- ============================================================================
-- POLITIQUES DE COMPRESSION (TimescaleDB)
-- Compresser les données anciennes pour économiser de l'espace
-- ============================================================================

-- Compresser les données OHLCV après 7 jours
ALTER TABLE ohlcv_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('ohlcv_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Compresser les features après 7 jours
ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('features', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Créer un utilisateur pour l'application (optionnel)
CREATE USER taki WITH PASSWORD 'takiensah00';
GRANT CONNECT ON DATABASE crypto_prediction TO crypto_app;
GRANT USAGE ON SCHEMA public TO crypto_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO crypto_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO crypto_app;

-- ============================================================================
-- DONNÉES INITIALES
-- ============================================================================

-- Insérer quelques cryptos de base
INSERT INTO crypto_metadata (symbol, name, base_currency, quote_currency) VALUES
    ('BTC/USDT', 'Bitcoin', 'BTC', 'USDT'),
    ('ETH/USDT', 'Ethereum', 'ETH', 'USDT'),
    ('BNB/USDT', 'Binance Coin', 'BNB', 'USDT'),
    ('SOL/USDT', 'Solana', 'SOL', 'USDT'),
    ('ADA/USDT', 'Cardano', 'ADA', 'USDT'),
    ('XRP/USDT', 'Ripple', 'XRP', 'USDT'),
    ('DOT/USDT', 'Polkadot', 'DOT', 'USDT'),
    ('AVAX/USDT', 'Avalanche', 'AVAX', 'USDT'),
    ('MATIC/USDT', 'Polygon', 'MATIC', 'USDT'),
    ('LINK/USDT', 'Chainlink', 'LINK', 'USDT')
ON CONFLICT (symbol) DO NOTHING;

-- ============================================================================
-- INFORMATIONS
-- ============================================================================

-- Afficher les informations sur les hypertables créées
SELECT * FROM timescaledb_information.hypertables;

-- Afficher les politiques de compression
SELECT * FROM timescaledb_information.compression_settings;

-- Afficher les politiques de rétention
SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention';

COMMENT ON DATABASE crypto_prediction IS 'Base de données pour le projet de prédiction de prix crypto';