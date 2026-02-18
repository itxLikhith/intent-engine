-- Fix conversion_tracking and fraud_detection tables to match ORM models

BEGIN;

-- Drop and recreate conversion_tracking
DROP TABLE IF EXISTS conversion_tracking CASCADE;

CREATE TABLE conversion_tracking (
    id SERIAL PRIMARY KEY,
    click_id INTEGER NOT NULL REFERENCES click_tracking(id) ON DELETE CASCADE,
    conversion_type VARCHAR(100),
    value DECIMAL(12,4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversion_tracking_click_id ON conversion_tracking(click_id);
CREATE INDEX idx_conversion_tracking_type_status ON conversion_tracking(conversion_type, status);

-- Drop and recreate fraud_detection
DROP TABLE IF EXISTS fraud_detection CASCADE;

CREATE TABLE fraud_detection (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER REFERENCES ads(id) ON DELETE CASCADE,
    event_type VARCHAR(50),
    event_id INTEGER,
    reason TEXT,
    severity VARCHAR(20),
    review_status VARCHAR(20) DEFAULT 'pending',
    fraud_score DECIMAL(5,4),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_fraud_detection_ad_id ON fraud_detection(ad_id);
CREATE INDEX idx_fraud_detection_event_type ON fraud_detection(event_type);
CREATE INDEX idx_fraud_detection_review_status ON fraud_detection(review_status);

COMMIT;
