-- Fix click_tracking table to match ORM model
-- Drop and recreate with correct columns

BEGIN;

-- Drop existing table (will lose data, but this is for testing)
DROP TABLE IF EXISTS click_tracking CASCADE;

-- Recreate with correct schema matching ORM
CREATE TABLE click_tracking (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
    session_id VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_hash VARCHAR(100),
    user_agent_hash TEXT,
    referring_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_click_tracking_ad_id ON click_tracking(ad_id);
CREATE INDEX idx_click_tracking_session_id ON click_tracking(session_id);
CREATE INDEX idx_click_tracking_timestamp ON click_tracking(timestamp);
CREATE INDEX idx_click_tracking_ad_id_timestamp ON click_tracking(ad_id, timestamp);

COMMIT;
