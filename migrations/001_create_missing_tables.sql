-- Missing Tables Migration Script
-- Run this to create missing tables in the intent_engine database

-- Creative Assets table
CREATE TABLE IF NOT EXISTS creative_assets (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL,
    asset_url TEXT NOT NULL,
    dimensions JSONB,
    checksum VARCHAR(64),
    file_size INTEGER,
    mime_type VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Ad Metrics table
CREATE TABLE IF NOT EXISTS ad_metrics (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    intent_goal VARCHAR(50),
    intent_use_case VARCHAR(100),
    impression_count INTEGER DEFAULT 0,
    click_count INTEGER DEFAULT 0,
    conversion_count INTEGER DEFAULT 0,
    ctr DECIMAL(10,6),
    cpc DECIMAL(10,4),
    roas DECIMAL(10,4),
    engagement_rate DECIMAL(10,6),
    spend DECIMAL(12,4),
    revenue DECIMAL(12,4),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ad_id, date, intent_goal, intent_use_case)
);

-- Click Tracking table
CREATE TABLE IF NOT EXISTS click_tracking (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    user_agent TEXT,
    ip_address VARCHAR(45),
    referrer TEXT,
    landing_page TEXT,
    device_type VARCHAR(50),
    browser VARCHAR(50),
    os VARCHAR(50),
    country VARCHAR(2),
    region VARCHAR(100),
    city VARCHAR(100),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Conversion Tracking table
CREATE TABLE IF NOT EXISTS conversion_tracking (
    id SERIAL PRIMARY KEY,
    click_id INTEGER NOT NULL REFERENCES click_tracking(id) ON DELETE CASCADE,
    conversion_type VARCHAR(50) NOT NULL,
    conversion_value DECIMAL(12,4),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB,
    revenue DECIMAL(12,4),
    quantity INTEGER,
    product_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fraud Detection table
CREATE TABLE IF NOT EXISTS fraud_detection (
    id SERIAL PRIMARY KEY,
    ad_id INTEGER NOT NULL REFERENCES ads(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    fraud_score DECIMAL(5,4),
    is_fraudulent BOOLEAN DEFAULT FALSE,
    severity VARCHAR(20),
    reason TEXT,
    metadata JSONB,
    review_status VARCHAR(20) DEFAULT 'pending',
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- A/B Tests table
CREATE TABLE IF NOT EXISTS ab_tests (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'draft',
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- A/B Test Variants table
CREATE TABLE IF NOT EXISTS ab_test_variants (
    id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES ab_tests(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    weight DECIMAL(5,4) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(test_id, name)
);

-- A/B Test Assignments table
CREATE TABLE IF NOT EXISTS ab_test_assignments (
    id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES ab_tests(id) ON DELETE CASCADE,
    user_id VARCHAR(100) NOT NULL,
    variant_id INTEGER NOT NULL REFERENCES ab_test_variants(id),
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    converted BOOLEAN DEFAULT FALSE,
    conversion_value DECIMAL(12,4),
    UNIQUE(test_id, user_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_creative_assets_ad_id ON creative_assets(ad_id);
CREATE INDEX IF NOT EXISTS idx_ad_metrics_ad_id ON ad_metrics(ad_id);
CREATE INDEX IF NOT EXISTS idx_ad_metrics_date ON ad_metrics(date);
CREATE INDEX IF NOT EXISTS idx_click_tracking_ad_id ON click_tracking(ad_id);
CREATE INDEX IF NOT EXISTS idx_click_tracking_session_id ON click_tracking(session_id);
CREATE INDEX IF NOT EXISTS idx_conversion_tracking_click_id ON conversion_tracking(click_id);
CREATE INDEX IF NOT EXISTS idx_fraud_detection_ad_id ON fraud_detection(ad_id);
CREATE INDEX IF NOT EXISTS idx_fraud_detection_event_type ON fraud_detection(event_type);
CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests(status);
CREATE INDEX IF NOT EXISTS idx_ab_test_variants_test_id ON ab_test_variants(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_assignments_test_id ON ab_test_assignments(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_assignments_user_id ON ab_test_assignments(user_id);

-- Add missing indexes to existing tables
CREATE INDEX IF NOT EXISTS idx_ads_approval_status ON ads(approval_status, status);
CREATE INDEX IF NOT EXISTS idx_campaigns_advertiser_id ON campaigns(advertiser_id);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_ad_groups_campaign_id ON ad_groups(campaign_id);
