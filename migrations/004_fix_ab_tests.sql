-- Fix ab_tests table to match ORM model

BEGIN;

-- Drop dependent tables first
DROP TABLE IF EXISTS ab_test_assignments CASCADE;
DROP TABLE IF EXISTS ab_test_variants CASCADE;

-- Add missing columns to ab_tests
ALTER TABLE ab_tests 
    ADD COLUMN IF NOT EXISTS campaign_id INTEGER NOT NULL,
    ADD COLUMN IF NOT EXISTS traffic_allocation DECIMAL(5,4) DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS min_sample_size INTEGER DEFAULT 1000,
    ADD COLUMN IF NOT EXISTS confidence_level DECIMAL(5,4) DEFAULT 0.95,
    ADD COLUMN IF NOT EXISTS primary_metric VARCHAR(50) DEFAULT 'ctr';

-- Add foreign key constraint
ALTER TABLE ab_tests 
    ADD CONSTRAINT ab_tests_campaign_id_fkey 
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;

-- Create index
CREATE INDEX IF NOT EXISTS idx_ab_tests_campaign_id ON ab_tests(campaign_id);

-- Recreate ab_test_variants
CREATE TABLE IF NOT EXISTS ab_test_variants (
    id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES ab_tests(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    weight DECIMAL(5,4) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(test_id, name)
);

-- Recreate ab_test_assignments
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

COMMIT;
