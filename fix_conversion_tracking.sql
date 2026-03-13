-- Add missing payload column to conversion_tracking
ALTER TABLE conversion_tracking ADD COLUMN IF NOT EXISTS payload JSONB;
