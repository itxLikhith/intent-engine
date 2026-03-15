-- Intent Engine Crawler Database Initialization Script
-- Run this after PostgreSQL starts to create all required tables

-- Create crawled_pages table
CREATE TABLE IF NOT EXISTS crawled_pages (
    id VARCHAR PRIMARY KEY,
    url VARCHAR NOT NULL,
    final_url VARCHAR,
    title VARCHAR,
    content TEXT,
    meta_description VARCHAR,
    meta_keywords VARCHAR,
    status_code INTEGER,
    content_type VARCHAR,
    content_length INTEGER,
    load_time_ms DOUBLE PRECISION,
    crawl_depth INTEGER DEFAULT 0,
    outbound_links INTEGER DEFAULT 0,
    inbound_links INTEGER DEFAULT 0,
    pagerank DOUBLE PRECISION DEFAULT 0,
    language VARCHAR DEFAULT 'en',
    is_indexed BOOLEAN DEFAULT false,
    crawled_at TIMESTAMP,
    updated_at TIMESTAMP,
    next_crawl_at TIMESTAMP
);

-- Create page_links table
CREATE TABLE IF NOT EXISTS page_links (
    id SERIAL PRIMARY KEY,
    source_page_id VARCHAR REFERENCES crawled_pages(id) ON DELETE CASCADE,
    target_url VARCHAR NOT NULL,
    anchor_text TEXT,
    link_type VARCHAR DEFAULT 'dofollow',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (source_page_id, target_url)
);

-- Create crawl_stats table
CREATE TABLE IF NOT EXISTS crawl_stats (
    id SERIAL PRIMARY KEY,
    stat_date DATE UNIQUE DEFAULT CURRENT_DATE,
    pages_crawled BIGINT DEFAULT 0,
    pages_failed BIGINT DEFAULT 0,
    pages_indexed BIGINT DEFAULT 0,
    links_extracted BIGINT DEFAULT 0,
    last_crawl_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_is_indexed ON crawled_pages(is_indexed);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_crawled_at ON crawled_pages(crawled_at);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_next_crawl_at ON crawled_pages(next_crawl_at);
CREATE INDEX IF NOT EXISTS idx_page_links_source ON page_links(source_page_id);
CREATE INDEX IF NOT EXISTS idx_page_links_target ON page_links(target_url);

-- Insert initial stats row if not exists
INSERT INTO crawl_stats (pages_crawled, pages_failed, pages_indexed, links_extracted)
VALUES (0, 0, 0, 0)
ON CONFLICT (stat_date) DO NOTHING;
