-- Crawler tables for intent_engine database
-- Crawled pages table
CREATE TABLE IF NOT EXISTS crawled_pages (
    id VARCHAR(255) PRIMARY KEY,
    url VARCHAR(2048) NOT NULL,
    final_url VARCHAR(2048),
    title VARCHAR(1024),
    content TEXT,
    meta_description TEXT,
    meta_keywords TEXT,
    status_code INTEGER,
    content_type VARCHAR(255),
    content_length INTEGER,
    load_time_ms DOUBLE PRECISION,
    crawl_depth INTEGER DEFAULT 0,
    outbound_links INTEGER DEFAULT 0,
    inbound_links INTEGER DEFAULT 0,
    pagerank DOUBLE PRECISION DEFAULT 0.0,
    language VARCHAR(10) DEFAULT 'en',
    is_indexed BOOLEAN DEFAULT FALSE,
    crawled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    next_crawl_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_is_indexed ON crawled_pages(is_indexed);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_pagerank ON crawled_pages(pagerank);

-- Page links table
CREATE TABLE IF NOT EXISTS page_links (
    id SERIAL PRIMARY KEY,
    source_page_id VARCHAR(255) NOT NULL,
    target_url VARCHAR(2048) NOT NULL,
    anchor_text VARCHAR(512),
    link_type VARCHAR(50) DEFAULT 'dofollow',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (source_page_id) REFERENCES crawled_pages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_page_links_source ON page_links(source_page_id);
CREATE INDEX IF NOT EXISTS idx_page_links_target ON page_links(target_url);

-- Crawl stats table
CREATE TABLE IF NOT EXISTS crawl_stats (
    id SERIAL PRIMARY KEY,
    stat_date DATE DEFAULT CURRENT_DATE,
    pages_crawled INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    pages_indexed INTEGER DEFAULT 0,
    links_extracted INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO crawl_stats (stat_date, pages_crawled, pages_failed, pages_indexed, links_extracted)
VALUES (CURRENT_DATE, 0, 0, 0, 0)
ON CONFLICT (stat_date) DO NOTHING;
