-- Crawler Database Schema
-- This creates the necessary tables for storing crawled pages and links

-- Connect to crawler database
\c crawler;

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

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_title ON crawled_pages(title);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_crawled_at ON crawled_pages(crawled_at);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_is_indexed ON crawled_pages(is_indexed);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_pagerank ON crawled_pages(pagerank);

-- Page links table (for PageRank calculation)
CREATE TABLE IF NOT EXISTS page_links (
    id SERIAL PRIMARY KEY,
    source_page_id VARCHAR(255) NOT NULL,
    target_url VARCHAR(2048) NOT NULL,
    anchor_text VARCHAR(512),
    link_type VARCHAR(50) DEFAULT 'dofollow',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (source_page_id) REFERENCES crawled_pages(id) ON DELETE CASCADE
);

-- Create indexes for link analysis
CREATE INDEX IF NOT EXISTS idx_page_links_source ON page_links(source_page_id);
CREATE INDEX IF NOT EXISTS idx_page_links_target ON page_links(target_url);
CREATE UNIQUE INDEX IF NOT EXISTS idx_page_links_unique ON page_links(source_page_id, target_url);

-- Crawl queue table
CREATE TABLE IF NOT EXISTS crawl_queue (
    id SERIAL PRIMARY KEY,
    url VARCHAR(2048) NOT NULL UNIQUE,
    priority INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error TEXT
);

-- Create indexes for queue management
CREATE INDEX IF NOT EXISTS idx_crawl_queue_status ON crawl_queue(status);
CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority ON crawl_queue(priority, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_crawl_queue_scheduled ON crawl_queue(scheduled_at);

-- Search index table (for storing indexed content)
CREATE TABLE IF NOT EXISTS search_index (
    id SERIAL PRIMARY KEY,
    page_id VARCHAR(255) NOT NULL UNIQUE,
    url VARCHAR(2048) NOT NULL,
    title VARCHAR(1024) NOT NULL,
    content TEXT NOT NULL,
    meta_description TEXT,
    term_frequencies JSONB,
    word_count INTEGER DEFAULT 0,
    document_length INTEGER DEFAULT 0,
    pagerank_score DOUBLE PRECISION DEFAULT 0.0,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (page_id) REFERENCES crawled_pages(id) ON DELETE CASCADE
);

-- Create indexes for full-text search
CREATE INDEX IF NOT EXISTS idx_search_index_url ON search_index(url);
CREATE INDEX IF NOT EXISTS idx_search_index_title ON search_index(title);
CREATE INDEX IF NOT EXISTS idx_search_index_pagerank ON search_index(pagerank_score);
CREATE INDEX IF NOT EXISTS idx_search_index_content ON search_index USING gin(to_tsvector('english', content));

-- Crawl statistics table
CREATE TABLE IF NOT EXISTS crawl_stats (
    id SERIAL PRIMARY KEY,
    stat_date DATE DEFAULT CURRENT_DATE,
    pages_crawled INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    pages_indexed INTEGER DEFAULT 0,
    links_extracted INTEGER DEFAULT 0,
    avg_load_time_ms DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create unique index for daily stats
CREATE UNIQUE INDEX IF NOT EXISTS idx_crawl_stats_date ON crawl_stats(stat_date);

-- Insert initial stats record
INSERT INTO crawl_stats (stat_date, pages_crawled, pages_failed, pages_indexed, links_extracted)
VALUES (CURRENT_DATE, 0, 0, 0, 0)
ON CONFLICT (stat_date) DO NOTHING;

-- Log schema creation
DO $$
BEGIN
    RAISE NOTICE 'Crawler database schema created successfully!';
    RAISE NOTICE 'Tables created: crawled_pages, page_links, crawl_queue, search_index, crawl_stats';
END $$;
