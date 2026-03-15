#!/usr/bin/env python
"""Vector Indexer Service - indexes crawled pages into Qdrant"""

import os
import sys
import time
import signal
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting vector indexer service...")
    
    # Wait for dependencies
    logger.info("Waiting for Qdrant to be ready...")
    time.sleep(10)
    
    # Import after delay to ensure dependencies are ready
    from vector_indexer import VectorIndexer
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    
    # Get configuration
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
    postgres_dsn = os.getenv('POSTGRES_DSN', '')
    
    logger.info(f"Qdrant: {qdrant_host}:{qdrant_port}")
    logger.info(f"PostgreSQL: {postgres_dsn}")
    
    # Initialize indexer
    indexer = VectorIndexer(qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    
    if not indexer.initialize():
        logger.error("Failed to initialize vector indexer")
        sys.exit(1)
    
    logger.info("Vector indexer initialized successfully")
    
    # Connect to database
    if not postgres_dsn:
        logger.warning("No PostgreSQL DSN provided, skipping database indexing")
        # Keep running for manual triggers
        signal.pause()
        return
    
    try:
        # Create engine
        db_engine = create_engine(postgres_dsn)
        logger.info("Connected to PostgreSQL")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        # Keep running anyway
        signal.pause()
        return
    
    # Index in batches
    batch_size = 50
    max_pages = 500  # Limit to avoid overwhelming Qdrant
    indexed_count = 0
    
    with Session(db_engine) as session:
        try:
            # Get crawled pages with content
            logger.info(f"Fetching crawled pages (max: {max_pages})...")
            
            stmt = text("""
                SELECT id, url, title, content, meta_description, crawled_at 
                FROM crawled_pages 
                WHERE content IS NOT NULL AND content != ''
                ORDER BY crawled_at DESC
                LIMIT :limit
            """)
            
            result = session.execute(stmt, {"limit": max_pages})
            pages = result.fetchall()
            
            logger.info(f"Found {len(pages)} pages to index")
            
            if not pages:
                logger.info("No pages to index")
                # Keep running
                signal.pause()
                return
            
            # Process in batches
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i+batch_size]
                
                documents = []
                for page in batch:
                    # Extract intent from content (simplified)
                    title = page[2] or ''
                    content = page[3] or ''
                    text_content = f"{title} {content[:2000]}"
                    
                    # Determine intent goal from content
                    intent_goal = 'learn'  # default
                    if any(word in text_content.lower() for word in ['fix', 'error', 'problem', 'issue', 'troubleshoot']):
                        intent_goal = 'troubleshooting'
                    elif any(word in text_content.lower() for word in ['compare', 'vs', 'versus', 'better', 'best']):
                        intent_goal = 'comparison'
                    elif any(word in text_content.lower() for word in ['buy', 'price', 'cost', 'purchase']):
                        intent_goal = 'purchase'
                    
                    # Extract topics (simple keyword extraction)
                    topics = []
                    for keyword in ['tutorial', 'guide', 'example', 'function', 'package', 'api', 'web', 'database']:
                        if keyword in text_content.lower():
                            topics.append(keyword)
                    
                    documents.append({
                        'id': str(page[0]),
                        'url': page[1],
                        'title': title,
                        'content': content[:4000],
                        'intent_goal': intent_goal,
                        'topics': topics[:10],
                        'skill_level': 'intermediate',
                        'crawled_at': str(page[5]) if page[5] else ''
                    })
                
                # Index batch
                if documents:
                    indexed = indexer.index_documents(documents)
                    indexed_count += indexed
                    logger.info(f"Indexed batch {i//batch_size + 1}: {indexed} documents")
            
            logger.info(f"✅ Indexing complete! Total indexed: {indexed_count}/{len(pages)}")
            
            # Get stats
            stats = indexer.get_stats()
            logger.info(f"Qdrant stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Vector indexer service ready - keeping alive for queries")
    
    # Keep running for queries
    signal.pause()

if __name__ == "__main__":
    main()
