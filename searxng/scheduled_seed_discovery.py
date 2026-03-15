"""
Intent Engine - Scheduled Seed Discovery

Automatically discovers and injects new seed URLs on a schedule
to continuously expand the crawler's coverage.
"""

import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from searxng.seed_injector import inject_discovered_urls, SeedURLInjector
from searxng.topic_expander import get_topic_expander

logger = logging.getLogger(__name__)


class ScheduledSeedDiscovery:
    """
    Manages scheduled seed URL discovery and injection.
    
    Features:
    - Periodic discovery (default: daily)
    - Topic expansion based on trending queries
    - Automatic category discovery
    - Rate limiting (avoid overwhelming crawler)
    - Statistics tracking
    """
    
    def __init__(
        self,
        discovery_interval_hours: int = 24,
        topic_expansion_interval_hours: int = 6,
        max_urls_per_run: int = 50
    ):
        self.discovery_interval_hours = discovery_interval_hours
        self.topic_expansion_interval_hours = topic_expansion_interval_hours
        self.max_urls_per_run = max_urls_per_run
        self.scheduler = AsyncIOScheduler()
        self.run_count = 0
        self.total_urls_discovered = 0
        self.topic_expander = None
        
    async def initialize(self):
        """Initialize topic expander."""
        self.topic_expander = get_topic_expander()
        await self.topic_expander.initialize_topics()
        logger.info("Topic expander initialized")
        
    def start(self):
        """Start the scheduled discovery."""
        # Initialize in background
        asyncio.create_task(self.initialize())
        
        # Schedule URL discovery job (daily at 3 AM)
        self.scheduler.add_job(
            self._run_discovery,
            trigger=CronTrigger(
                hour=3,  # Run at 3 AM (low traffic time)
                minute=0
            ),
            id='seed_discovery',
            name='Seed URL Discovery',
            replace_existing=True
        )
        
        # Schedule topic expansion job (every 6 hours)
        self.scheduler.add_job(
            self._run_topic_expansion,
            trigger=IntervalTrigger(
                hours=self.topic_expansion_interval_hours
            ),
            id='topic_expansion',
            name='Topic Expansion',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info(
            f"Started scheduled seed discovery "
            f"(URL discovery: daily at 3 AM, Topic expansion: every {self.topic_expansion_interval_hours}h)"
        )
    
    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("Stopped scheduled seed discovery")
    
    async def _run_discovery(self):
        """Run a discovery cycle."""
        self.run_count += 1
        logger.info(f"Starting seed discovery run #{self.run_count}")
        
        try:
            # Get expanded topics
            if self.topic_expander:
                expanded_topics = await self.topic_expander.get_expanded_topics(
                    limit_per_category=15
                )
                logger.info(
                    f"Using {sum(len(t) for t in expanded_topics.values())} topics "
                    f"from {len(expanded_topics)} categories"
                )
            
            results = await inject_discovered_urls()
            
            urls_found = (
                results.get('go_urls_discovered', 0) +
                results.get('prog_urls_discovered', 0)
            )
            
            self.total_urls_discovered += urls_found
            
            logger.info(
                f"Seed discovery run #{self.run_count} complete: "
                f"{urls_found} URLs discovered, "
                f"queue size: {results.get('queue_size', 0)}, "
                f"total discovered: {self.total_urls_discovered}"
            )
            
        except Exception as e:
            logger.error(f"Seed discovery run #{self.run_count} failed: {e}")
    
    async def _run_topic_expansion(self):
        """Run topic expansion cycle."""
        try:
            if not self.topic_expander:
                return
            
            # Get trending keywords from query history
            stats = await self.topic_expander.get_stats()
            
            logger.info(
                f"Topic expansion: {stats['total_categories']} categories, "
                f"{stats['total_topics']} topics, "
                f"{stats['queries_analyzed']} queries analyzed"
            )
            
        except Exception as e:
            logger.error(f"Topic expansion failed: {e}")
    
    def get_stats(self) -> dict:
        """Get discovery statistics."""
        stats = {
            "run_count": self.run_count,
            "total_urls_discovered": self.total_urls_discovered,
            "discovery_interval_hours": self.discovery_interval_hours,
            "topic_expansion_interval_hours": self.topic_expansion_interval_hours,
            "max_urls_per_run": self.max_urls_per_run,
            "scheduler_running": self.scheduler.running
        }
        
        if self.topic_expander:
            # Run async stats in background
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't block in running loop, return basic stats
                    stats["topic_expander"] = "initialized"
                else:
                    stats["topic_expander"] = loop.run_until_complete(
                        self.topic_expander.get_stats()
                    )
            except:
                stats["topic_expander"] = "initializing"
        
        return stats


# Global instance
_seed_discovery: ScheduledSeedDiscovery = None


def get_seed_discovery() -> ScheduledSeedDiscovery:
    """Get or create the scheduled seed discovery instance."""
    global _seed_discovery
    if _seed_discovery is None:
        _seed_discovery = ScheduledSeedDiscovery(
            discovery_interval_hours=24,
            topic_expansion_interval_hours=6,
            max_urls_per_run=50
        )
    return _seed_discovery


async def run_immediate_discovery():
    """Run seed discovery immediately (for testing/manual trigger)."""
    logger.info("Running immediate seed discovery...")
    return await inject_discovered_urls()


if __name__ == "__main__":
    # For testing: run immediate discovery
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        results = await run_immediate_discovery()
        print(f"Discovery results: {results}")
    
    asyncio.run(main())
