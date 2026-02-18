"""
Intent Engine - Comprehensive Load Testing Suite

This module provides load testing using Locust for:
- Intent extraction endpoint
- Ranking endpoint
- URL ranking endpoint
- Ad matching endpoint
- Campaign/ad group management endpoints

Usage:
    locust -f load_testing/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 in your browser
"""

import random
import threading

from locust import HttpUser, between, events, task

# FIX: Add locks for shared state in load tests
_query_history_lock = threading.Lock()


# Diverse test queries - NOT just email!
TEST_QUERIES = {
    "technology": [
        "best laptop for programming 2024",
        "how to setup kubernetes cluster",
        "compare python vs javascript performance",
        "docker compose tutorial for beginners",
        "secure coding practices checklist",
        "best IDE for web development",
        "how to optimize database queries",
        "machine learning frameworks comparison",
        "blockchain vs traditional database",
        "CI/CD pipeline best practices",
    ],
    "shopping": [
        "best wireless headphones under 100",
        "compare iPhone vs Samsung Galaxy",
        "where to buy sustainable clothing",
        "mechanical keyboard reviews 2024",
        "smart home devices privacy concerns",
        "best budget smartphone camera",
        "electric vehicle charging stations near me",
        "compare running shoes for marathon",
        "standing desk recommendations",
        "best noise cancelling headphones",
    ],
    "health": [
        "how to improve sleep quality naturally",
        "meditation apps privacy comparison",
        "home workout routines without equipment",
        "nutrition tracking apps without data sharing",
        "yoga for beginners tutorial",
        "mental health resources anonymous",
        "compare fitness trackers privacy",
        "healthy meal prep ideas",
        "stress management techniques",
        "ergonomic office setup guide",
    ],
    "privacy_security": [
        "best password manager comparison",
        "how to browse anonymously",
        "VPN services privacy review",
        "secure messaging apps comparison",
        "privacy focused browser alternatives",
        "two factor authentication setup",
        "how to encrypt files locally",
        "privacy settings for social media",
        "secure cloud storage options",
        "digital footprint cleanup guide",
    ],
    "finance": [
        "best budgeting apps without tracking",
        "how to invest in index funds",
        "cryptocurrency wallet security",
        "compare credit card rewards",
        "personal finance management tools",
        "tax preparation software privacy",
        "how to save money on groceries",
        "retirement planning calculator",
        "student loan refinancing options",
        "emergency fund savings strategy",
    ],
    "travel": [
        "best travel booking sites privacy",
        "how to find cheap flights",
        "solo travel safety tips",
        "compare travel insurance options",
        "digital nomad destinations 2024",
        "offline maps for travel",
        "travel photography tips",
        "how to pack light for travel",
        "local transportation apps",
        "travel rewards programs comparison",
    ],
    "learning": [
        "best online courses for programming",
        "how to learn data science",
        "language learning apps comparison",
        "free educational resources online",
        "how to study effectively",
        "online degree vs bootcamp",
        "skill building for career change",
        "learn guitar online tutorial",
        "public speaking courses online",
        "time management strategies",
    ],
    "productivity": [
        "best task management apps privacy",
        "how to organize digital files",
        "note taking apps comparison",
        "calendar apps without google",
        "focus techniques for deep work",
        "email management strategies",
        "project management tools comparison",
        "how to reduce digital clutter",
        "automation tools for productivity",
        "meeting scheduling without tracking",
    ],
}

# Test URLs for diverse domains
TEST_URLS = {
    "technology": [
        "https://github.com",
        "https://stackoverflow.com",
        "https://docs.python.org",
        "https://kubernetes.io",
        "https://docker.com",
        "https://vuejs.org",
        "https://react.dev",
        "https://developer.mozilla.org",
    ],
    "shopping": [
        "https://amazon.com",
        "https://ebay.com",
        "https://etsy.com",
        "https://newegg.com",
        "https://bestbuy.com",
        "https://bhphotovideo.com",
    ],
    "privacy": [
        "https://protonmail.com",
        "https://duckduckgo.com",
        "https://signal.org",
        "https://privacytools.io",
        "https://eff.org",
        "https://wikipedia.org",
        "https://mozilla.org",
    ],
    "news": [
        "https://techcrunch.com",
        "https://arstechnica.com",
        "https://wired.com",
        "https://theverge.com",
        "https://hackernews.com",
    ],
    "education": [
        "https://coursera.org",
        "https://edx.org",
        "https://khanacademy.org",
        "https://udemy.com",
        "https://freecodecamp.org",
    ],
}


class IntentEngineLoadTest(HttpUser):
    """Simulates users interacting with the Intent Engine"""

    # Wait between 1-5 seconds between requests (realistic user behavior)
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a user starts"""
        self.session_id = f"locust-{random.randint(10000, 99999)}"
        self.query_history = []

    def get_random_query(self, category=None):
        """Get a random query from specified category or any category"""
        if category and category in TEST_QUERIES:
            return random.choice(TEST_QUERIES[category])
        # Pick random category
        category = random.choice(list(TEST_QUERIES.keys()))
        return random.choice(TEST_QUERIES[category])

    def get_random_urls(self, count=5):
        """Get random URLs from various categories"""
        all_urls = []
        for urls in TEST_URLS.values():
            all_urls.extend(urls)
        return random.sample(all_urls, min(count, len(all_urls)))

    @task(40)  # 40% of requests
    def extract_intent(self):
        """Test intent extraction with diverse queries"""
        query = self.get_random_query()
        # FIX: Use lock for shared state mutation
        with _query_history_lock:
            self.query_history.append(query)

        payload = {
            "product": "search",
            "input": {"text": query},
            "context": {
                "sessionId": self.session_id,
                "userLocale": random.choice(["en-US", "en-GB", "de-DE", "fr-FR"]),
            },
        }

        with self.client.post("/extract-intent", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "intent" in data:
                    response.success()
                else:
                    response.failure("Missing intent in response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(30)  # 30% of requests
    def rank_urls(self):
        """Test URL ranking with diverse queries and URLs"""
        query = self.get_random_query()
        urls = self.get_random_urls(random.randint(3, 10))

        payload = {
            "query": query,
            "urls": urls,
            "options": {
                "exclude_big_tech": random.choice([True, False]),
                "min_privacy_score": random.choice([0.0, 0.3, 0.5, 0.7]),
                "weights": {"relevance": 0.40, "privacy": 0.30, "quality": 0.20, "ethics": 0.10},
            },
        }

        with self.client.post("/rank-urls", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "ranked_urls" in data and len(data["ranked_urls"]) > 0:
                    response.success()
                else:
                    response.failure("No ranked URLs in response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(15)  # 15% of requests
    def rank_results(self):
        """Test result ranking"""
        # First extract intent
        query = self.get_random_query("technology")
        intent_response = self.client.post(
            "/extract-intent",
            json={
                "product": "search",
                "input": {"text": query},
                "context": {"sessionId": self.session_id, "userLocale": "en-US"},
            },
        )

        if intent_response.status_code != 200:
            return

        intent = intent_response.json()["intent"]

        # Create test candidates
        candidates = []
        for i in range(random.randint(3, 8)):
            candidates.append(
                {
                    "id": f"result-{i}",
                    "title": f"Result {i} for {query[:20]}",
                    "description": f"Description for result {i} with keywords from query",
                    "platform": random.choice(["web", "mobile", "desktop"]),
                    "qualityScore": random.uniform(0.5, 1.0),
                    "tags": query.split()[:3],
                }
            )

        payload = {"intent": intent, "candidates": candidates}

        with self.client.post("/rank-results", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)  # 10% of requests
    def match_ads(self):
        """Test ad matching"""
        query = self.get_random_query("shopping")

        intent_response = self.client.post(
            "/extract-intent",
            json={
                "product": "search",
                "input": {"text": query},
                "context": {"sessionId": self.session_id, "userLocale": "en-US"},
            },
        )

        if intent_response.status_code != 200:
            return

        intent = intent_response.json()["intent"]

        # Create test ads
        ads = []
        for i in range(random.randint(2, 5)):
            ads.append(
                {
                    "id": f"ad-{i}",
                    "title": f"Ad {i} for {query[:20]}",
                    "description": "Great product related to your search",
                    "targetingConstraints": {},
                    "forbiddenDimensions": [],
                    "qualityScore": random.uniform(0.6, 0.95),
                    "ethicalTags": ["privacy", "quality"],
                    "advertiser": f"advertiser-{i}",
                }
            )

        payload = {"intent": intent, "ad_inventory": ads, "config": {"minThreshold": 0.3, "topK": 3}}

        with self.client.post("/match-ads", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)  # 5% of requests
    def recommend_services(self):
        """Test service recommendation"""
        query = self.get_random_query()

        intent_response = self.client.post(
            "/extract-intent",
            json={
                "product": "search",
                "input": {"text": query},
                "context": {"sessionId": self.session_id, "userLocale": "en-US"},
            },
        )

        if intent_response.status_code != 200:
            return

        intent = intent_response.json()["intent"]

        payload = {
            "intent": intent,
            "available_services": [
                {"name": "Search", "capabilities": ["search", "ranking", "privacy"], "ethical_alignment": ["privacy"]},
                {"name": "Docs", "capabilities": ["documentation", "tutorials"], "ethical_alignment": ["openness"]},
                {
                    "name": "Help",
                    "capabilities": ["support", "troubleshooting"],
                    "ethical_alignment": ["accessibility"],
                },
            ],
        }

        with self.client.post("/recommend-services", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class CampaignManagementLoadTest(HttpUser):
    """Simulates advertiser campaign management"""

    wait_time = between(3, 10)  # Slower pace for management tasks
    weight = 1  # 1:10 ratio with main user class

    def on_start(self):
        """Login as advertiser"""
        self.advertiser_id = random.randint(1, 100)

    @task(50)
    def list_campaigns(self):
        """List campaigns"""
        with self.client.get("/campaigns", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(30)
    def create_campaign(self):
        """Create a new campaign"""
        payload = {
            "advertiser_id": self.advertiser_id,
            "name": f"Test Campaign {random.randint(1000, 9999)}",
            "start_date": "2026-02-16T00:00:00",
            "end_date": "2026-03-16T23:59:59",
            "budget": random.uniform(1000, 10000),
            "daily_budget": random.uniform(50, 200),
            "status": "active",
        }

        with self.client.post("/campaigns", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(20)
    def get_campaign_performance(self):
        """Get campaign performance report"""
        with self.client.get("/reports/campaign-performance", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Custom events for detailed metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Log slow requests"""
    if response_time > 500:  # Log requests taking more than 500ms
        print(f"⚠️ SLOW REQUEST: {name} took {response_time:.2f}ms")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print summary when tests complete"""
    print("\n" + "=" * 60)
    print("LOAD TEST COMPLETED")
    print("=" * 60)
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Failed requests: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
    print("=" * 60)


if __name__ == "__main__":
    # Allow running with: python locustfile.py
    # This is just for testing the file structure
    print("Load testing configuration loaded successfully!")
    print(f"Total test queries: {sum(len(v) for v in TEST_QUERIES.values())}")
    print(f"Query categories: {list(TEST_QUERIES.keys())}")
    print("\nTo run load tests:")
    print("  locust -f load_testing/locustfile.py --host=http://localhost:8000")
