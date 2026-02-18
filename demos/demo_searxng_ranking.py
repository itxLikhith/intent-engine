"""
SearXNG + Intent Engine URL Ranking Demo

This script demonstrates the full pipeline:
1. Query SearXNG (privacy-focused metasearch engine) to get search results
2. Extract URLs from SearXNG results
3. Send URLs to our /rank-urls API endpoint for privacy-aware ranking
4. Display the re-ranked results with privacy scores

Usage:
    python demos/demo_searxng_ranking.py "your search query"
    python demos/demo_searxng_ranking.py "best privacy focused email" --exclude-big-tech
    python demos/demo_searxng_ranking.py "open source chat apps" --min-privacy 0.5 --num-results 25
"""

import argparse
import io
import sys
from typing import Any

import requests

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


SEARXNG_URL = "http://localhost:8080"
RANKING_API_URL = "http://localhost:8000"


def query_searxng(
    query: str,
    num_results: int = 25,
    categories: str = "general",
    searxng_url: str = SEARXNG_URL,
) -> list[dict[str, Any]]:
    """Query SearXNG and return search results."""
    params = {
        "q": query,
        "format": "json",
        "categories": categories,
    }

    print(f'\n🔍 Querying SearXNG: "{query}"')
    print(f"   URL: {searxng_url}/search?q={query}&format=json")

    try:
        resp = requests.get(f"{searxng_url}/search", params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to SearXNG at {searxng_url}")
        print("   Make sure SearXNG is running:")
        print('   cd "intent engine" && docker compose -f docker-compose.searxng.yml up -d')
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ SearXNG returned an error: {e}")
        sys.exit(1)

    data = resp.json()
    results = data.get("results", [])

    if not results:
        print("   ⚠️  No results returned from SearXNG")
        return []

    # Deduplicate by URL and limit
    seen_urls = set()
    unique_results = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
        if len(unique_results) >= num_results:
            break

    print(f"   ✅ Got {len(unique_results)} unique results from SearXNG")
    return unique_results


def rank_urls(
    query: str,
    urls: list[str],
    exclude_big_tech: bool = False,
    min_privacy_score: float = 0.0,
    custom_weights: dict[str, float] | None = None,
    ranking_api_url: str = RANKING_API_URL,
) -> dict[str, Any]:
    """Send URLs to our /rank-urls API endpoint for privacy-aware ranking."""
    options: dict[str, Any] = {}
    if exclude_big_tech:
        options["exclude_big_tech"] = True
    if min_privacy_score > 0:
        options["min_privacy_score"] = min_privacy_score
    if custom_weights:
        options["weights"] = custom_weights

    payload = {
        "query": query,
        "urls": urls,
        "options": options if options else None,
    }

    print(f"\n🏷️  Sending {len(urls)} URLs to Intent Engine /rank-urls ...")
    print(f"   URL: {ranking_api_url}/rank-urls")

    try:
        resp = requests.post(f"{ranking_api_url}/rank-urls", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to Intent Engine API at {ranking_api_url}")
        print("   Make sure the API is running:")
        print('   cd "intent engine" && python main_api.py')
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Ranking API returned an error: {e}")
        print(f"   Response: {resp.text}")
        sys.exit(1)

    return resp.json()


def print_searxng_results(results: list[dict[str, Any]]) -> None:
    """Print raw SearXNG results for comparison."""
    print("\n" + "=" * 80)
    print("📋 ORIGINAL SearXNG Results (before re-ranking)")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        engines = ", ".join(r.get("engines", []))
        print(f"\n  {i:2d}. {title}")
        print(f"      🔗 {url}")
        if engines:
            print(f"      🔧 Engines: {engines}")


def print_ranked_results(
    ranking_response: dict[str, Any],
    searxng_results: list[dict[str, Any]],
) -> None:
    """Print the re-ranked results with privacy scores."""
    ranked = ranking_response.get("ranked_urls", [])
    proc_time = ranking_response.get("processing_time_ms", 0)
    total = ranking_response.get("total_urls", 0)
    filtered = ranking_response.get("filtered_count", 0)

    # Build a lookup from URL -> SearXNG metadata for enrichment
    searxng_lookup: dict[str, dict[str, Any]] = {}
    for r in searxng_results:
        searxng_lookup[r.get("url", "")] = r

    print("\n" + "=" * 80)
    print("🛡️  PRIVACY-RANKED Results (re-ranked by Intent Engine)")
    print("=" * 80)
    print(f'   Query: "{ranking_response.get("query", "")}"')
    print(f"   Total URLs: {total} | Filtered out: {filtered} | Ranked: {len(ranked)}")
    print(f"   Processing time: {proc_time:.1f}ms")

    if not ranked:
        print("\n   ⚠️  No results after filtering.")
        return

    for i, r in enumerate(ranked, 1):
        url = r.get("url", "")
        domain = r.get("domain", "")
        privacy = r.get("privacy_score", 0)
        relevance = r.get("relevance_score", 0)
        final = r.get("final_score", 0)
        trackers = r.get("tracker_count", 0)
        encrypted = r.get("encryption_enabled", True)
        is_oss = r.get("is_open_source", False)
        is_np = r.get("is_non_profit", False)
        content_type = r.get("content_type", "unknown")

        # Get original SearXNG title/content if available
        searxng_data = searxng_lookup.get(url, {})
        title = searxng_data.get("title") or r.get("title") or domain
        snippet = searxng_data.get("content", "")

        # Privacy badge
        if privacy >= 0.85:
            privacy_badge = "🟢 Excellent"
        elif privacy >= 0.7:
            privacy_badge = "🟡 Good"
        elif privacy >= 0.5:
            privacy_badge = "🟠 Fair"
        else:
            privacy_badge = "🔴 Poor"

        # Tags
        tags = []
        if is_oss:
            tags.append("📖 Open Source")
        if is_np:
            tags.append("🏛️ Non-Profit")
        if encrypted:
            tags.append("🔒 HTTPS")
        if trackers == 0:
            tags.append("🚫 No Trackers")
        elif trackers > 0:
            tags.append(f"⚠️ {trackers} Tracker{'s' if trackers > 1 else ''}")

        print(f"\n  {i:2d}. {title}")
        print(f"      🔗 {url}")
        if snippet:
            # Truncate snippet to 120 chars
            snippet_clean = snippet[:120].strip()
            if len(snippet) > 120:
                snippet_clean += "..."
            print(f"      📝 {snippet_clean}")
        print(f"      📊 Score: {final:.3f} | Relevance: {relevance:.3f} | Privacy: {privacy:.2f} ({privacy_badge})")
        print(f"      🏷️  {' | '.join(tags)}")
        print(f"      📂 Type: {content_type}")

    # Summary stats
    print("\n" + "-" * 80)
    print("📊 SUMMARY")
    print("-" * 80)

    avg_privacy = sum(r.get("privacy_score", 0) for r in ranked) / len(ranked) if ranked else 0
    oss_count = sum(1 for r in ranked if r.get("is_open_source", False))
    no_tracker_count = sum(1 for r in ranked if r.get("tracker_count", 0) == 0)

    print(f"   Average Privacy Score: {avg_privacy:.2f}")
    print(f"   Open Source Results:   {oss_count}/{len(ranked)}")
    print(f"   Zero-Tracker Results:  {no_tracker_count}/{len(ranked)}")
    print(f"   Processing Time:       {proc_time:.1f}ms")
    print()


def main():
    parser = argparse.ArgumentParser(description="Demo: SearXNG + Intent Engine Privacy-Aware URL Ranking")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--num-results", type=int, default=25, help="Number of URLs to fetch from SearXNG (default: 25)"
    )
    parser.add_argument(
        "--exclude-big-tech", action="store_true", help="Exclude big tech domains (Google, Facebook, Amazon, etc.)"
    )
    parser.add_argument("--min-privacy", type=float, default=0.0, help="Minimum privacy score threshold (0.0-1.0)")
    parser.add_argument("--searxng-url", default=SEARXNG_URL, help=f"SearXNG base URL (default: {SEARXNG_URL})")
    parser.add_argument(
        "--api-url", default=RANKING_API_URL, help=f"Intent Engine API base URL (default: {RANKING_API_URL})"
    )
    parser.add_argument("--categories", default="general", help="SearXNG search categories (default: general)")
    parser.add_argument(
        "--show-original", action="store_true", help="Also show original SearXNG results before re-ranking"
    )

    args = parser.parse_args()

    searxng_url = args.searxng_url
    ranking_api_url = args.api_url

    print("=" * 80)
    print("🛡️  SearXNG + Intent Engine — Privacy-Aware Search Demo")
    print("=" * 80)

    # Step 1: Query SearXNG
    searxng_results = query_searxng(
        query=args.query,
        num_results=args.num_results,
        categories=args.categories,
        searxng_url=searxng_url,
    )

    if not searxng_results:
        print("\nNo results to rank. Exiting.")
        sys.exit(0)

    # Optionally show original results
    if args.show_original:
        print_searxng_results(searxng_results)

    # Step 2: Extract URLs
    urls = [r["url"] for r in searxng_results if r.get("url")]
    print(f"\n📦 Extracted {len(urls)} URLs for ranking")

    # Step 3: Rank URLs via our API
    ranking_response = rank_urls(
        query=args.query,
        urls=urls,
        exclude_big_tech=args.exclude_big_tech,
        min_privacy_score=args.min_privacy,
        ranking_api_url=ranking_api_url,
    )

    # Step 4: Print results
    print_ranked_results(ranking_response, searxng_results)


if __name__ == "__main__":
    main()
