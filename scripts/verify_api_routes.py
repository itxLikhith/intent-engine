"""
Quick test to verify the API endpoints are properly defined
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


from main_api import app


def test_api_routes():
    """Test that the new routes are registered"""
    routes = [route.path for route in app.routes]

    print("Registered API routes:")
    for route in sorted(routes):
        print(f"  {route}")

    # Check for new advertising system endpoints
    new_routes = [
        "/campaigns",
        "/campaigns/{campaign_id}",
        "/adgroups",
        "/adgroups/{ad_group_id}",
        "/creatives",
        "/creatives/{creative_id}",
        "/reports/campaign-performance",
        "/match-ads-advanced",
    ]

    print("\nChecking for new advertising system endpoints:")
    all_present = True
    for route in new_routes:
        # For routes with parameters, check the base path
        if "{" in route:
            base_route = route.split("{")[0][:-1]  # Remove the parameter part
            found = any(base_route in r for r in routes)
        else:
            found = route in routes

        status = "[OK]" if found else "[FAIL]"
        print(f"  {status} {route}")
        if not found:
            all_present = False

    print(f"\nAll new endpoints present: {'YES' if all_present else 'NO'}")
    return all_present


if __name__ == "__main__":
    test_api_routes()
