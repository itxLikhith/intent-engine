"""
Intent Engine - Load Testing Module

This module provides comprehensive load testing capabilities:
- Locust-based load testing (locustfile.py)
- Stress testing (stress_test.py)
- Performance benchmarks

Usage:
    # Run Locust load test
    locust -f load_testing/locustfile.py --host=http://localhost:8000
    
    # Run stress tests
    python -m load_testing.stress_test
"""

__version__ = "1.0.0"
__all__ = [
    "StressTestSuite",
    "IntentEngineLoadTest",
    "CampaignManagementLoadTest",
]

# Make imports available
try:
    from .stress_test import StressTestSuite
except ImportError:
    pass

try:
    from .locustfile import IntentEngineLoadTest, CampaignManagementLoadTest
except ImportError:
    pass
