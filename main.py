"""
Intent Engine - Main Entry Point

This module provides the main entry point for the Intent Engine system.
It supports CLI commands and can serve as a server entry point.
"""

import argparse
import sys
from typing import Dict, Any
import json

from extraction.extractor import extract_intent, IntentExtractionRequest
from ranking.ranker import rank_results, RankingRequest, SearchResult
from services.recommender import recommend_services, ServiceRecommendationRequest, ServiceMetadata
from ads.matcher import match_ads, AdMatchingRequest, AdMetadata
from core.schema import UniversalIntent
from config.model_cache import initialize_models


def run_demo_search():
    """Run the search demo showing intent extraction"""
    from demos.demo_search import test_basic_functionality
    print("Running Intent Extraction Demo...")
    test_basic_functionality()


def run_demo_service():
    """Run the service recommendation demo"""
    from demos.demo_service import demo_service_recommendation
    print("Running Service Recommendation Demo...")
    demo_service_recommendation()


def run_demo_ad():
    """Run the ad matching demo"""
    from demos.demo_ad import demo_ad_matching
    print("Running Ad Matching Demo...")
    demo_ad_matching()


def run_demo_ranking():
    """Run the ranking demo"""
    from demos.demo_ranking import demo_ranking
    print("Running Ranking Demo...")
    demo_ranking()


def run_tests():
    """Run all unit tests"""
    import unittest
    import os
    import sys
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_perf_tests():
    """Run all performance tests"""
    import time
    
    print("Running Performance Tests...\n")
    
    # Import and run each performance test
    from perf_tests.perf_test_extractor import test_performance as test_extractor_perf
    print("Testing Intent Extractor Performance:")
    test_extractor_perf()
    print()
    
    from perf_tests.perf_test_ranker import test_performance as test_ranker_perf
    print("Testing Intent Ranker Performance:")
    test_ranker_perf()
    print()
    
    from perf_tests.perf_test_service import test_performance as test_service_perf
    print("Testing Service Recommender Performance:")
    test_service_perf()
    print()
    
    from perf_tests.perf_test_ad import test_performance as test_ad_perf
    print("Testing Ad Matcher Performance:")
    test_ad_perf()
    print()


def cli_main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Intent Engine - Privacy-First Intent Processing System')
    parser.add_argument('command', nargs='?', default='demo',
                        choices=['demo', 'demo-search', 'demo-service', 'demo-ad', 'demo-ranking', 
                                'test', 'perf-test', 'server', 'extract', 'rank', 'recommend', 'match'],
                        help='Command to execute')
    
    # Arguments for specific commands
    parser.add_argument('--query', '-q', type=str, help='Query text for intent processing')
    parser.add_argument('--input-file', '-i', type=str, help='Input file with query')
    parser.add_argument('--output-file', '-o', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize models on startup
    print("Initializing models...")
    initialize_models()
    print("Models initialized.\n")
    
    if args.command == 'demo':
        # Run all demos
        run_demo_search()
        print("\n" + "="*60 + "\n")
        run_demo_ranking()
        print("\n" + "="*60 + "\n")
        run_demo_service()
        print("\n" + "="*60 + "\n")
        run_demo_ad()
        
    elif args.command == 'demo-search':
        run_demo_search()
        
    elif args.command == 'demo-service':
        run_demo_service()
        
    elif args.command == 'demo-ad':
        run_demo_ad()
        
    elif args.command == 'demo-ranking':
        run_demo_ranking()
        
    elif args.command == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)
        
    elif args.command == 'perf-test':
        run_perf_tests()
        
    elif args.command == 'extract':
        # Process a single query for intent extraction
        query = args.query
        if not query:
            if args.input_file:
                with open(args.input_file, 'r') as f:
                    query = f.read().strip()
            else:
                print("Error: Either --query or --input-file must be provided for 'extract' command")
                sys.exit(1)
        
        # Create extraction request
        request = IntentExtractionRequest(
            product='cli',
            input={'text': query},
            context={'sessionId': 'cli-session', 'userLocale': 'en-US'}
        )
        
        # Extract intent
        response = extract_intent(request)
        intent_data = {
            'query': query,
            'intentId': response.intent.intentId,
            'declared': {
                'goal': response.intent.declared.goal.value if response.intent.declared.goal else None,
                'constraints': [
                    {'type': c.type.value, 'dimension': c.dimension, 'value': c.value, 'hardFilter': c.hardFilter}
                    for c in response.intent.declared.constraints
                ],
                'skillLevel': response.intent.declared.skillLevel.value
            },
            'inferred': {
                'useCases': [uc.value for uc in response.intent.inferred.useCases],
                'ethicalSignals': [
                    {'dimension': es.dimension.value, 'preference': es.preference}
                    for es in response.intent.inferred.ethicalSignals
                ]
            },
            'metrics': response.extractionMetrics
        }
        
        # Output result
        output = json.dumps(intent_data, indent=2)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
        else:
            print(output)
    
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli_main()