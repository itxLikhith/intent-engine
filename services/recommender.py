"""
Intent Engine - Phase 3: Service Recommendation and Cross-Product Routing

This module implements service recommendation logic based on user intent.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import logging
from datetime import timezone

from core.schema import (
    UniversalIntent, IntentGoal, UseCase, EthicalSignal, EthicalDimension,
    TemporalIntent, TemporalHorizon, Recency, Frequency
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServiceMetadata:
    """Metadata for a service in the workspace ecosystem"""
    id: str
    name: str
    supportedGoals: List[str]  # List of supported IntentGoal values
    primaryUseCases: List[str]  # List of primary use cases
    temporalPatterns: List[str]  # List of temporal patterns
    ethicalAlignment: List[str]  # List of ethical alignments
    description: Optional[str] = None  # Optional description


@dataclass
class ServiceRecommendation:
    """Represents a recommended service with score and reasons"""
    service: ServiceMetadata
    serviceScore: float
    matchReasons: List[str]


@dataclass
class ServiceRecommendationRequest:
    """Request object for service recommendation API"""
    intent: UniversalIntent
    availableServices: List[ServiceMetadata]
    options: Optional[Dict[str, Any]] = None  # RecommendationOptions


@dataclass
class ServiceRecommendationResponse:
    """Response object for service recommendation API"""
    recommendations: List[ServiceRecommendation]


class EmbeddingCache:
    """Cache for embeddings to improve performance"""
    
    def __init__(self):
        self.cache = {}
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Use a lightweight model optimized for CPU
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move model to CPU
            self.model = self.model.to('cpu')
            
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning("Transformers library not available. Using mock embeddings.")
            self.tokenizer = None
            self.model = None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector using the sentence transformer model"""
        if self.model is None or self.tokenizer is None:
            # Return random vector for mock implementation
            return np.random.rand(384).astype(np.float32)
        
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        try:
            import torch
            
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            result = embeddings.cpu().numpy().flatten().astype(np.float32)
            
            # Cache the result
            self.cache[text] = result
            
            return result
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return float(dot_product / (norm_vec1 * norm_vec2))


class ServiceScoringEngine:
    """Engine for scoring services based on user intent"""
    
    def __init__(self):
        self.embedding_cache = EmbeddingCache()
    
    def compute_service_match(self, intent: UniversalIntent, service: ServiceMetadata) -> Tuple[float, List[str]]:
        """
        Compute match score between intent and service
        """
        scores = []
        reasons = []
        
        # 1. Goal matching
        goal_score, goal_reasons = self._compute_goal_matching(intent, service)
        scores.append(goal_score)
        reasons.extend(goal_reasons)
        
        # 2. Use case matching (semantic similarity)
        use_case_score, use_case_reasons = self._compute_use_case_matching(intent, service)
        scores.append(use_case_score)
        reasons.extend(use_case_reasons)
        
        # 3. Temporal pattern matching
        temporal_score, temporal_reasons = self._compute_temporal_matching(intent, service)
        scores.append(temporal_score)
        reasons.extend(temporal_reasons)
        
        # 4. Ethical alignment matching
        ethical_score, ethical_reasons = self._compute_ethical_matching(intent, service)
        scores.append(ethical_score)
        reasons.extend(ethical_reasons)
        
        # Weighted combination of scores
        weights = [0.40, 0.30, 0.15, 0.15]  # Sum to 1.0
        
        # Ensure we have the right number of weights
        if len(scores) != len(weights):
            # Adjust weights if needed
            adjusted_weights = [w for w in weights[:len(scores)]]
            remaining_weight = 1.0 - sum(adjusted_weights)
            if remaining_weight > 0 and len(adjusted_weights) > 0:
                adjusted_weights[0] += remaining_weight
            elif remaining_weight > 0:
                adjusted_weights = [remaining_weight]
            weights = adjusted_weights
        
        service_score = sum(score * weight for score, weight in zip(scores, weights))
        
        # Clamp the score between 0 and 1
        service_score = max(0.0, min(1.0, service_score))
        
        return service_score, reasons
    
    def _compute_goal_matching(self, intent: UniversalIntent, service: ServiceMetadata) -> Tuple[float, List[str]]:
        """Compute score based on goal matching"""
        reasons = []
        
        if not intent.declared.goal:
            return 0.5, reasons  # Neutral score if no goal
        
        goal_value = intent.declared.goal.value
        
        if goal_value in service.supportedGoals:
            score = 1.0
            reasons.append(f"goal={goal_value} supported")
            return score, reasons
        else:
            # Check for semantic similarity with supported goals
            goal_embedding = self.embedding_cache.encode_text(goal_value.replace('_', ' '))
            service_goals_text = " ".join(service.supportedGoals).replace('_', ' ')
            service_goals_embedding = self.embedding_cache.encode_text(service_goals_text)
            
            if goal_embedding is not None and service_goals_embedding is not None:
                similarity = self.embedding_cache.cosine_similarity(goal_embedding, service_goals_embedding)
                # Normalize to 0-1 range
                score = (similarity + 1) / 2
                if score > 0.3:  # Threshold for relevance
                    reasons.append(f"goal-semantically-matched ({goal_value})")
                return score, reasons
            else:
                # Fallback: partial match based on keywords
                goal_lower = goal_value.lower()
                service_goals_lower = [g.lower() for g in service.supportedGoals]
                
                for service_goal in service_goals_lower:
                    if goal_lower in service_goal or service_goal in goal_lower:
                        score = 0.7
                        reasons.append(f"goal-partially-matched ({goal_value})")
                        return score, reasons
                
                return 0.1, reasons  # Low score if no match
    
    def _compute_use_case_matching(self, intent: UniversalIntent, service: ServiceMetadata) -> Tuple[float, List[str]]:
        """Compute score based on use case matching"""
        reasons = []
        
        if not intent.inferred.useCases:
            return 0.5, reasons  # Neutral score if no use cases
        
        if not service.primaryUseCases:
            return 0.0, reasons  # No use cases to match against
        
        total_score = 0.0
        matches_found = 0
        
        for use_case in intent.inferred.useCases:
            use_case_str = use_case.value.replace('_', ' ')
            
            # Check for direct matches
            direct_match = False
            for service_use_case in service.primaryUseCases:
                if service_use_case.lower() in use_case_str.lower() or use_case_str.lower() in service_use_case.lower():
                    total_score += 1.0
                    matches_found += 1
                    reasons.append(f"use case '{service_use_case}' matched")
                    direct_match = True
                    break
            
            if not direct_match:
                # Use semantic similarity as fallback
                use_case_embedding = self.embedding_cache.encode_text(use_case_str)
                service_use_cases_embedding = self.embedding_cache.encode_text(" ".join(service.primaryUseCases))
                
                if use_case_embedding is not None and service_use_cases_embedding is not None:
                    similarity = self.embedding_cache.cosine_similarity(use_case_embedding, service_use_cases_embedding)
                    # Normalize to 0-1 range
                    similarity_score = (similarity + 1) / 2
                    total_score += similarity_score
                    matches_found += 1
                    if similarity_score > 0.3:
                        reasons.append(f"use case semantically matched '{use_case_str}'")
        
        # Normalize score based on number of use cases
        if matches_found > 0:
            avg_score = total_score / len(intent.inferred.useCases)
            return min(1.0, avg_score), reasons
        else:
            return 0.0, reasons
    
    def _compute_temporal_matching(self, intent: UniversalIntent, service: ServiceMetadata) -> Tuple[float, List[str]]:
        """Compute score based on temporal pattern matching"""
        reasons = []
        
        temporal_intent = intent.inferred.temporalIntent
        if not temporal_intent:
            return 0.5, reasons  # Neutral score if no temporal intent
        
        score = 0.5  # Base score
        
        # Check horizon matching
        if temporal_intent.horizon == TemporalHorizon.TODAY:
            if 'short_session' in service.temporalPatterns or 'quick_access' in service.temporalPatterns:
                score += 0.2
                reasons.append("temporal-horizon-today-matched")
        elif temporal_intent.horizon in [TemporalHorizon.WEEK, TemporalHorizon.LONGTERM]:
            if 'long_session' in service.temporalPatterns or 'extended_work' in service.temporalPatterns:
                score += 0.2
                reasons.append("temporal-horizon-longterm-matched")
        
        # Check frequency matching
        if temporal_intent.frequency == Frequency.RECURRING:
            if 'recurring_edit' in service.temporalPatterns or 'regular_updates' in service.temporalPatterns:
                score += 0.15
                reasons.append("temporal-frequency-recurring-matched")
        elif temporal_intent.frequency == Frequency.ONEOFF:
            if 'one_time_task' in service.temporalPatterns or 'quick_completion' in service.temporalPatterns:
                score += 0.15
                reasons.append("temporal-frequency-oneoff-matched")
        
        # Check recency matching
        if temporal_intent.recency == Recency.RECENT:
            if 'real_time' in service.temporalPatterns or 'live_updates' in service.temporalPatterns:
                score += 0.1
                reasons.append("temporal-recency-recent-matched")
        
        # Clamp score between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score, reasons
    
    def _compute_ethical_matching(self, intent: UniversalIntent, service: ServiceMetadata) -> Tuple[float, List[str]]:
        """Compute score based on ethical alignment matching"""
        reasons = []
        
        if not intent.inferred.ethicalSignals:
            return 0.5, reasons  # Neutral score if no ethical signals
        
        score = 0.0
        matches_found = 0
        
        for signal in intent.inferred.ethicalSignals:
            # Check dimension and preference alignment
            signal_dimension = signal.dimension.value
            signal_preference = signal.preference
            
            # Look for matches in service ethical alignment
            for eth_align in service.ethicalAlignment:
                if signal_dimension.lower() in eth_align.lower() or signal_preference.lower() in eth_align.lower():
                    score += 0.5
                    matches_found += 1
                    reasons.append(f"ethical alignment: {eth_align}")
                    break
        
        # Normalize score based on number of ethical signals
        if matches_found > 0:
            avg_score = score / len(intent.inferred.ethicalSignals)
            return min(1.0, avg_score), reasons
        else:
            return 0.0, reasons


class ServiceRecommender:
    """Main class that orchestrates service recommendation"""
    
    def __init__(self):
        self.scoring_engine = ServiceScoringEngine()
    
    def recommend_services(self, request: ServiceRecommendationRequest) -> ServiceRecommendationResponse:
        """
        Main function to recommend services based on user intent
        """
        recommendations = []
        
        for service in request.availableServices:
            service_score, match_reasons = self.scoring_engine.compute_service_match(
                request.intent, 
                service
            )
            
            # Only include recommendations above a minimum threshold
            if service_score > 0.1:  # Minimum relevance threshold
                recommendation = ServiceRecommendation(
                    service=service,
                    serviceScore=service_score,
                    matchReasons=match_reasons
                )
                recommendations.append(recommendation)
        
        # Sort by service score (descending)
        recommendations.sort(key=lambda x: x.serviceScore, reverse=True)
        
        return ServiceRecommendationResponse(recommendations=recommendations)


# Global instance for caching
_service_recommender_instance = None


def get_service_recommender() -> ServiceRecommender:
    """
    Get singleton instance of ServiceRecommender with cached model
    """
    global _service_recommender_instance
    if _service_recommender_instance is None:
        _service_recommender_instance = ServiceRecommender()
    return _service_recommender_instance


def recommend_services(request: ServiceRecommendationRequest) -> ServiceRecommendationResponse:
    """
    Main entry point for recommending services based on intent
    """
    recommender = get_service_recommender()
    return recommender.recommend_services(request)