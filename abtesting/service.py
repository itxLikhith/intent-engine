"""
A/B Testing Service

This module implements A/B testing capabilities including:
- Test and variant management
- Traffic splitting algorithms
- Statistical significance calculation
- Winner determination
"""

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func

from database import Base


class ABTestStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTest(Base):
    """A/B Test configuration"""

    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    status = Column(String, default=ABTestStatus.DRAFT.value)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    traffic_allocation = Column(Float, default=1.0)  # Percentage of traffic to include in test
    min_sample_size = Column(Integer, default=1000)  # Minimum samples before significance
    confidence_level = Column(Float, default=0.95)  # Statistical confidence level
    primary_metric = Column(String, default="ctr")  # Primary metric to optimize
    winner_variant_id = Column(Integer)  # ID of winning variant (if determined)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    variants = relationship("ABTestVariant", back_populates="test", cascade="all, delete-orphan")


class ABTestVariant(Base):
    """A/B Test variant"""

    __tablename__ = "ab_test_variants"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id"), nullable=False)
    name = Column(String, nullable=False)  # e.g., "Control", "Variant A"
    ad_id = Column(Integer, ForeignKey("ads.id"))  # The ad to show for this variant
    traffic_weight = Column(Float, default=0.5)  # Weight for traffic splitting
    is_control = Column(Boolean, default=False)  # Is this the control variant
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    test = relationship("ABTest", back_populates="variants")


class ABTestAssignment(Base):
    """Tracks which variant a user was assigned to"""

    __tablename__ = "ab_test_assignments"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id"), nullable=False)
    variant_id = Column(Integer, ForeignKey("ab_test_variants.id"), nullable=False)
    user_hash = Column(String, nullable=False)  # Hashed user/session identifier
    assigned_at = Column(DateTime, server_default=func.now())


@dataclass
class VariantStats:
    """Statistics for a variant"""

    variant_id: int
    name: str
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    conversion_rate: float
    confidence_interval: tuple[float, float]
    is_winner: bool
    lift_vs_control: float | None


@dataclass
class TestResults:
    """Complete test results"""

    test_id: int
    status: str
    total_impressions: int
    variants: list[VariantStats]
    is_significant: bool
    winner_variant_id: int | None
    p_value: float | None
    recommended_action: str


class ABTestService:
    """Service for managing A/B tests"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_test(
        self,
        name: str,
        campaign_id: int,
        description: str = "",
        traffic_allocation: float = 1.0,
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
        primary_metric: str = "ctr",
    ) -> ABTest:
        """
        Create a new A/B test.
        """
        test = ABTest(
            name=name,
            campaign_id=campaign_id,
            description=description,
            traffic_allocation=traffic_allocation,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            primary_metric=primary_metric,
            status=ABTestStatus.DRAFT.value,
        )
        self.db.add(test)
        self.db.commit()
        self.db.refresh(test)
        return test

    def add_variant(
        self, test_id: int, name: str, ad_id: int, traffic_weight: float = 0.5, is_control: bool = False
    ) -> ABTestVariant:
        """
        Add a variant to an A/B test.
        """
        variant = ABTestVariant(
            test_id=test_id, name=name, ad_id=ad_id, traffic_weight=traffic_weight, is_control=is_control
        )
        self.db.add(variant)
        self.db.commit()
        self.db.refresh(variant)
        return variant

    def start_test(self, test_id: int) -> ABTest:
        """
        Start an A/B test.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test:
            raise ValueError(f"Test {test_id} not found")

        # Validate test has at least 2 variants
        if len(test.variants) < 2:
            raise ValueError("Test must have at least 2 variants")

        # Validate traffic weights sum to approximately 1
        total_weight = sum(v.traffic_weight for v in test.variants)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant traffic weights must sum to 1.0 (current: {total_weight})")

        test.status = ABTestStatus.RUNNING.value
        test.start_date = datetime.utcnow()
        self.db.commit()
        self.db.refresh(test)
        return test

    def pause_test(self, test_id: int) -> ABTest:
        """
        Pause an A/B test.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test:
            raise ValueError(f"Test {test_id} not found")

        test.status = ABTestStatus.PAUSED.value
        self.db.commit()
        self.db.refresh(test)
        return test

    def complete_test(self, test_id: int, winner_variant_id: int | None = None) -> ABTest:
        """
        Complete an A/B test and optionally declare a winner.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test:
            raise ValueError(f"Test {test_id} not found")

        test.status = ABTestStatus.COMPLETED.value
        test.end_date = datetime.utcnow()

        if winner_variant_id:
            test.winner_variant_id = winner_variant_id
        else:
            # Auto-determine winner based on results
            results = self.get_test_results(test_id)
            if results.winner_variant_id:
                test.winner_variant_id = results.winner_variant_id

        self.db.commit()
        self.db.refresh(test)
        return test

    def assign_variant(self, test_id: int, user_identifier: str) -> ABTestVariant | None:
        """
        Assign a user to a variant using consistent hashing.
        This ensures the same user always sees the same variant.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test or test.status != ABTestStatus.RUNNING.value:
            return None

        # Check for existing assignment
        user_hash = hashlib.sha256(f"{test_id}:{user_identifier}".encode()).hexdigest()
        existing = (
            self.db.query(ABTestAssignment)
            .filter(ABTestAssignment.test_id == test_id, ABTestAssignment.user_hash == user_hash)
            .first()
        )

        if existing:
            return self.db.query(ABTestVariant).filter(ABTestVariant.id == existing.variant_id).first()

        # Check if user should be included in test (traffic allocation)
        inclusion_hash = int(hashlib.md5(user_hash.encode()).hexdigest(), 16) % 100
        if inclusion_hash >= test.traffic_allocation * 100:
            return None  # User not included in test

        # Assign variant based on weighted random selection
        variant = self._select_variant_weighted(test.variants, user_hash)

        if variant:
            # Record assignment
            assignment = ABTestAssignment(test_id=test_id, variant_id=variant.id, user_hash=user_hash)
            self.db.add(assignment)
            self.db.commit()

        return variant

    def _select_variant_weighted(self, variants: list[ABTestVariant], seed: str) -> ABTestVariant | None:
        """
        Select a variant based on traffic weights using deterministic hashing.
        """
        if not variants:
            return None

        # Use hash to get consistent random value
        hash_value = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 10000 / 10000.0

        # Select variant based on cumulative weights
        cumulative_weight = 0.0
        for variant in sorted(variants, key=lambda v: v.id):
            cumulative_weight += variant.traffic_weight
            if hash_value < cumulative_weight:
                return variant

        return variants[-1]

    def record_impression(self, variant_id: int):
        """
        Record an impression for a variant.
        """
        variant = self.db.query(ABTestVariant).filter(ABTestVariant.id == variant_id).first()
        if variant:
            variant.impressions += 1
            self.db.commit()

    def record_click(self, variant_id: int):
        """
        Record a click for a variant.
        """
        variant = self.db.query(ABTestVariant).filter(ABTestVariant.id == variant_id).first()
        if variant:
            variant.clicks += 1
            self.db.commit()

    def record_conversion(self, variant_id: int, revenue: float = 0.0):
        """
        Record a conversion for a variant.
        """
        variant = self.db.query(ABTestVariant).filter(ABTestVariant.id == variant_id).first()
        if variant:
            variant.conversions += 1
            variant.revenue += revenue
            self.db.commit()

    def get_test_results(self, test_id: int) -> TestResults:
        """
        Get comprehensive test results with statistical analysis.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test:
            raise ValueError(f"Test {test_id} not found")

        variants = test.variants
        total_impressions = sum(v.impressions for v in variants)

        # Find control variant
        control = next((v for v in variants if v.is_control), variants[0] if variants else None)

        # Calculate stats for each variant
        variant_stats = []
        for variant in variants:
            ctr = variant.clicks / variant.impressions if variant.impressions > 0 else 0
            conv_rate = variant.conversions / variant.clicks if variant.clicks > 0 else 0

            # Calculate confidence interval for CTR
            ci = self._calculate_confidence_interval(variant.clicks, variant.impressions, test.confidence_level)

            # Calculate lift vs control
            lift = None
            if control and control.id != variant.id and control.impressions > 0:
                control_ctr = control.clicks / control.impressions
                if control_ctr > 0:
                    lift = (ctr - control_ctr) / control_ctr * 100

            variant_stats.append(
                VariantStats(
                    variant_id=variant.id,
                    name=variant.name,
                    impressions=variant.impressions,
                    clicks=variant.clicks,
                    conversions=variant.conversions,
                    ctr=ctr * 100,  # as percentage
                    conversion_rate=conv_rate * 100,
                    confidence_interval=(ci[0] * 100, ci[1] * 100),
                    is_winner=False,
                    lift_vs_control=lift,
                )
            )

        # Determine statistical significance and winner
        is_significant = False
        winner_id = None
        p_value = None

        if len(variants) >= 2 and control:
            # Compare best performing variant against control
            best_variant = max(
                [v for v in variants if v.id != control.id],
                key=lambda v: v.clicks / v.impressions if v.impressions > 0 else 0,
                default=None,
            )

            if best_variant:
                p_value = self._calculate_p_value(
                    control.clicks, control.impressions, best_variant.clicks, best_variant.impressions
                )

                is_significant = p_value < (1 - test.confidence_level)

                if is_significant and total_impressions >= test.min_sample_size:
                    # Determine winner based on primary metric
                    best_ctr = best_variant.clicks / best_variant.impressions if best_variant.impressions > 0 else 0
                    control_ctr = control.clicks / control.impressions if control.impressions > 0 else 0

                    if best_ctr > control_ctr:
                        winner_id = best_variant.id
                    else:
                        winner_id = control.id

                    # Mark winner in stats
                    for stat in variant_stats:
                        if stat.variant_id == winner_id:
                            stat.is_winner = True

        # Determine recommended action
        if is_significant and winner_id:
            recommended_action = f"Declare variant {winner_id} as winner and implement"
        elif total_impressions < test.min_sample_size:
            remaining = test.min_sample_size - total_impressions
            recommended_action = f"Continue test - need {remaining} more impressions"
        elif not is_significant:
            recommended_action = "Continue test - no significant difference yet"
        else:
            recommended_action = "Review results manually"

        return TestResults(
            test_id=test_id,
            status=test.status,
            total_impressions=total_impressions,
            variants=variant_stats,
            is_significant=is_significant,
            winner_variant_id=winner_id,
            p_value=p_value,
            recommended_action=recommended_action,
        )

    def _calculate_confidence_interval(self, successes: int, trials: int, confidence: float) -> tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        """
        if trials == 0:
            return (0.0, 0.0)

        # Wilson score interval
        p = successes / trials
        z = self._get_z_score(confidence)

        denominator = 1 + z * z / trials
        center = (p + z * z / (2 * trials)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator

        return (max(0, center - spread), min(1, center + spread))

    def _calculate_p_value(
        self, control_successes: int, control_trials: int, treatment_successes: int, treatment_trials: int
    ) -> float:
        """
        Calculate p-value using two-proportion z-test.
        """
        if control_trials == 0 or treatment_trials == 0:
            return 1.0

        p1 = control_successes / control_trials
        p2 = treatment_successes / treatment_trials

        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_trials + treatment_trials)

        if p_pooled == 0 or p_pooled == 1:
            return 1.0

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / control_trials + 1 / treatment_trials))

        if se == 0:
            return 1.0

        # Z-score
        z = abs(p1 - p2) / se

        # Convert to p-value (two-tailed)
        p_value = 2 * (1 - self._normal_cdf(z))

        return p_value

    def _get_z_score(self, confidence: float) -> float:
        """
        Get z-score for given confidence level.
        """
        # Common z-scores
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        return z_scores.get(confidence, 1.96)

    def _normal_cdf(self, x: float) -> float:
        """
        Approximation of standard normal CDF.
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def get_all_tests(self, campaign_id: int | None = None, status: str | None = None) -> list[ABTest]:
        """
        Get all A/B tests with optional filters.
        """
        query = self.db.query(ABTest)

        if campaign_id:
            query = query.filter(ABTest.campaign_id == campaign_id)

        if status:
            query = query.filter(ABTest.status == status)

        return query.all()

    def get_test(self, test_id: int) -> ABTest | None:
        """
        Get a specific test by ID.
        """
        return self.db.query(ABTest).filter(ABTest.id == test_id).first()

    def delete_test(self, test_id: int) -> bool:
        """
        Delete a test and all its variants.
        """
        test = self.db.query(ABTest).filter(ABTest.id == test_id).first()
        if not test:
            return False

        # Delete assignments first
        self.db.query(ABTestAssignment).filter(ABTestAssignment.test_id == test_id).delete()

        # Delete test (variants will be cascade deleted)
        self.db.delete(test)
        self.db.commit()
        return True


def get_ab_test_service(db_session: Session) -> ABTestService:
    """Factory function to get A/B test service instance"""
    return ABTestService(db_session)
