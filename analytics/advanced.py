"""
Advanced Analytics Module

This module implements advanced analytics capabilities including:
- Conversion attribution models (first-touch, last-touch, linear, time-decay)
- ROI/ROAS calculation
- Trend analysis
- Performance forecasting
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session


class AttributionModel(Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"  # 40% first, 40% last, 20% middle


@dataclass
class AttributionResult:
    """Result of attribution analysis for a conversion"""

    conversion_id: int
    touchpoints: list[dict[str, Any]]
    attribution_weights: dict[int, float]  # ad_id -> weight
    total_value: float
    attributed_values: dict[int, float]  # ad_id -> attributed value


@dataclass
class CampaignROI:
    """ROI metrics for a campaign"""

    campaign_id: int
    campaign_name: str
    total_spend: float
    total_revenue: float
    roi: float  # (revenue - spend) / spend * 100
    roas: float  # revenue / spend
    cpa: float  # cost per acquisition
    clv: float  # customer lifetime value estimate
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cvr: float  # conversion rate


@dataclass
class TrendAnalysis:
    """Trend analysis results"""

    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # "up", "down", "stable"
    data_points: list[dict[str, Any]]
    forecast_next_period: float | None


class AdvancedAnalytics:
    """
    Advanced analytics service for conversion attribution, ROI calculation, and trend analysis.
    """

    def __init__(self, db_session: Session):
        self.db = db_session

        # Time decay half-life in days (for time decay attribution)
        self.time_decay_half_life = 7

    def attribute_conversion(
        self, conversion_id: int, model: AttributionModel = AttributionModel.LAST_TOUCH
    ) -> AttributionResult:
        """
        Attribute a conversion to ad touchpoints using the specified model.
        """
        from database import ClickTracking, ConversionTracking

        # Get conversion
        conversion = (
            self.db.query(ConversionTracking)
            .filter(ConversionTracking.id == conversion_id)
            .first()
        )

        if not conversion:
            raise ValueError(f"Conversion {conversion_id} not found")

        # Get the click that led to conversion
        click = (
            self.db.query(ClickTracking)
            .filter(ClickTracking.id == conversion.click_id)
            .first()
        )

        if not click:
            raise ValueError(f"Click for conversion {conversion_id} not found")

        # Get all touchpoints for this session (within 30 days before conversion)
        lookback_window = (
            conversion.timestamp - timedelta(days=30)
            if conversion.timestamp
            else datetime.utcnow() - timedelta(days=30)
        )

        touchpoints = (
            self.db.query(ClickTracking)
            .filter(
                ClickTracking.session_id == click.session_id,
                ClickTracking.timestamp >= lookback_window,
                (
                    ClickTracking.timestamp <= conversion.timestamp
                    if conversion.timestamp
                    else datetime.utcnow()
                ),
            )
            .order_by(ClickTracking.timestamp)
            .all()
        )

        # If no touchpoints found, use the conversion click
        if not touchpoints:
            touchpoints = [click]

        # Calculate attribution weights based on model
        weights = self._calculate_attribution_weights(
            touchpoints, model, conversion.timestamp
        )

        # Calculate attributed values
        conversion_value = conversion.value or 0.0
        attributed_values = {
            ad_id: weight * conversion_value for ad_id, weight in weights.items()
        }

        # Build touchpoint data
        touchpoint_data = [
            {
                "click_id": tp.id,
                "ad_id": tp.ad_id,
                "timestamp": tp.timestamp.isoformat() if tp.timestamp else None,
                "weight": weights.get(tp.ad_id, 0),
            }
            for tp in touchpoints
        ]

        return AttributionResult(
            conversion_id=conversion_id,
            touchpoints=touchpoint_data,
            attribution_weights=weights,
            total_value=conversion_value,
            attributed_values=attributed_values,
        )

    def _calculate_attribution_weights(
        self,
        touchpoints: list,
        model: AttributionModel,
        conversion_time: datetime | None,
    ) -> dict[int, float]:
        """
        Calculate attribution weights for touchpoints based on model.
        """
        if not touchpoints:
            return {}

        # Group by ad_id (in case same ad was clicked multiple times)
        ad_touchpoints = defaultdict(list)
        for tp in touchpoints:
            ad_touchpoints[tp.ad_id].append(tp)

        weights = {}

        if model == AttributionModel.FIRST_TOUCH:
            # 100% credit to first touchpoint
            first_ad_id = touchpoints[0].ad_id
            weights = {
                ad_id: (1.0 if ad_id == first_ad_id else 0.0)
                for ad_id in ad_touchpoints.keys()
            }

        elif model == AttributionModel.LAST_TOUCH:
            # 100% credit to last touchpoint
            last_ad_id = touchpoints[-1].ad_id
            weights = {
                ad_id: (1.0 if ad_id == last_ad_id else 0.0)
                for ad_id in ad_touchpoints.keys()
            }

        elif model == AttributionModel.LINEAR:
            # Equal credit to all touchpoints
            n = len(touchpoints)
            ad_counts = defaultdict(int)
            for tp in touchpoints:
                ad_counts[tp.ad_id] += 1
            weights = {ad_id: count / n for ad_id, count in ad_counts.items()}

        elif model == AttributionModel.TIME_DECAY:
            # More credit to touchpoints closer to conversion
            if not conversion_time:
                conversion_time = datetime.utcnow()

            decay_weights = {}
            total_weight = 0

            for tp in touchpoints:
                if tp.timestamp:
                    days_before = (
                        conversion_time - tp.timestamp
                    ).total_seconds() / 86400
                    # Exponential decay
                    weight = math.pow(0.5, days_before / self.time_decay_half_life)
                else:
                    weight = 0.5

                if tp.ad_id not in decay_weights:
                    decay_weights[tp.ad_id] = 0
                decay_weights[tp.ad_id] += weight
                total_weight += weight

            # Normalize
            weights = (
                {ad_id: w / total_weight for ad_id, w in decay_weights.items()}
                if total_weight > 0
                else {}
            )

        elif model == AttributionModel.POSITION_BASED:
            # 40% first, 40% last, 20% distributed among middle
            if len(touchpoints) == 1:
                weights = {touchpoints[0].ad_id: 1.0}
            elif len(touchpoints) == 2:
                first_ad = touchpoints[0].ad_id
                last_ad = touchpoints[-1].ad_id
                if first_ad == last_ad:
                    weights = {first_ad: 1.0}
                else:
                    weights = {first_ad: 0.5, last_ad: 0.5}
            else:
                first_ad = touchpoints[0].ad_id
                last_ad = touchpoints[-1].ad_id
                middle_touchpoints = touchpoints[1:-1]

                weights = defaultdict(float)
                weights[first_ad] += 0.4
                weights[last_ad] += 0.4

                middle_weight = (
                    0.2 / len(middle_touchpoints) if middle_touchpoints else 0
                )
                for tp in middle_touchpoints:
                    weights[tp.ad_id] += middle_weight

                weights = dict(weights)

        return weights

    def calculate_campaign_roi(
        self,
        campaign_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> CampaignROI:
        """
        Calculate comprehensive ROI metrics for a campaign.
        """
        from database import (
            Ad,
            AdGroup,
            AdMetric,
            Campaign,
            ClickTracking,
            ConversionTracking,
        )

        # Get campaign
        campaign = self.db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Set date range
        if not start_date:
            start_date = campaign.start_date or (datetime.utcnow() - timedelta(days=30))
        if not end_date:
            end_date = campaign.end_date or datetime.utcnow()

        # Get ads for this campaign
        ad_ids = (
            self.db.query(Ad.id)
            .join(AdGroup)
            .filter(AdGroup.campaign_id == campaign_id)
            .all()
        )
        ad_ids = [a[0] for a in ad_ids]

        if not ad_ids:
            return CampaignROI(
                campaign_id=campaign_id,
                campaign_name=campaign.name,
                total_spend=0,
                total_revenue=0,
                roi=0,
                roas=0,
                cpa=0,
                clv=0,
                impressions=0,
                clicks=0,
                conversions=0,
                ctr=0,
                cvr=0,
            )

        # Get metrics
        metrics = (
            self.db.query(
                func.sum(AdMetric.impression_count).label("impressions"),
                func.sum(AdMetric.click_count).label("clicks"),
                func.sum(AdMetric.conversion_count).label("conversions"),
            )
            .filter(
                AdMetric.ad_id.in_(ad_ids),
                AdMetric.date >= start_date.date(),
                AdMetric.date <= end_date.date(),
            )
            .first()
        )

        impressions = metrics.impressions or 0
        clicks = metrics.clicks or 0
        conversions = metrics.conversions or 0

        # Calculate spend (impressions * bid amount)
        spend_data = (
            self.db.query(func.sum(AdMetric.impression_count * Ad.bid_amount))
            .select_from(AdMetric)
            .join(Ad)
            .filter(
                AdMetric.ad_id.in_(ad_ids),
                AdMetric.date >= start_date.date(),
                AdMetric.date <= end_date.date(),
            )
            .scalar()
            or 0
        )

        total_spend = float(spend_data)

        # Calculate revenue from conversions
        click_ids = (
            self.db.query(ClickTracking.id)
            .filter(
                ClickTracking.ad_id.in_(ad_ids),
                ClickTracking.timestamp >= start_date,
                ClickTracking.timestamp <= end_date,
            )
            .all()
        )
        click_ids = [c[0] for c in click_ids]

        if click_ids:
            total_revenue = (
                self.db.query(func.sum(ConversionTracking.value))
                .filter(
                    ConversionTracking.click_id.in_(click_ids),
                    ConversionTracking.status == "verified",
                )
                .scalar()
                or 0
            )
        else:
            total_revenue = 0

        total_revenue = float(total_revenue)

        # Calculate ROI metrics
        roi = (
            ((total_revenue - total_spend) / total_spend * 100)
            if total_spend > 0
            else 0
        )
        roas = (total_revenue / total_spend) if total_spend > 0 else 0
        cpa = (total_spend / conversions) if conversions > 0 else 0
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0

        # Estimate CLV (simple: average conversion value * estimated repeat rate)
        avg_conversion_value = (total_revenue / conversions) if conversions > 0 else 0
        estimated_repeat_rate = 1.5  # Assume 50% repeat
        clv = avg_conversion_value * estimated_repeat_rate

        return CampaignROI(
            campaign_id=campaign_id,
            campaign_name=campaign.name,
            total_spend=total_spend,
            total_revenue=total_revenue,
            roi=roi,
            roas=roas,
            cpa=cpa,
            clv=clv,
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            ctr=ctr,
            cvr=cvr,
        )

    def analyze_trend(
        self, metric_name: str, campaign_id: int | None = None, days: int = 30
    ) -> TrendAnalysis:
        """
        Analyze trends for a specific metric over time.
        """
        from database import Ad, AdGroup, AdMetric

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        mid_date = end_date - timedelta(days=days // 2)

        # Build base query
        query = self.db.query(
            AdMetric.date,
            func.sum(AdMetric.impression_count).label("impressions"),
            func.sum(AdMetric.click_count).label("clicks"),
            func.sum(AdMetric.conversion_count).label("conversions"),
        )

        if campaign_id:
            ad_ids = (
                self.db.query(Ad.id)
                .join(AdGroup)
                .filter(AdGroup.campaign_id == campaign_id)
                .all()
            )
            ad_ids = [a[0] for a in ad_ids]
            if ad_ids:
                query = query.filter(AdMetric.ad_id.in_(ad_ids))

        # Get daily data
        daily_data = (
            query.filter(
                AdMetric.date >= start_date.date(), AdMetric.date <= end_date.date()
            )
            .group_by(AdMetric.date)
            .order_by(AdMetric.date)
            .all()
        )

        # Build data points
        data_points = []
        for row in daily_data:
            impressions = row.impressions or 0
            clicks = row.clicks or 0
            conversions = row.conversions or 0

            ctr = (clicks / impressions * 100) if impressions > 0 else 0
            cvr = (conversions / clicks * 100) if clicks > 0 else 0

            data_points.append(
                {
                    "date": row.date.isoformat(),
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "ctr": ctr,
                    "cvr": cvr,
                }
            )

        # Calculate current and previous period values
        current_period = [
            d for d in data_points if d["date"] >= mid_date.date().isoformat()
        ]
        previous_period = [
            d for d in data_points if d["date"] < mid_date.date().isoformat()
        ]

        # Get values based on metric name
        metric_key = metric_name.lower()
        if metric_key not in ["impressions", "clicks", "conversions", "ctr", "cvr"]:
            metric_key = "impressions"

        current_value = sum(d[metric_key] for d in current_period) / max(
            len(current_period), 1
        )
        previous_value = sum(d[metric_key] for d in previous_period) / max(
            len(previous_period), 1
        )

        # Calculate change
        if previous_value > 0:
            change_percent = ((current_value - previous_value) / previous_value) * 100
        else:
            change_percent = 100 if current_value > 0 else 0

        # Determine trend direction
        if change_percent > 5:
            trend_direction = "up"
        elif change_percent < -5:
            trend_direction = "down"
        else:
            trend_direction = "stable"

        # Simple forecast (linear extrapolation)
        forecast = None
        if len(data_points) >= 7:
            recent_values = [d[metric_key] for d in data_points[-7:]]
            avg_daily_change = (
                (recent_values[-1] - recent_values[0]) / 6
                if len(recent_values) > 1
                else 0
            )
            forecast = recent_values[-1] + avg_daily_change * 7  # Forecast next week

        return TrendAnalysis(
            metric_name=metric_name,
            current_value=current_value,
            previous_value=previous_value,
            change_percent=change_percent,
            trend_direction=trend_direction,
            data_points=data_points,
            forecast_next_period=forecast,
        )

    def get_top_performing_ads(
        self, campaign_id: int | None = None, metric: str = "ctr", limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get top performing ads by specified metric.
        """
        from database import Ad, AdGroup, AdMetric

        # Build query
        query = self.db.query(
            Ad.id,
            Ad.title,
            func.sum(AdMetric.impression_count).label("impressions"),
            func.sum(AdMetric.click_count).label("clicks"),
            func.sum(AdMetric.conversion_count).label("conversions"),
        ).join(AdMetric)

        if campaign_id:
            query = query.join(AdGroup).filter(AdGroup.campaign_id == campaign_id)

        results = query.group_by(Ad.id, Ad.title).all()

        # Calculate metrics and sort
        ad_metrics = []
        for row in results:
            impressions = row.impressions or 0
            clicks = row.clicks or 0
            conversions = row.conversions or 0

            ctr = (clicks / impressions) if impressions > 0 else 0
            cvr = (conversions / clicks) if clicks > 0 else 0

            ad_metrics.append(
                {
                    "ad_id": row.id,
                    "title": row.title,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "ctr": ctr * 100,
                    "cvr": cvr * 100,
                }
            )

        # Sort by specified metric
        metric_key = metric.lower()
        if metric_key not in ["impressions", "clicks", "conversions", "ctr", "cvr"]:
            metric_key = "ctr"

        ad_metrics.sort(key=lambda x: x[metric_key], reverse=True)

        return ad_metrics[:limit]


def get_advanced_analytics(db_session: Session) -> AdvancedAnalytics:
    """Factory function to get advanced analytics instance"""
    return AdvancedAnalytics(db_session)
