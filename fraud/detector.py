"""
Fraud Detection Service

This module implements comprehensive fraud detection algorithms including:
- Click fraud detection (rapid clicks, bot patterns)
- Velocity checks
- IP/User agent anomaly detection
- Bot detection using behavioral patterns
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session


class FraudSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudType(Enum):
    CLICK_FRAUD = "click_fraud"
    BOT_TRAFFIC = "bot_traffic"
    VELOCITY_VIOLATION = "velocity_violation"
    IP_ANOMALY = "ip_anomaly"
    IMPRESSION_FRAUD = "impression_fraud"
    CONVERSION_FRAUD = "conversion_fraud"
    DEVICE_SPOOFING = "device_spoofing"


@dataclass
class FraudSignal:
    """Represents a detected fraud signal"""

    fraud_type: FraudType
    severity: FraudSeverity
    confidence: float  # 0.0 to 1.0
    reason: str
    metadata: Dict[str, Any]


@dataclass
class FraudAnalysisResult:
    """Result of fraud analysis"""

    is_fraudulent: bool
    risk_score: float  # 0.0 to 1.0
    signals: List[FraudSignal]
    recommended_action: str


# Known bot user agent patterns
BOT_USER_AGENT_PATTERNS = [
    r"bot",
    r"crawler",
    r"spider",
    r"scraper",
    r"curl",
    r"wget",
    r"python-requests",
    r"httpx",
    r"headless",
    r"phantom",
    r"selenium",
    r"puppeteer",
    r"googlebot",
    r"bingbot",
    r"yandex",
    r"baidu",
]

# Suspicious user agent characteristics
SUSPICIOUS_UA_PATTERNS = [
    r"^$",  # Empty user agent
    r"^-$",  # Dash only
    r"^Mozilla/4\.0$",  # Very old Mozilla without details
]


class FraudDetector:
    """
    Comprehensive fraud detection service for ad clicks, impressions, and conversions.
    """

    def __init__(self, db_session: Session):
        self.db = db_session

        # Configuration thresholds
        self.config = {
            # Click velocity thresholds
            "max_clicks_per_ip_per_minute": 5,
            "max_clicks_per_ip_per_hour": 30,
            "max_clicks_per_session_per_minute": 3,
            "max_clicks_per_ad_per_ip_per_day": 5,
            # Conversion thresholds
            "min_time_to_conversion_seconds": 5,  # Too fast = suspicious
            "max_conversion_rate_threshold": 0.5,  # 50% conversion rate is suspicious
            # Impression thresholds
            "max_impressions_per_ip_per_minute": 100,
            # Risk score thresholds
            "low_risk_threshold": 0.3,
            "medium_risk_threshold": 0.5,
            "high_risk_threshold": 0.7,
        }

    def analyze_click(self, click_data: Dict[str, Any]) -> FraudAnalysisResult:
        """
        Analyze a click event for potential fraud.

        Args:
            click_data: Dict containing ad_id, session_id, ip_hash, user_agent_hash, timestamp, etc.

        Returns:
            FraudAnalysisResult with risk assessment
        """
        signals: List[FraudSignal] = []

        # Run all detection algorithms
        signals.extend(self._check_click_velocity(click_data))
        signals.extend(self._check_bot_user_agent(click_data))
        signals.extend(self._check_ip_anomalies(click_data))
        signals.extend(self._check_click_patterns(click_data))
        signals.extend(self._check_device_fingerprint(click_data))

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(signals)

        # Determine if fraudulent based on risk score
        is_fraudulent = risk_score >= self.config["high_risk_threshold"]

        # Determine recommended action
        recommended_action = self._get_recommended_action(risk_score, signals)

        return FraudAnalysisResult(
            is_fraudulent=is_fraudulent, risk_score=risk_score, signals=signals, recommended_action=recommended_action
        )

    def analyze_conversion(self, conversion_data: Dict[str, Any]) -> FraudAnalysisResult:
        """
        Analyze a conversion event for potential fraud.
        """
        signals: List[FraudSignal] = []

        # Check time between click and conversion
        signals.extend(self._check_conversion_timing(conversion_data))

        # Check conversion patterns
        signals.extend(self._check_conversion_patterns(conversion_data))

        # Calculate risk score
        risk_score = self._calculate_risk_score(signals)
        is_fraudulent = risk_score >= self.config["high_risk_threshold"]
        recommended_action = self._get_recommended_action(risk_score, signals)

        return FraudAnalysisResult(
            is_fraudulent=is_fraudulent, risk_score=risk_score, signals=signals, recommended_action=recommended_action
        )

    def analyze_impression(self, impression_data: Dict[str, Any]) -> FraudAnalysisResult:
        """
        Analyze an impression event for potential fraud.
        """
        signals: List[FraudSignal] = []

        # Check impression velocity
        signals.extend(self._check_impression_velocity(impression_data))

        # Check for bot traffic
        signals.extend(self._check_bot_user_agent(impression_data))

        # Calculate risk score
        risk_score = self._calculate_risk_score(signals)
        is_fraudulent = risk_score >= self.config["high_risk_threshold"]
        recommended_action = self._get_recommended_action(risk_score, signals)

        return FraudAnalysisResult(
            is_fraudulent=is_fraudulent, risk_score=risk_score, signals=signals, recommended_action=recommended_action
        )

    def _check_click_velocity(self, click_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for abnormal click velocity patterns.
        """
        signals = []
        ip_hash = click_data.get("ip_hash")
        session_id = click_data.get("session_id")
        ad_id = click_data.get("ad_id")

        if not ip_hash:
            return signals

        from database import ClickTracking as DbClickTracking

        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        # Check clicks per IP per minute
        clicks_per_minute = (
            self.db.query(DbClickTracking)
            .filter(DbClickTracking.ip_hash == ip_hash, DbClickTracking.timestamp >= one_minute_ago)
            .count()
        )

        if clicks_per_minute >= self.config["max_clicks_per_ip_per_minute"]:
            threshold = self.config["max_clicks_per_ip_per_minute"]
            signals.append(
                FraudSignal(
                    fraud_type=FraudType.VELOCITY_VIOLATION,
                    severity=FraudSeverity.HIGH,
                    confidence=0.9,
                    reason=f"Excessive clicks per minute: {clicks_per_minute} (threshold: {threshold})",
                    metadata={"clicks_per_minute": clicks_per_minute, "ip_hash": ip_hash},
                )
            )

        # Check clicks per IP per hour
        clicks_per_hour = (
            self.db.query(DbClickTracking)
            .filter(DbClickTracking.ip_hash == ip_hash, DbClickTracking.timestamp >= one_hour_ago)
            .count()
        )

        if clicks_per_hour >= self.config["max_clicks_per_ip_per_hour"]:
            threshold = self.config["max_clicks_per_ip_per_hour"]
            signals.append(
                FraudSignal(
                    fraud_type=FraudType.VELOCITY_VIOLATION,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.8,
                    reason=f"Excessive clicks per hour: {clicks_per_hour} (threshold: {threshold})",
                    metadata={"clicks_per_hour": clicks_per_hour, "ip_hash": ip_hash},
                )
            )

        # Check clicks per session per minute
        if session_id:
            session_clicks_per_minute = (
                self.db.query(DbClickTracking)
                .filter(DbClickTracking.session_id == session_id, DbClickTracking.timestamp >= one_minute_ago)
                .count()
            )

            if session_clicks_per_minute >= self.config["max_clicks_per_session_per_minute"]:
                signals.append(
                    FraudSignal(
                        fraud_type=FraudType.CLICK_FRAUD,
                        severity=FraudSeverity.HIGH,
                        confidence=0.85,
                        reason=f"Excessive clicks per session per minute: {session_clicks_per_minute}",
                        metadata={"session_clicks": session_clicks_per_minute, "session_id": session_id},
                    )
                )

        # Check clicks on same ad from same IP per day
        if ad_id:
            ad_clicks_per_day = (
                self.db.query(DbClickTracking)
                .filter(
                    DbClickTracking.ip_hash == ip_hash,
                    DbClickTracking.ad_id == ad_id,
                    DbClickTracking.timestamp >= one_day_ago,
                )
                .count()
            )

            if ad_clicks_per_day >= self.config["max_clicks_per_ad_per_ip_per_day"]:
                signals.append(
                    FraudSignal(
                        fraud_type=FraudType.CLICK_FRAUD,
                        severity=FraudSeverity.MEDIUM,
                        confidence=0.75,
                        reason=f"Multiple clicks on same ad from same IP: {ad_clicks_per_day}",
                        metadata={"ad_clicks": ad_clicks_per_day, "ad_id": ad_id},
                    )
                )

        return signals

    def _check_bot_user_agent(self, data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check if user agent indicates bot traffic.
        """
        signals = []
        user_agent = data.get("user_agent", "") or data.get("user_agent_hash", "")

        if not user_agent:
            signals.append(
                FraudSignal(
                    fraud_type=FraudType.BOT_TRAFFIC,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.6,
                    reason="Missing user agent",
                    metadata={"user_agent": None},
                )
            )
            return signals

        # Check for known bot patterns
        user_agent_lower = user_agent.lower()
        for pattern in BOT_USER_AGENT_PATTERNS:
            if re.search(pattern, user_agent_lower):
                signals.append(
                    FraudSignal(
                        fraud_type=FraudType.BOT_TRAFFIC,
                        severity=FraudSeverity.HIGH,
                        confidence=0.95,
                        reason=f"Known bot user agent pattern detected: {pattern}",
                        metadata={"user_agent": user_agent, "pattern": pattern},
                    )
                )
                break

        # Check for suspicious patterns
        for pattern in SUSPICIOUS_UA_PATTERNS:
            if re.match(pattern, user_agent):
                signals.append(
                    FraudSignal(
                        fraud_type=FraudType.BOT_TRAFFIC,
                        severity=FraudSeverity.MEDIUM,
                        confidence=0.7,
                        reason=f"Suspicious user agent pattern: {pattern}",
                        metadata={"user_agent": user_agent},
                    )
                )
                break

        return signals

    def _check_ip_anomalies(self, click_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for IP-based anomalies like datacenter IPs, VPNs, proxies.
        """
        signals = []
        ip_hash = click_data.get("ip_hash")

        if not ip_hash:
            return signals

        from database import ClickTracking as DbClickTracking

        # Check if multiple user agents from same IP (proxy/VPN indicator)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        unique_user_agents = (
            self.db.query(func.count(func.distinct(DbClickTracking.user_agent_hash)))
            .filter(DbClickTracking.ip_hash == ip_hash, DbClickTracking.timestamp >= one_hour_ago)
            .scalar()
            or 0
        )

        if unique_user_agents >= 5:
            signals.append(
                FraudSignal(
                    fraud_type=FraudType.IP_ANOMALY,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.7,
                    reason=f"Multiple user agents ({unique_user_agents}) from same IP - possible proxy/VPN",
                    metadata={"unique_user_agents": unique_user_agents, "ip_hash": ip_hash},
                )
            )

        return signals

    def _check_click_patterns(self, click_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for suspicious click patterns (e.g., clicks without impressions).
        """
        signals = []

        # Check for rapid consecutive clicks (machine-like timing)
        ip_hash = click_data.get("ip_hash")
        if not ip_hash:
            return signals

        from database import ClickTracking as DbClickTracking

        # Get recent clicks from this IP
        five_seconds_ago = datetime.utcnow() - timedelta(seconds=5)
        recent_clicks = (
            self.db.query(DbClickTracking)
            .filter(DbClickTracking.ip_hash == ip_hash, DbClickTracking.timestamp >= five_seconds_ago)
            .order_by(DbClickTracking.timestamp.desc())
            .limit(10)
            .all()
        )

        if len(recent_clicks) >= 3:
            # Check for suspiciously regular timing intervals
            timestamps = [c.timestamp for c in recent_clicks if c.timestamp]
            if len(timestamps) >= 3:
                intervals = []
                for i in range(len(timestamps) - 1):
                    if timestamps[i] and timestamps[i + 1]:
                        interval = abs((timestamps[i] - timestamps[i + 1]).total_seconds())
                        intervals.append(interval)

                if intervals:
                    # Check if intervals are suspiciously similar (machine-like)
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)

                    # Low variance in intervals suggests automated clicking
                    if variance < 0.1 and avg_interval < 1.0:
                        signals.append(
                            FraudSignal(
                                fraud_type=FraudType.CLICK_FRAUD,
                                severity=FraudSeverity.HIGH,
                                confidence=0.85,
                                reason=f"Machine-like click timing pattern detected (variance: {variance:.4f})",
                                metadata={"intervals": intervals, "variance": variance},
                            )
                        )

        return signals

    def _check_device_fingerprint(self, click_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for device fingerprint anomalies.
        """
        signals = []

        # Check for mismatched device characteristics
        user_agent = click_data.get("user_agent", "")
        if user_agent:
            # Check for desktop browser claiming to be mobile or vice versa
            is_mobile_ua = any(m in user_agent.lower() for m in ["mobile", "android", "iphone", "ipad"])
            screen_width = click_data.get("screen_width", 0)

            if screen_width:
                is_mobile_screen = screen_width < 768
                if is_mobile_ua != is_mobile_screen:
                    signals.append(
                        FraudSignal(
                            fraud_type=FraudType.DEVICE_SPOOFING,
                            severity=FraudSeverity.MEDIUM,
                            confidence=0.65,
                            reason="Device type mismatch between user agent and screen size",
                            metadata={"user_agent": user_agent, "screen_width": screen_width},
                        )
                    )

        return signals

    def _check_conversion_timing(self, conversion_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for suspicious conversion timing.
        """
        signals = []

        click_timestamp = conversion_data.get("click_timestamp")
        conversion_timestamp = conversion_data.get("conversion_timestamp") or datetime.utcnow()

        if click_timestamp:
            time_to_conversion = (conversion_timestamp - click_timestamp).total_seconds()

            # Too fast conversion is suspicious
            if time_to_conversion < self.config["min_time_to_conversion_seconds"]:
                signals.append(
                    FraudSignal(
                        fraud_type=FraudType.CONVERSION_FRAUD,
                        severity=FraudSeverity.HIGH,
                        confidence=0.9,
                        reason=f"Suspiciously fast conversion: {time_to_conversion:.1f}s",
                        metadata={"time_to_conversion": time_to_conversion},
                    )
                )

        return signals

    def _check_conversion_patterns(self, conversion_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for suspicious conversion patterns.
        """
        signals = []

        click_id = conversion_data.get("click_id")
        if not click_id:
            return signals

        from database import ConversionTracking as DbConversionTracking

        # Check for duplicate conversions from same click
        existing_conversions = (
            self.db.query(DbConversionTracking).filter(DbConversionTracking.click_id == click_id).count()
        )

        if existing_conversions > 1:
            signals.append(
                FraudSignal(
                    fraud_type=FraudType.CONVERSION_FRAUD,
                    severity=FraudSeverity.HIGH,
                    confidence=0.95,
                    reason=f"Multiple conversions ({existing_conversions}) from same click",
                    metadata={"click_id": click_id, "conversion_count": existing_conversions},
                )
            )

        return signals

    def _check_impression_velocity(self, impression_data: Dict[str, Any]) -> List[FraudSignal]:
        """
        Check for abnormal impression velocity.
        """
        signals = []

        # In a real implementation, this would check impression logs
        # For now, we'll return empty signals

        return signals

    def _calculate_risk_score(self, signals: List[FraudSignal]) -> float:
        """
        Calculate overall risk score based on detected signals.
        """
        if not signals:
            return 0.0

        # Weight signals by severity
        severity_weights = {
            FraudSeverity.LOW: 0.2,
            FraudSeverity.MEDIUM: 0.4,
            FraudSeverity.HIGH: 0.7,
            FraudSeverity.CRITICAL: 1.0,
        }

        weighted_scores = []
        for signal in signals:
            weight = severity_weights.get(signal.severity, 0.5)
            weighted_scores.append(signal.confidence * weight)

        # Use the maximum weighted score, with a bonus for multiple signals
        max_score = max(weighted_scores)
        multiple_signal_bonus = min(0.1 * (len(signals) - 1), 0.2)

        return min(max_score + multiple_signal_bonus, 1.0)

    def _get_recommended_action(self, risk_score: float, signals: List[FraudSignal]) -> str:
        """
        Get recommended action based on risk score and signals.
        """
        if risk_score >= self.config["high_risk_threshold"]:
            return "block_and_investigate"
        elif risk_score >= self.config["medium_risk_threshold"]:
            return "flag_for_review"
        elif risk_score >= self.config["low_risk_threshold"]:
            return "monitor"
        else:
            return "allow"

    def run_batch_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Run batch fraud analysis on recent events.
        Returns summary statistics.
        """
        from database import ClickTracking as DbClickTracking
        from database import FraudDetection as DbFraudDetection

        start_time = datetime.utcnow() - timedelta(hours=hours)

        # Analyze recent clicks
        recent_clicks = self.db.query(DbClickTracking).filter(DbClickTracking.timestamp >= start_time).all()

        fraud_summary = {
            "total_analyzed": len(recent_clicks),
            "fraudulent_count": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "fraud_types": defaultdict(int),
            "new_fraud_reports": 0,
        }

        for click in recent_clicks:
            result = self.analyze_click(
                {
                    "ad_id": click.ad_id,
                    "session_id": click.session_id,
                    "ip_hash": click.ip_hash,
                    "user_agent_hash": click.user_agent_hash,
                    "timestamp": click.timestamp,
                }
            )

            if result.is_fraudulent:
                fraud_summary["fraudulent_count"] += 1

                # Create fraud report
                for signal in result.signals:
                    fraud_report = DbFraudDetection(
                        event_id=click.id,
                        event_type="click",
                        reason=signal.reason,
                        severity=signal.severity.value,
                        review_status="pending",
                        ad_id=click.ad_id,
                    )
                    self.db.add(fraud_report)
                    fraud_summary["new_fraud_reports"] += 1
                    fraud_summary["fraud_types"][signal.fraud_type.value] += 1

            if result.risk_score >= self.config["high_risk_threshold"]:
                fraud_summary["high_risk_count"] += 1
            elif result.risk_score >= self.config["medium_risk_threshold"]:
                fraud_summary["medium_risk_count"] += 1
            elif result.risk_score >= self.config["low_risk_threshold"]:
                fraud_summary["low_risk_count"] += 1

        self.db.commit()
        fraud_summary["fraud_types"] = dict(fraud_summary["fraud_types"])

        return fraud_summary


def get_fraud_detector(db_session: Session) -> FraudDetector:
    """Factory function to get fraud detector instance"""
    return FraudDetector(db_session)
