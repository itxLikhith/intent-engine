"""
Intent Engine - Kafka Event Streaming

Publishes and subscribes to intent/search events for real-time analytics.

Topics:
- intents.extracted: Raw intent extractions
- intents.processed: Processed intents with enrichment
- searches.executed: Search queries
- results.served: Results returned to users
- clicks.recorded: User clicks (privacy-preserving)

Features:
- Async event publishing
- Event serialization
- Batch publishing
- Graceful degradation (works without Kafka)
"""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class KafkaEventPublisher:
    """
    Publishes events to Kafka topics.

    Usage:
        publisher = KafkaEventPublisher()
        publisher.publish_intent_extracted(intent_data)
        publisher.publish_search_executed(search_data)

    Topics:
    - intents.extracted: Intent extraction events
    - searches.executed: Search query events
    - results.served: Result serving events
    - clicks.recorded: Click tracking events
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.bootstrap_servers = self.config.get(
            "bootstrap_servers",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        )
        self._producer = None
        self._enabled = self.config.get("enabled", True)

        # Event buffers for batch publishing
        self._event_buffer = []
        self._buffer_size = self.config.get("buffer_size", 100)

        logger.info(f"KafkaEventPublisher initialized: servers={self.bootstrap_servers}, enabled={self._enabled}")

    def _get_producer(self) -> Any | None:
        """Lazy Kafka producer initialization"""
        if self._producer is None:
            try:
                from kafka import KafkaProducer

                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    compression_type="gzip",
                    acks=1,  # Wait for leader acknowledgment
                    retries=3,
                    retry_backoff_ms=100,
                )

                logger.info("Kafka producer initialized")

            except ImportError:
                logger.warning("kafka-python not installed. Install with: pip install kafka-python")
                self._enabled = False
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Kafka producer: {e}")
                self._enabled = False
                return None

        return self._producer

    def publish_intent_extracted(self, intent_data: dict[str, Any]):
        """
        Publish intent extraction event.

        Args:
            intent_data: Intent extraction data (UniversalIntent dict)
        """
        if not self._enabled:
            return

        event = {
            "event_type": "intent.extracted",
            "timestamp": self._get_timestamp(),
            "data": intent_data,
        }

        self._publish("intents.extracted", event)

    def publish_search_executed(self, search_data: dict[str, Any]):
        """
        Publish search execution event.

        Args:
            search_data: Search execution data (query, results_count, latency, etc.)
        """
        if not self._enabled:
            return

        event = {
            "event_type": "search.executed",
            "timestamp": self._get_timestamp(),
            "data": search_data,
        }

        self._publish("searches.executed", event)

    def publish_results_served(self, results_data: dict[str, Any]):
        """
        Publish results served event.

        Args:
            results_data: Results data (query, result_count, backend_distribution, etc.)
        """
        if not self._enabled:
            return

        event = {
            "event_type": "results.served",
            "timestamp": self._get_timestamp(),
            "data": results_data,
        }

        self._publish("results.served", event)

    def publish_click_recorded(self, click_data: dict[str, Any]):
        """
        Publish click event (privacy-preserving).

        Args:
            click_data: Click data (session_id, result_url, dwell_time, etc.)
        """
        if not self._enabled:
            return

        # Ensure no PII is included
        sanitized_data = self._sanitize_click_data(click_data)

        event = {
            "event_type": "click.recorded",
            "timestamp": self._get_timestamp(),
            "data": sanitized_data,
        }

        self._publish("clicks.recorded", event)

    def _publish(self, topic: str, event: dict[str, Any]):
        """
        Publish event to Kafka topic.

        Args:
            topic: Kafka topic name
            event: Event data
        """
        producer = self._get_producer()
        if not producer:
            # Store in buffer for later (if needed)
            self._event_buffer.append((topic, event))
            if len(self._event_buffer) >= self._buffer_size:
                self._flush_buffer()
            return

        try:
            producer.send(topic, value=event)
            # Don't wait for confirmation (async)
            # In production, you might want to add a callback
        except Exception as e:
            logger.warning(f"Failed to publish event to {topic}: {e}")
            # Store in buffer
            self._event_buffer.append((topic, event))

    def _sanitize_click_data(self, click_data: dict[str, Any]) -> dict[str, Any]:
        """
        Remove PII from click data for privacy compliance.

        Args:
            click_data: Raw click data

        Returns:
            Sanitized click data
        """
        # Fields to remove
        pii_fields = {
            "ip_address",
            "user_agent",
            "user_id",
            "email",
            "name",
            "location",
            "device_id",
        }

        sanitized = {}
        for key, value in click_data.items():
            if key.lower() not in pii_fields:
                sanitized[key] = value
            else:
                # Hash sensitive fields if needed
                if key == "session_id":
                    # Keep session_id but ensure it's anonymized
                    sanitized[key] = self._hash_value(str(value))

        return sanitized

    def _hash_value(self, value: str) -> str:
        """Hash a value for privacy"""
        import hashlib

        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp"""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    def flush(self):
        """Flush pending messages"""
        if self._producer:
            try:
                self._producer.flush(timeout=10)
                logger.debug("Kafka producer flushed")
            except Exception as e:
                logger.warning(f"Failed to flush Kafka producer: {e}")

        # Also flush buffer
        self._flush_buffer()

    def _flush_buffer(self):
        """Flush event buffer"""
        if not self._event_buffer:
            return

        producer = self._get_producer()
        if not producer:
            logger.warning(f"Buffer has {len(self._event_buffer)} events, but no producer")
            return

        try:
            for topic, event in self._event_buffer:
                producer.send(topic, value=event)
            producer.flush(timeout=10)
            logger.info(f"Flushed {len(self._event_buffer)} buffered events")
            self._event_buffer = []
        except Exception as e:
            logger.warning(f"Failed to flush buffer: {e}")

    def close(self):
        """Close producer and flush pending messages"""
        self.flush()
        if self._producer:
            try:
                self._producer.close(timeout=10)
                logger.info("Kafka producer closed")
            except Exception as e:
                logger.warning(f"Failed to close Kafka producer: {e}")


class KafkaEventSubscriber:
    """
    Subscribes to Kafka topics for real-time processing.

    Usage:
        subscriber = KafkaEventSubscriber()
        subscriber.subscribe(['intents.extracted', 'searches.executed'], callback)

    Callback signature:
        def callback(topic: str, message: dict):
            # Process message
            pass
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.bootstrap_servers = self.config.get(
            "bootstrap_servers",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        )
        self.group_id = self.config.get("group_id", "intent-engine-workers")
        self._consumer = None
        self._running = False

        logger.info(f"KafkaEventSubscriber initialized: servers={self.bootstrap_servers}, group={self.group_id}")

    def subscribe(self, topics: list[str], callback: Callable[[str, dict], None]):
        """
        Subscribe to topics and process messages.

        Args:
            topics: List of Kafka topics
            callback: Function to call for each message
        """
        try:
            from kafka import KafkaConsumer

            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id=self.group_id,
                consumer_timeout_ms=1000,  # For graceful shutdown
            )

            logger.info(f"Subscribed to topics: {topics}")

            self._running = True
            self._consumer = consumer

            for message in consumer:
                if not self._running:
                    break

                try:
                    callback(message.topic, message.value)
                except Exception as e:
                    logger.error(f"Error processing message from {message.topic}: {e}")

        except ImportError:
            logger.warning("kafka-python not installed")
        except Exception as e:
            logger.error(f"Failed to subscribe to Kafka: {e}")

    def stop(self):
        """Stop subscription"""
        self._running = False
        if self._consumer:
            try:
                self._consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.warning(f"Failed to close Kafka consumer: {e}")


# Singleton instances
_event_publisher: KafkaEventPublisher | None = None
_event_subscriber: KafkaEventSubscriber | None = None


def get_event_publisher(config: dict[str, Any] | None = None) -> KafkaEventPublisher:
    """Get event publisher singleton"""
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = KafkaEventPublisher(config)
    return _event_publisher


def get_event_subscriber(config: dict[str, Any] | None = None) -> KafkaEventSubscriber:
    """Get event subscriber singleton"""
    global _event_subscriber
    if _event_subscriber is None:
        _event_subscriber = KafkaEventSubscriber(config)
    return _event_subscriber


def publish_event(event_type: str, data: dict[str, Any]):
    """
    Convenience function to publish events.

    Args:
        event_type: Type of event (intent.extracted, search.executed, etc.)
        data: Event data
    """
    publisher = get_event_publisher()

    if event_type == "intent.extracted":
        publisher.publish_intent_extracted(data)
    elif event_type == "search.executed":
        publisher.publish_search_executed(data)
    elif event_type == "results.served":
        publisher.publish_results_served(data)
    elif event_type == "click.recorded":
        publisher.publish_click_recorded(data)
    else:
        logger.warning(f"Unknown event type: {event_type}")
