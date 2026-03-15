"""
Intent Engine - Distributed Tracing Configuration

Uses OpenTelemetry for distributed tracing with Jaeger/Prometheus integration.

Features:
- Automatic span creation
- Context propagation
- Performance monitoring
- Error tracking
- Integration with FastAPI

Usage:
    setup_tracing()
    
    @traced("custom_operation")
    async def my_function():
        # Your code here
        pass
"""

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Global tracer reference
_tracer = None
_tracer_provider = None


def setup_tracing(service_name: str = "intent-engine") -> Optional[Any]:
    """
    Setup distributed tracing with OpenTelemetry.

    Supports multiple exporters:
    - Jaeger (for development)
    - OTLP (for production with Prometheus/Grafana)
    - Console (for debugging)

    Args:
        service_name: Service name for tracing

    Returns:
        TracerProvider if successful, None otherwise
    """
    global _tracer, _tracer_provider

    # Check if tracing is enabled
    tracing_enabled = os.getenv("TRACING_ENABLED", "true").lower() == "true"
    if not tracing_enabled:
        logger.info("Distributed tracing disabled")
        return None

    # Get exporter configuration
    exporter_type = os.getenv(
        "TRACING_EXPORTER", "console"
    ).lower()  # console, jaeger, otlp
    jaeger_endpoint = os.getenv(
        "JAEGER_ENDPOINT", "http://localhost:14268/api/traces"
    )
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        # Create resource with service metadata
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": os.getenv("VERSION", "1.0.0"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        exporter = None
        if exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                exporter = JaegerExporter(
                    agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
                    agent_port=int(os.getenv("JAEGER_PORT", "6831")),
                    endpoint=jaeger_endpoint,
                )
                logger.info(f"Jaeger exporter configured: {jaeger_endpoint}")

            except ImportError:
                logger.warning("Jaeger exporter not installed, falling back to console")
                exporter = ConsoleSpanExporter()

        elif exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                logger.info(f"OTLP exporter configured: {otlp_endpoint}")

            except ImportError:
                logger.warning("OTLP exporter not installed, falling back to console")
                exporter = ConsoleSpanExporter()

        else:
            exporter = ConsoleSpanExporter()
            logger.info("Using console exporter for tracing")

        # Add span processor
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            scheduled_delay_millis=5000,
            max_export_batch_size=512,
        )
        provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # Get tracer
        _tracer = trace.get_tracer(__name__)

        logger.info(
            f"Distributed tracing setup complete: service={service_name}, "
            f"exporter={exporter_type}"
        )

        return provider

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry not installed. Install with: "
            f"pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        return None


def get_tracer() -> Optional[Any]:
    """Get the current tracer instance"""
    return _tracer


def traced(span_name: Optional[str] = None):
    """
    Decorator to trace a function or method.

    Usage:
        @traced()
        async def my_function():
            pass

        @traced("custom_span_name")
        def sync_function():
            pass

    Args:
        span_name: Optional custom span name (defaults to function name)
    """

    def decorator(func: Callable) -> Callable:
        tracer = get_tracer()

        if not tracer:
            # Tracing not enabled, return original function
            return func

        span_name_final = span_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name_final) as span:
                # Add function attributes to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name_final) as span:
                # Add function attributes to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.record_exception(e)
                    raise

        # Detect if function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_context(span_name: str, attributes: Optional[dict[str, Any]] = None):
    """
    Context manager for tracing a code block.

    Usage:
        with trace_context("database_query", {"table": "users"}):
            # Your code here
            pass

    Args:
        span_name: Name of the span
        attributes: Optional attributes to add to span
    """
    tracer = get_tracer()

    if not tracer:
        # Tracing not enabled, yield without context
        yield
        return

    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.record_exception(e)
            raise


def trace_async_context(span_name: str, attributes: Optional[dict[str, Any]] = None):
    """
    Async context manager for tracing.

    Usage:
        async with trace_async_context("api_call"):
            # Your async code here
            pass
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def async_trace_context():
        tracer = get_tracer()

        if not tracer:
            yield
            return

        with tracer.start_as_current_span(span_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.record_exception(e)
                raise

    return async_trace_context()


def add_span_attribute(key: str, value: Any):
    """Add attribute to current active span"""
    tracer = get_tracer()
    if not tracer:
        return

    span = tracer.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def shutdown_tracing():
    """Shutdown tracing and flush pending spans"""
    global _tracer, _tracer_provider

    if _tracer_provider:
        try:
            # Force flush pending spans
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.info("Tracing spans flushed")

            # Shutdown provider
            _tracer_provider.shutdown()
            logger.info("Tracing shutdown complete")

        except Exception as e:
            logger.warning(f"Failed to shutdown tracing: {e}")

        finally:
            _tracer = None
            _tracer_provider = None


# FastAPI middleware integration
def setup_fastapi_tracing(app):
    """
    Setup tracing middleware for FastAPI application.

    Args:
        app: FastAPI application instance
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI tracing instrumentation enabled")

    except ImportError:
        logger.warning(
            "FastAPI instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-fastapi"
        )


# HTTP client instrumentation
def setup_httpx_tracing():
    """Setup tracing for httpx HTTP client"""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX client tracing enabled")

    except ImportError:
        logger.warning(
            "HTTPX instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-httpx"
        )


# SQLAlchemy instrumentation
def setup_sqlalchemy_tracing():
    """Setup tracing for SQLAlchemy database operations"""
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy tracing enabled")

    except ImportError:
        logger.warning(
            "SQLAlchemy instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-sqlalchemy"
        )


# Redis instrumentation
def setup_redis_tracing():
    """Setup tracing for Redis operations"""
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
        logger.info("Redis tracing enabled")

    except ImportError:
        logger.warning(
            "Redis instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-redis"
        )


def setup_all_instrumentation():
    """Setup all available instrumentation"""
    setup_httpx_tracing()
    setup_sqlalchemy_tracing()
    setup_redis_tracing()


# Lifespan integration for FastAPI
@contextmanager
def tracing_lifespan(app):
    """
    Lifespan context manager for tracing setup/cleanup.

    Usage with FastAPI:
        app = FastAPI(lifespan=tracing_lifespan)
    """
    # Startup
    setup_tracing()
    setup_all_instrumentation()
    yield
    # Shutdown
    shutdown_tracing()
