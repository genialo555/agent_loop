# Observability Implementation Guide

This guide provides step-by-step instructions for implementing the missing observability components identified in the review.

## 1. OpenTelemetry Tracing Implementation

### Step 1: Add Dependencies

```bash
# Add to requirements.txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
opentelemetry-instrumentation-httpx>=0.41b0
opentelemetry-instrumentation-logging>=0.41b0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-exporter-jaeger>=1.20.0
```

### Step 2: Create Tracing Configuration

```python
# core/telemetry.py
import os
from typing import Optional
from opentelemetry import trace, baggage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap
import structlog

logger = structlog.get_logger(__name__)

def configure_tracing(
    service_name: str = "gemma-agent-api",
    service_version: str = "1.0.0",
    environment: str = "production",
    otlp_endpoint: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None
) -> TracerProvider:
    """Configure OpenTelemetry tracing with OTLP or Jaeger exporter."""
    
    # Create resource attributes
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "service.environment": environment,
        "service.namespace": "agent-loop",
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure exporters
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use False in production with proper certs
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTLP tracing configured", endpoint=otlp_endpoint)
    
    if jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(":")[0],
            agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
            udp_split_oversized_batches=True,
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        logger.info("Jaeger tracing configured", endpoint=jaeger_endpoint)
    
    # Set tracer provider
    trace.set_tracer_provider(provider)
    
    # Set propagator for distributed tracing
    set_global_textmap(TraceContextTextMapPropagator())
    
    # Auto-instrument libraries
    FastAPIInstrumentor.instrument(tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    return provider

def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for manual instrumentation."""
    return trace.get_tracer(name)

def inject_trace_context(headers: dict) -> dict:
    """Inject trace context into outgoing request headers."""
    from opentelemetry.propagate import inject
    inject(headers)
    return headers

def extract_trace_context(headers: dict) -> dict:
    """Extract trace context from incoming request headers."""
    from opentelemetry.propagate import extract
    return extract(headers)
```

### Step 3: Integrate with FastAPI

```python
# inference/app.py - Add to lifespan function
from core.telemetry import configure_tracing

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager with telemetry."""
    # ... existing code ...
    
    # Configure tracing
    configure_tracing(
        service_name="gemma-agent-api",
        service_version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317"),
        jaeger_endpoint=os.getenv("JAEGER_ENDPOINT", "localhost:6831")
    )
    
    # ... rest of existing code ...
```

### Step 4: Add Manual Instrumentation

```python
# Example: inference/services/ollama.py
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class OllamaService:
    async def generate(self, prompt: str, model: str = "gemma:3n-e2b") -> str:
        """Generate text using Ollama with tracing."""
        with tracer.start_as_current_span(
            "ollama.generate",
            attributes={
                "model": model,
                "prompt.length": len(prompt),
                "service.name": "ollama"
            }
        ) as span:
            try:
                # Existing implementation
                response = await self._call_ollama(prompt, model)
                
                # Add response attributes
                span.set_attribute("response.length", len(response))
                span.set_status(Status(StatusCode.OK))
                
                return response
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

## 2. Sentry Error Tracking Implementation

### Step 1: Add Sentry SDK

```bash
# Add to requirements.txt
sentry-sdk[fastapi,httpx,sqlalchemy]>=1.40.0
```

### Step 2: Configure Sentry

```python
# core/error_tracking.py
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.excepthook import ExcepthookIntegration
import structlog

logger = structlog.get_logger(__name__)

def configure_sentry(
    dsn: Optional[str] = None,
    environment: str = "production",
    release: Optional[str] = None,
    sample_rate: float = 1.0,
    traces_sample_rate: float = 0.1
):
    """Configure Sentry error tracking and performance monitoring."""
    
    if not dsn:
        dsn = os.getenv("SENTRY_DSN")
    
    if not dsn:
        logger.warning("Sentry DSN not provided, error tracking disabled")
        return
    
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release or os.getenv("GIT_COMMIT_SHA", "unknown"),
        sample_rate=sample_rate,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=0.1,  # Enable profiling
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",
                failed_request_status_codes={403, 404, 429, 500, 502, 503, 504}
            ),
            HttpxIntegration(),
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            ),
            ExcepthookIntegration(always_run=True)
        ],
        before_send=before_send_filter,
        attach_stacktrace=True,
        send_default_pii=False,  # Don't send PII
        debug=environment == "development"
    )
    
    logger.info("Sentry configured", environment=environment, release=release)

def before_send_filter(event, hint):
    """Filter sensitive data before sending to Sentry."""
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        sensitive_headers = ["authorization", "cookie", "x-api-key"]
        for header in sensitive_headers:
            event["request"]["headers"].pop(header, None)
    
    # Remove sensitive data from extra context
    if "extra" in event:
        sensitive_keys = ["password", "token", "secret", "api_key"]
        for key in list(event["extra"].keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                event["extra"][key] = "[REDACTED]"
    
    return event

def capture_exception(error: Exception, **extra_context):
    """Manually capture an exception with additional context."""
    with sentry_sdk.push_scope() as scope:
        for key, value in extra_context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(error)

def add_user_context(user_id: str, email: Optional[str] = None, **extra):
    """Add user context to Sentry events."""
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        **extra
    })

def add_breadcrumb(message: str, category: str = "custom", level: str = "info", **data):
    """Add a breadcrumb for debugging context."""
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data
    )
```

### Step 3: Integrate with Application

```python
# inference/app.py
from core.error_tracking import configure_sentry

# In lifespan function
configure_sentry(
    environment=os.getenv("ENVIRONMENT", "production"),
    traces_sample_rate=0.1 if os.getenv("ENVIRONMENT") == "production" else 1.0
)
```

## 3. Enhanced Correlation ID Propagation

### Step 1: Create Correlation ID Middleware

```python
# inference/middleware/correlation.py
import uuid
from typing import Optional, Callable, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace, baggage
from opentelemetry.trace import Span
import structlog

logger = structlog.get_logger(__name__)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Enhanced correlation ID middleware with OpenTelemetry integration."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for existing correlation ID
        correlation_id = (
            request.headers.get("X-Correlation-ID") or
            request.headers.get("X-Request-ID") or
            str(uuid.uuid4())
        )
        
        # Store in request state
        request.state.correlation_id = correlation_id
        
        # Add to OpenTelemetry baggage for propagation
        baggage.set_baggage("correlation_id", correlation_id)
        
        # Add to current span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("correlation_id", correlation_id)
        
        # Configure structlog context
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            request_path=request.url.path,
            request_method=request.method
        )
        
        try:
            response = await call_next(request)
            
            # Add correlation ID to response
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        finally:
            # Clear context
            structlog.contextvars.clear_contextvars()

def get_correlation_id(request: Request) -> Optional[str]:
    """Get correlation ID from request."""
    return getattr(request.state, "correlation_id", None)

# Helper for propagating correlation ID to external services
class CorrelatedHTTPClient:
    """HTTP client that automatically propagates correlation IDs."""
    
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with correlation ID propagation."""
        headers = kwargs.get("headers", {})
        
        # Get correlation ID from context
        correlation_id = baggage.get_baggage("correlation_id")
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        
        # Inject OpenTelemetry trace context
        from core.telemetry import inject_trace_context
        headers = inject_trace_context(headers)
        
        kwargs["headers"] = headers
        
        return await self.client.request(method, url, **kwargs)
```

## 4. Log Aggregation with Loki

### Step 1: Add Loki to Docker Compose

```yaml
# docker-compose.yml
services:
  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./monitoring/loki:/etc/loki
      - loki_data:/loki
    networks:
      - agent-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]
      interval: 30s
      timeout: 10s
      retries: 3

  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - ./logs:/var/log/app:ro
      - ./monitoring/promtail:/etc/promtail
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - agent-network
    depends_on:
      - loki
```

### Step 2: Configure Promtail

```yaml
# monitoring/promtail/config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: fastapi
    static_configs:
      - targets:
          - localhost
        labels:
          job: fastapi
          __path__: /var/log/app/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            correlation_id: correlation_id
            timestamp: timestamp
      - labels:
          level:
          correlation_id:
      - timestamp:
          source: timestamp
          format: RFC3339
```

## 5. SLI/SLO Configuration

### Step 1: Define SLIs

```yaml
# monitoring/slo/slis.yaml
slis:
  - name: api_latency
    description: "API request latency"
    query: |
      histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le, handler)
      )
    unit: seconds
    
  - name: error_rate
    description: "API error rate"
    query: |
      sum(rate(http_requests_total{status=~"5.."}[5m])) /
      sum(rate(http_requests_total[5m]))
    unit: ratio
    
  - name: availability
    description: "Service availability"
    query: |
      1 - (sum(rate(http_requests_total{status=~"5.."}[5m])) /
           sum(rate(http_requests_total[5m])))
    unit: ratio
```

### Step 2: Create Prometheus Recording Rules

```yaml
# monitoring/prometheus/rules/slo.yml
groups:
  - name: slo_rules
    interval: 30s
    rules:
      - record: slo:api_latency_p95:5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          )
      
      - record: slo:error_rate:5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) /
          sum(rate(http_requests_total[5m]))
      
      - record: slo:availability:5m
        expr: |
          1 - (sum(rate(http_requests_total{status=~"5.."}[5m])) /
               sum(rate(http_requests_total[5m])))
      
      - alert: HighAPILatency
        expr: slo:api_latency_p95:5m > 0.5
        for: 5m
        labels:
          severity: warning
          slo: latency
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s (threshold: 0.5s)"
      
      - alert: HighErrorRate
        expr: slo:error_rate:5m > 0.01
        for: 5m
        labels:
          severity: critical
          slo: error_rate
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"
```

## 6. Monitoring Dashboard Updates

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Agent API Observability",
    "panels": [
      {
        "title": "Request Rate by Endpoint",
        "targets": [{
          "expr": "sum(rate(http_requests_total[5m])) by (handler)"
        }]
      },
      {
        "title": "Latency P95 by Endpoint",
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, handler))"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"
        }]
      },
      {
        "title": "Active Traces",
        "datasource": "Tempo",
        "targets": [{
          "query": "{ service.name='gemma-agent-api' }"
        }]
      }
    ]
  }
}
```

## Testing the Implementation

### 1. Verify Tracing

```bash
# Check if traces are being exported
curl http://localhost:16686/api/traces?service=gemma-agent-api
```

### 2. Verify Metrics

```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=http_requests_total
```

### 3. Verify Logs

```bash
# Check Loki logs
curl http://localhost:3100/loki/api/v1/query_range?query={job="fastapi"}
```

### 4. Test Correlation ID Propagation

```bash
# Make request with correlation ID
curl -H "X-Correlation-ID: test-123" http://localhost:8000/agents/run-agent
# Check response headers for X-Correlation-ID: test-123
```

## Performance Considerations

1. **Sampling**: Use head-based sampling for traces in production
2. **Batching**: Configure batch processors for efficient exporting
3. **Async**: Ensure all telemetry operations are non-blocking
4. **Resource Limits**: Set memory limits for telemetry collectors

## Security Best Practices

1. **Data Sanitization**: Always sanitize logs and traces for PII
2. **Access Control**: Secure telemetry endpoints with authentication
3. **Encryption**: Use TLS for telemetry data in transit
4. **Retention**: Configure appropriate data retention policies

This implementation guide provides a comprehensive approach to achieving production-grade observability for the Gemma-3N-Agent-Loop project.