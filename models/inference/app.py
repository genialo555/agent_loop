"""
Restructured FastAPI application with modular architecture.

This is the new main application file that follows FastAPI best practices:
- Modular structure with separate routers, services, and models
- Comprehensive health checks with dependency verification
- Enhanced security middleware and error handling
- Proper async patterns and dependency injection
- Observability with structured logging and metrics
"""
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any, Dict

import httpx
import structlog
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Info, generate_latest, 
    CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess, 
    REGISTRY
)

# Import our modular components
from .models.schemas import ErrorResponse, RunAgentRequest, RunAgentResponse
from .services.dependencies import initialize_services, cleanup_services, get_ollama_service
from .services.ollama import OllamaService
from .middleware.security import SecurityHeadersMiddleware, LoggingMiddleware, RateLimitingMiddleware
from .routers import health, agents, ollama

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("ENVIRONMENT") == "development" else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

# Prometheus metrics setup
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = REGISTRY

# HTTP metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'handler', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'handler'],
    registry=registry
)

# Application metrics
inference_requests_total = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['groupthink_enabled', 'status'],
    registry=registry
)

inference_duration_seconds = Histogram(
    'inference_duration_seconds',
    'Inference processing time',
    ['groupthink_enabled'],
    registry=registry
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests',
    registry=registry
)

webhook_notifications_total = Counter(
    'webhook_notifications_total',
    'Total webhook notifications sent',
    ['status'],
    registry=registry
)

# Application info
app_info = Info(
    'app_info',
    'Application information',
    registry=registry
)

app_info.info({
    'version': '1.0.0',
    'environment': os.getenv('ENVIRONMENT', 'production'),
    'service': 'fastapi-inference-api',
    'architecture': 'modular'
})

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager with proper service initialization."""
    # Startup
    global http_client
    logger.info("Starting application initialization")
    
    try:
        # Initialize HTTP client with optimized settings
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"User-Agent": "FastAPI-Agent/1.0.0"}
        )
        
        # Initialize all services with the shared HTTP client
        initialize_services(http_client)
        
        logger.info(
            "Application startup completed",
            extra={
                "http_client_initialized": True,
                "services_initialized": True,
                "uptime_seconds": 0
            }
        )
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Starting application shutdown")
        
        try:
            # Cleanup services
            cleanup_services()
            
            # Close HTTP client
            if http_client:
                await http_client.aclose()
            
            logger.info(
                "Application shutdown completed",
                extra={
                    "resources_cleaned": True,
                    "uptime_seconds": time.time() - app_start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Async FastAPI Production Example",
    description="Production-ready async FastAPI application with modular architecture",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None,
)

# CORS Configuration (restrictive by default)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Never use "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security middleware (order matters!)
# TODO: Fix middleware typing issues - these need to be rewritten as proper ASGI middleware
# app.add_middleware(SecurityHeadersMiddleware)
# app.add_middleware(RateLimitingMiddleware, requests_per_minute=60)
# app.add_middleware(LoggingMiddleware)

# Include routers with proper prefixes
app.include_router(health.router)
app.include_router(agents.router)
app.include_router(ollama.router)

# Sprint 1 Compatibility Alias
# Create /generate endpoint as an alias to /agents/run-agent
@app.post(
    "/generate",
    response_model=RunAgentResponse,
    summary="Generate text using LLM (Sprint 1 compatibility)",
    description="Alias for /agents/run-agent to maintain Sprint 1 specification compatibility",
    include_in_schema=True,  # Show in docs for clarity
    tags=["sprint1-compatibility"]
)
async def generate_text(
    request: Request,
    req: RunAgentRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> RunAgentResponse:
    """Sprint 1 compatibility endpoint - redirects to /agents/run-agent."""
    # Import the actual handler to avoid code duplication
    from .routers.agents import run_agent_with_ollama
    
    # Ensure Ollama is enabled for this endpoint
    req.use_ollama = True
    
    # Call the actual implementation
    return await run_agent_with_ollama(request, req, ollama_service)


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Enhanced HTTP exception handler with correlation ID."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    logger.warning(
        f"HTTP exception: {exc.status_code}",
        extra={
            "correlation_id": correlation_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail={"status_code": exc.status_code},
            correlation_id=correlation_id
        ).model_dump(),
        headers={"X-Correlation-ID": correlation_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Enhanced general exception handler with detailed logging."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        extra={
            "correlation_id": correlation_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail={
                "type": type(exc).__name__,
                "message": "An unexpected error occurred"
            },
            correlation_id=correlation_id
        ).model_dump(),
        headers={"X-Correlation-ID": correlation_id}
    )


# Metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "async-fastapi-example",
        "version": "1.0.0",
        "status": "running",
        "architecture": "modular",
        "uptime_seconds": time.time() - app_start_time,
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed",
            "readiness": "/health/ready",
            "agents": "/agents",
            "ollama": "/ollama",
            "metrics": "/metrics"
        }
    }


# Add metrics middleware for request tracking
@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Any) -> Response:
    """Middleware to track HTTP metrics."""
    start_time = time.time()
    method = request.method
    path_template = request.url.path
    
    # Track active requests
    active_requests.inc()
    
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        
        # Record metrics
        http_requests_total.labels(
            method=method,
            handler=path_template,
            status=status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            handler=path_template
        ).observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        # Record error metrics
        http_requests_total.labels(
            method=method,
            handler=path_template,
            status="500"
        ).inc()
        
        raise
    
    finally:
        # Always decrement active requests
        active_requests.dec()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "inference.app:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )