"""Security middleware for headers and basic protections."""
import time
import uuid
from typing import Callable, Dict, List, Tuple, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger(__name__)


class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable[..., Any]) -> Response:
        response: Response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class LoggingMiddleware:
    """Enhanced logging middleware with correlation IDs and metrics."""
    
    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """Add correlation IDs and log requests with enhanced context."""
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        try:
            response: Response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add correlation and timing headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log successful response
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "response_size": response.headers.get("content-length", "unknown")
                }
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time": process_time
                },
                exc_info=True
            )
            
            # Return proper error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "correlation_id": correlation_id
                },
                headers={"X-Correlation-ID": correlation_id}
            )


class RateLimitingMiddleware:
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app: Callable[..., Any], requests_per_minute: int = 60) -> None:
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, List[Tuple[float, int]]] = {}  # {ip: [(timestamp, count), ...]}
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self) -> None:
        """Remove old request timestamps to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 60  # Remove requests older than 1 minute
        
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                (timestamp, count) for timestamp, count in self.request_counts[ip]
                if timestamp > cutoff_time
            ]
            
            # Remove IPs with no recent requests
            if not self.request_counts[ip]:
                del self.request_counts[ip]
        
        self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Get requests from the last minute
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        recent_requests = [
            (timestamp, count) for timestamp, count in self.request_counts[client_ip]
            if timestamp > minute_ago
        ]
        
        # Count total requests in the last minute
        total_requests = sum(count for _, count in recent_requests)
        
        # Update the request count
        self.request_counts[client_ip] = recent_requests + [(current_time, 1)]
        
        return total_requests >= self.requests_per_minute
    
    async def __call__(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """Apply rate limiting based on client IP."""
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/health/live", "/metrics"]:
            response: Response = await call_next(request)
            return response
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(
                f"Rate limit exceeded for IP: {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute allowed"
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        final_response: Response = await call_next(request)
        return final_response