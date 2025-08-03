"""Comprehensive health check service with dependency verification."""
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import httpx
import structlog
from .ollama import OllamaService
from .external_api import ExternalAPIService

logger = structlog.get_logger(__name__)


class HealthService:
    """Service for comprehensive health and readiness checks."""
    
    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        ollama_service: Optional[OllamaService] = None,
        external_api_service: Optional[ExternalAPIService] = None,
        app_start_time: Optional[float] = None
    ):
        """Initialize health service.
        
        Args:
            http_client: HTTP client for connectivity checks
            ollama_service: Ollama service for LLM health checks
            external_api_service: External API service for dependency checks
            app_start_time: Application start time for uptime calculation
        """
        self.http_client = http_client
        self.ollama_service = ollama_service
        self.external_api_service = external_api_service
        self.app_start_time = app_start_time or time.time()
        self.service_info = {
            "name": "async-fastapi-example",
            "version": "1.0.0",
            "environment": "production"
        }
    
    async def basic_health_check(self) -> Dict[str, Any]:
        """Perform basic health check (fast, for load balancer).
        
        Returns:
            Basic health status
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "service": self.service_info["name"],
            "version": self.service_info["version"],
            "checks": {
                "basic": True
            },
            "uptime_seconds": time.time() - self.app_start_time
        }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check with all dependencies.
        
        Returns:
            Detailed health status with component checks
        """
        checks: Dict[str, Any] = {}
        overall_healthy = True
        
        # Basic component checks
        http_client_ok = self.http_client is not None
        checks["http_client"] = http_client_ok
        if not http_client_ok:
            overall_healthy = False
        
        # Ollama health check
        if self.ollama_service:
            try:
                ollama_health = await self.ollama_service.detailed_health_check()
                ollama_check = {
                    "available": ollama_health["available"],
                    "model_available": ollama_health.get("model_available", False),
                    "response_time_ms": ollama_health.get("response_time_ms"),
                    "error": ollama_health.get("error")
                }
                checks["ollama"] = ollama_check
                if not ollama_health["available"]:
                    overall_healthy = False
            except Exception as e:
                ollama_error_check = {
                    "available": False,
                    "error": str(e)
                }
                checks["ollama"] = ollama_error_check
                overall_healthy = False
        else:
            ollama_unavailable_check = {
                "available": False,
                "error": "Ollama service not initialized"
            }
            checks["ollama"] = ollama_unavailable_check
            overall_healthy = False
        
        # Memory and system checks (basic)
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_check: Dict[str, Any] = {
                "memory_available_mb": memory.available // (1024 * 1024),
                "memory_percent": memory.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024),
                "disk_percent": (disk.used / disk.total) * 100
            }
            
            # Consider system unhealthy if critical resources are low
            if memory.percent > 90 or (disk.used / disk.total) > 0.95:
                system_check["warning"] = "High resource usage"
            
            checks["system"] = system_check
                
        except ImportError:
            system_error_check = {
                "available": False,
                "error": "psutil not available"
            }
            checks["system"] = system_error_check
        except Exception as e:
            system_exception_check = {
                "available": False,
                "error": str(e)
            }
            checks["system"] = system_exception_check
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc),
            "service": self.service_info["name"],
            "version": self.service_info["version"],
            "checks": checks,
            "uptime_seconds": time.time() - self.app_start_time
        }
    
    async def readiness_check(self) -> Dict[str, Any]:
        """Perform readiness check for Kubernetes probes.
        
        Returns:
            Readiness status with dependency checks
        """
        checks: Dict[str, Any] = {}
        all_ready = True
        
        # HTTP client readiness
        http_client_ready = self.http_client is not None
        checks["http_client"] = http_client_ready
        if not http_client_ready:
            all_ready = False
        
        # Ollama readiness (must be available and have model loaded)
        if self.ollama_service:
            try:
                ollama_ready = await self.ollama_service.health_check()
                if ollama_ready:
                    # Additional check for model availability
                    model_info = await self.ollama_service.get_model_info()
                    ollama_readiness_check = {
                        "available": True,
                        "model_loaded": "error" not in model_info,
                        "model_name": self.ollama_service.model
                    }
                    if "error" in model_info:
                        all_ready = False
                        ollama_readiness_check["error"] = model_info["error"]
                    checks["ollama"] = ollama_readiness_check
                else:
                    ollama_not_ready_check = {
                        "available": False,
                        "model_loaded": False
                    }
                    checks["ollama"] = ollama_not_ready_check
                    all_ready = False
            except Exception as e:
                ollama_exception_check = {
                    "available": False,
                    "error": str(e)
                }
                checks["ollama"] = ollama_exception_check
                all_ready = False
        else:
            ollama_not_initialized_check = {
                "available": False,
                "error": "Service not initialized"
            }
            checks["ollama"] = ollama_not_initialized_check
            all_ready = False
        
        # External API dependencies (if configured)
        if self.external_api_service:
            # Add specific external API checks here
            # For now, just check if service is available
            external_api_check = {
                "service_available": True
            }
            checks["external_apis"] = external_api_check
        else:
            external_api_not_configured_check = {
                "service_available": False,
                "note": "External API service not configured"
            }
            checks["external_apis"] = external_api_not_configured_check
        
        return {
            "status": "ready" if all_ready else "not_ready",
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def liveness_check(self) -> Dict[str, Any]:
        """Perform liveness check for Kubernetes probes.
        
        This should only check if the application process is alive and responsive.
        
        Returns:
            Liveness status
        """
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.app_start_time
        }
    
    async def startup_check(self) -> Dict[str, Any]:
        """Perform startup probe check for Kubernetes.
        
        This checks if the application has finished initializing.
        
        Returns:
            Startup status
        """
        startup_checks: Dict[str, Any] = {}
        startup_complete = True
        
        # Check if critical services are initialized
        startup_checks["http_client"] = self.http_client is not None
        startup_checks["ollama_service"] = self.ollama_service is not None
        
        # If Ollama service exists, check if it's accessible
        if self.ollama_service:
            try:
                ollama_accessible = await asyncio.wait_for(
                    self.ollama_service.health_check(force=True),
                    timeout=10.0
                )
                startup_checks["ollama_accessible"] = ollama_accessible
                if not ollama_accessible:
                    startup_complete = False
            except asyncio.TimeoutError:
                startup_checks["ollama_accessible"] = False
                startup_checks["ollama_timeout"] = True
                startup_complete = False
            except Exception as e:
                startup_checks["ollama_accessible"] = False
                startup_checks["ollama_error"] = str(e)
                startup_complete = False
        
        return {
            "status": "started" if startup_complete else "starting",
            "checks": startup_checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.app_start_time
        }