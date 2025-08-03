"""Health check endpoints with comprehensive dependency verification."""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ..models.schemas import HealthResponse, ReadinessResponse
from ..services.dependencies import get_health_service
from ..services.health import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Lightweight health check for load balancers and basic monitoring"
)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
) -> HealthResponse:
    """
    Basic health check endpoint (fast, for load balancers).
    
    This endpoint performs minimal checks and should respond quickly.
    Use this for load balancer health checks and basic uptime monitoring.
    """
    health_data = await health_service.basic_health_check()
    return HealthResponse(**health_data)


@router.get(
    "/detailed",
    response_model=Dict[str, Any],
    summary="Detailed health check",
    description="Comprehensive health check with all dependencies and system metrics"
)
async def detailed_health_check(
    health_service: HealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    Detailed health check with comprehensive dependency verification.
    
    This endpoint checks:
    - HTTP client availability
    - Ollama service status and model availability
    - System resources (memory, disk)
    - All critical dependencies
    
    Use this for detailed monitoring and debugging.
    """
    return await health_service.comprehensive_health_check()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {"description": "Service not ready"}
    },
    summary="Readiness probe",
    description="Kubernetes readiness probe - checks if service is ready to receive traffic"
)
async def readiness_check(
    health_service: HealthService = Depends(get_health_service)
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes deployments.
    
    This endpoint verifies that the service is ready to handle requests:
    - All dependencies are available
    - Ollama model is loaded and responsive
    - Required external services are accessible
    
    Returns 503 if the service is not ready to receive traffic.
    """
    readiness_data = await health_service.readiness_check()
    
    if readiness_data["status"] != "ready":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    return ReadinessResponse(**readiness_data)


@router.get(
    "/live",
    response_model=Dict[str, Any],
    summary="Liveness probe",
    description="Kubernetes liveness probe - checks if the application process is alive"
)
async def liveness_check(
    health_service: HealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    Liveness probe for Kubernetes deployments.
    
    This endpoint only checks if the application process is alive and responsive.
    It should not check dependencies - only the application's core functionality.
    
    Use this to determine if the container should be restarted.
    """
    return await health_service.liveness_check()


@router.get(
    "/startup",
    response_model=Dict[str, Any],
    responses={
        503: {"description": "Application still starting up"}
    },
    summary="Startup probe",
    description="Kubernetes startup probe - checks if application has finished initializing"
)
async def startup_check(
    health_service: HealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    Startup probe for Kubernetes deployments.
    
    This endpoint checks if the application has finished its startup process:
    - All services are initialized
    - Critical dependencies are accessible
    - Application is ready to begin serving traffic
    
    Returns 503 if the application is still starting up.
    """
    startup_data = await health_service.startup_check()
    
    if startup_data["status"] != "started":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application still starting up"
        )
    
    return startup_data