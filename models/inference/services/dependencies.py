"""Dependency injection factories for FastAPI services."""
from typing import Optional
import time
import httpx
from .ollama import OllamaService
from .external_api import ExternalAPIService
from .health import HealthService
from .training import TrainingService

# Global variables to store shared services
_http_client: Optional[httpx.AsyncClient] = None
_ollama_service: Optional[OllamaService] = None
_external_api_service: Optional[ExternalAPIService] = None
_health_service: Optional[HealthService] = None
_training_service: Optional[TrainingService] = None
_app_start_time: float = time.time()


async def get_http_client() -> Optional[httpx.AsyncClient]:
    """Get the shared HTTP client instance."""
    return _http_client


async def get_ollama_service() -> OllamaService:
    """Factory to get the Ollama service instance."""
    if _ollama_service is None:
        raise RuntimeError("Ollama service not initialized")
    return _ollama_service


async def get_external_api_service() -> ExternalAPIService:
    """Factory to get the External API service instance."""
    if _external_api_service is None:
        raise RuntimeError("External API service not initialized")
    return _external_api_service


async def get_health_service() -> HealthService:
    """Factory to get the Health service instance."""
    if _health_service is None:
        raise RuntimeError("Health service not initialized")
    return _health_service


async def get_training_service() -> TrainingService:
    """Factory to get the Training service instance."""
    if _training_service is None:
        raise RuntimeError("Training service not initialized")
    return _training_service


def initialize_services(http_client: httpx.AsyncClient) -> None:
    """Initialize all services with shared dependencies.
    
    Args:
        http_client: Shared HTTP client for all services
    """
    global _http_client, _ollama_service, _external_api_service, _health_service, _training_service, _app_start_time
    
    _http_client = http_client
    _ollama_service = OllamaService(http_client=http_client)
    _external_api_service = ExternalAPIService(http_client=http_client)
    _training_service = TrainingService(http_client=http_client)
    _health_service = HealthService(
        http_client=http_client,
        ollama_service=_ollama_service,
        external_api_service=_external_api_service,
        app_start_time=_app_start_time
    )


def cleanup_services() -> None:
    """Cleanup service instances (called during app shutdown)."""
    global _http_client, _ollama_service, _external_api_service, _health_service, _training_service
    
    _http_client = None
    _ollama_service = None
    _external_api_service = None
    _health_service = None
    _training_service = None