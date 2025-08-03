"""Pytest configuration and fixtures for the agent loop project.

This module provides shared fixtures and configuration for testing the
modernized agent loop with proper type safety and async patterns.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock
import pytest
import httpx
from httpx import AsyncClient

# Add parent directory to path for imports  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_loop.models.inference.api import app
from agent_loop.models.inference.services.health import HealthService
from agent_loop.models.inference.services.ollama import OllamaService
from agent_loop.models.inference.services.external_api import ExternalAPIService


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", 
        "property: marks tests as property-based tests using hypothesis"
    )
    config.addinivalue_line(
        "markers", 
        "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", 
        "ollama: marks tests that require Ollama service"
    )
    config.addinivalue_line(
        "markers", 
        "external: marks tests that require external services"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Modify test items during collection."""
    # Auto-mark tests based on their path/name
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in item.nodeid or "/test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark Ollama-dependent tests
        if "ollama" in item.nodeid.lower():
            item.add_marker(pytest.mark.ollama)
        
        # Mark slow tests based on timeout or name patterns
        if any(pattern in item.nodeid.lower() for pattern in ['benchmark', 'performance', 'load']):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture(scope="session")
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client for the FastAPI application."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a fresh HTTP client for each test."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture(scope="function")
def mock_http_client() -> AsyncMock:
    """Create a mock HTTP client for testing without network calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    
    # Configure common mock responses
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"status": "ok"}
    
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"success": True}
    
    return client


# ============================================================================
# Service Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_ollama_service() -> OllamaService:
    """Create a mock Ollama service for testing."""
    service = AsyncMock(spec=OllamaService)
    
    # Configure healthy responses by default
    service.health_check = AsyncMock(return_value=True)
    service.detailed_health_check = AsyncMock(return_value={
        "available": True,
        "model_available": True,
        "response_time_ms": 150.0,
        "model": "gemma:3n-e2b"
    })
    service.get_model_info = AsyncMock(return_value={
        "name": "gemma:3n-e2b",
        "size": 2345678901,
        "modified_at": "2025-01-28T10:30:00Z",
        "digest": "sha256:abc123...",
        "details": {"format": "gguf"}
    })
    service.model = "gemma:3n-e2b"
    
    return service


@pytest.fixture(scope="function")
def mock_ollama_service_unavailable() -> OllamaService:
    """Create a mock Ollama service that simulates service unavailable."""
    service = AsyncMock(spec=OllamaService)
    
    # Configure unavailable responses
    service.health_check = AsyncMock(return_value=False)
    service.detailed_health_check = AsyncMock(return_value={
        "available": False,
        "error": "Ollama service not available"
    })
    service.get_model_info = AsyncMock(return_value={
        "error": "Service unavailable"
    })
    service.model = "gemma:3n-e2b"
    
    return service


@pytest.fixture(scope="function")
def mock_external_api_service() -> ExternalAPIService:
    """Create a mock external API service for testing."""
    service = AsyncMock(spec=ExternalAPIService)
    
    # Configure default successful responses
    service.is_available = AsyncMock(return_value=True)
    service.check_dependencies = AsyncMock(return_value={
        "database": True,
        "cache": True,
        "external_apis": True
    })
    
    return service


@pytest.fixture(scope="function")
def health_service(
    mock_http_client: AsyncMock,
    mock_ollama_service: OllamaService,
    mock_external_api_service: ExternalAPIService
) -> HealthService:
    """Create a HealthService instance with mocked dependencies."""
    return HealthService(
        http_client=mock_http_client,
        ollama_service=mock_ollama_service,
        external_api_service=mock_external_api_service,
        app_start_time=time.time() - 100  # Simulate 100 seconds uptime
    )


@pytest.fixture(scope="function")
def health_service_unhealthy(
    mock_http_client: AsyncMock,
    mock_ollama_service_unavailable: OllamaService,
    mock_external_api_service: ExternalAPIService
) -> HealthService:
    """Create a HealthService instance with unhealthy dependencies."""
    # Configure external service as unavailable
    mock_external_api_service.is_available = AsyncMock(return_value=False)
    mock_external_api_service.check_dependencies = AsyncMock(return_value={
        "database": False,
        "cache": True,
        "external_apis": False
    })
    
    return HealthService(
        http_client=None,  # Simulate no HTTP client
        ollama_service=mock_ollama_service_unavailable,
        external_api_service=mock_external_api_service,
        app_start_time=time.time() - 50
    )


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def sample_health_response() -> Dict[str, Any]:
    """Sample health response data for testing."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
        "service": "async-fastapi-example",
        "version": "1.0.0",
        "checks": {
            "http_client": True,
            "ollama": True,
            "system": True
        },
        "uptime_seconds": 150.5
    }


@pytest.fixture(scope="function")
def sample_readiness_response() -> Dict[str, Any]:
    """Sample readiness response data for testing."""
    return {
        "status": "ready",
        "checks": {
            "http_client": True,
            "ollama": {
                "available": True,
                "model_loaded": True,
                "model_name": "gemma:3n-e2b"
            },
            "external_apis": {
                "service_available": True
            }
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture(scope="function")
def sample_ollama_model_info() -> Dict[str, Any]:
    """Sample Ollama model info for testing."""
    return {
        "success": True,
        "model_info": {
            "name": "gemma:3n-e2b",
            "size": 2345678901,
            "modified_at": "2025-01-28T10:30:00Z",
            "digest": "sha256:abc123...",
            "details": {"format": "gguf"}
        },
        "timestamp": "2025-01-28T10:30:00Z"
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "rounds": 100,
        "warmup_rounds": 10,
        "timeout_seconds": 30,
        "max_response_time_ms": 1000,
        "concurrent_requests": 10
    }


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture(scope="function")
def assert_response_time():
    """Utility fixture to assert response times."""
    def _assert_response_time(start_time: float, max_time_ms: float = 1000.0):
        """Assert that response time is within acceptable limits."""
        elapsed_ms = (time.time() - start_time) * 1000
        assert elapsed_ms <= max_time_ms, f"Response time {elapsed_ms:.2f}ms exceeds limit {max_time_ms}ms"
    
    return _assert_response_time


@pytest.fixture(scope="function")
def assert_json_schema():
    """Utility fixture to validate JSON schema."""
    def _assert_json_schema(data: Dict[str, Any], required_fields: list):
        """Assert that JSON data contains required fields."""
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"
        
        # Check for timestamp format if present
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if isinstance(timestamp, str):
                # Should be ISO format
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return _assert_json_schema


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def requires_ollama():
    """Skip test if Ollama is not available."""
    pytest.importorskip("ollama", reason="Ollama not available")


@pytest.fixture(scope="function")
async def wait_for_service():
    """Utility to wait for services to be ready."""
    async def _wait_for_service(
        url: str,
        timeout: int = 30,
        check_interval: float = 1.0
    ) -> bool:
        """Wait for a service to become available."""
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url, timeout=5.0)
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
                
                await asyncio.sleep(check_interval)
        
        return False
    
    return _wait_for_service


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    # Cleanup code here if needed
    await asyncio.sleep(0)  # Give time for any pending async operations