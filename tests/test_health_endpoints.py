"""Comprehensive tests for health check endpoints."""
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient
from fastapi import status

from agent_loop.models.inference.services.health import HealthService


# ============================================================================
# Unit Tests for /health endpoint
# ============================================================================

@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_success(test_client: AsyncClient):
    """Test basic health endpoint returns 200 OK with correct structure."""
    start_time = time.time()
    
    response = await test_client.get("/health")
    
    # Assert response code and timing
    assert response.status_code == status.HTTP_200_OK
    assert (time.time() - start_time) < 1.0  # Should be fast
    
    json_response = response.json()
    
    # Validate response structure
    required_fields = ["status", "timestamp", "service", "version", "checks", "uptime_seconds"]
    for field in required_fields:
        assert field in json_response, f"Missing required field: {field}"
    
    # Validate field values
    assert json_response["status"] == "healthy"
    assert json_response["service"] == "async-fastapi-example"
    assert json_response["version"] == "1.0.0"
    assert isinstance(json_response["checks"], dict)
    assert isinstance(json_response["uptime_seconds"], (int, float))
    assert json_response["uptime_seconds"] >= 0
    
    # Validate timestamp format
    timestamp = json_response["timestamp"]
    assert isinstance(timestamp, str)
    # Should be parseable as ISO format
    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))


@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_response_time(test_client: AsyncClient, assert_response_time):
    """Test health endpoint responds within acceptable time limits."""
    start_time = time.time()
    
    response = await test_client.get("/health")
    
    assert response.status_code == status.HTTP_200_OK
    assert_response_time(start_time, max_time_ms=500.0)  # Health should be very fast


@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_concurrent_requests(test_client: AsyncClient):
    """Test health endpoint handles concurrent requests correctly."""
    # Send multiple concurrent requests
    tasks = []
    for _ in range(10):
        task = asyncio.create_task(test_client.get("/health"))
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    # All requests should succeed
    for response in responses:
        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        assert json_response["status"] == "healthy"


@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_consistent_uptime(test_client: AsyncClient):
    """Test health endpoint reports consistent uptime across requests."""
    # First request
    response1 = await test_client.get("/health")
    uptime1 = response1.json()["uptime_seconds"]
    
    # Wait a bit
    await asyncio.sleep(0.1)
    
    # Second request
    response2 = await test_client.get("/health")
    uptime2 = response2.json()["uptime_seconds"]
    
    # Uptime should have increased
    assert uptime2 > uptime1
    assert (uptime2 - uptime1) >= 0.1  # Should reflect elapsed time


# ============================================================================
# Unit Tests for Health Service
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_basic_check(health_service: HealthService):
    """Test HealthService basic_health_check method."""
    result = await health_service.basic_health_check()
    
    # Validate structure
    required_fields = ["status", "timestamp", "service", "version", "checks", "uptime_seconds"]
    for field in required_fields:
        assert field in result
    
    assert result["status"] == "healthy"
    assert result["service"] == "async-fastapi-example"
    assert result["version"] == "1.0.0"
    assert "basic" in result["checks"]
    assert result["checks"]["basic"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_comprehensive_check_healthy(health_service: HealthService):
    """Test HealthService comprehensive_health_check with healthy dependencies."""
    result = await health_service.comprehensive_health_check()
    
    # Should be healthy with all services mocked as available
    assert result["status"] == "healthy"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["http_client"] is True
    assert checks["ollama"]["available"] is True
    assert checks["ollama"]["model_available"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_comprehensive_check_unhealthy(health_service_unhealthy: HealthService):
    """Test HealthService comprehensive_health_check with unhealthy dependencies."""
    result = await health_service_unhealthy.comprehensive_health_check()
    
    # Should be unhealthy with some services unavailable
    assert result["status"] == "unhealthy"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["http_client"] is False  # No HTTP client
    assert checks["ollama"]["available"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_readiness_check_ready(health_service: HealthService):
    """Test HealthService readiness_check when all services are ready."""
    result = await health_service.readiness_check()
    
    assert result["status"] == "ready"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["http_client"] is True
    assert checks["ollama"]["available"] is True
    assert checks["ollama"]["model_loaded"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_readiness_check_not_ready(health_service_unhealthy: HealthService):
    """Test HealthService readiness_check when services are not ready."""
    result = await health_service_unhealthy.readiness_check()
    
    assert result["status"] == "not_ready"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["http_client"] is False
    assert checks["ollama"]["available"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_liveness_check(health_service: HealthService):
    """Test HealthService liveness_check always succeeds."""
    result = await health_service.liveness_check()
    
    assert result["status"] == "alive"
    assert "timestamp" in result
    assert "uptime_seconds" in result
    assert result["uptime_seconds"] >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_startup_check_started(health_service: HealthService):
    """Test HealthService startup_check when startup is complete."""
    result = await health_service.startup_check()
    
    assert result["status"] == "started"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["http_client"] is True
    assert checks["ollama_service"] is True
    assert checks["ollama_accessible"] is True


@pytest.mark.unit
@pytest.mark.asyncio
@patch('asyncio.wait_for')
async def test_health_service_startup_check_timeout(mock_wait_for, health_service: HealthService):
    """Test HealthService startup_check handles timeout correctly."""
    # Mock timeout exception
    mock_wait_for.side_effect = asyncio.TimeoutError()
    
    result = await health_service.startup_check()
    
    assert result["status"] == "starting"
    assert "checks" in result
    
    checks = result["checks"]
    assert checks["ollama_accessible"] is False
    assert checks["ollama_timeout"] is True


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_handles_service_errors(test_client: AsyncClient):
    """Test health endpoint gracefully handles service errors."""
    with patch('inference.services.dependencies.get_health_service') as mock_get_service:
        # Mock service that raises exception
        mock_service = AsyncMock()
        mock_service.basic_health_check.side_effect = Exception("Service error")
        mock_get_service.return_value = mock_service
        
        response = await test_client.get("/health")
        
        # Should still return some response, not crash
        assert response.status_code in [200, 500, 503]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_ollama_exception_handling(health_service: HealthService):
    """Test HealthService handles Ollama service exceptions."""
    # Configure mock to raise exception
    health_service.ollama_service.detailed_health_check.side_effect = Exception("Connection error")
    
    result = await health_service.comprehensive_health_check()
    
    # Should handle exception gracefully
    assert result["status"] == "unhealthy"
    assert "ollama" in result["checks"]
    assert result["checks"]["ollama"]["available"] is False
    assert "error" in result["checks"]["ollama"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_service_system_check_missing_psutil():
    """Test HealthService handles missing psutil gracefully."""
    # Create service without system dependencies
    service = HealthService(
        http_client=AsyncMock(),
        ollama_service=None,
        external_api_service=None
    )
    
    with patch('psutil.virtual_memory', side_effect=ImportError("No module named 'psutil'")):
        result = await service.comprehensive_health_check()
        
        # Should handle missing dependency gracefully
        assert "system" in result["checks"]
        assert result["checks"]["system"]["available"] is False
        assert "psutil not available" in result["checks"]["system"]["error"]


# ============================================================================
# Performance Tests  
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_performance(test_client: AsyncClient, benchmark_config: Dict[str, Any]):
    """Benchmark health endpoint performance."""
    async def make_health_request():
        response = await test_client.get("/health")
        assert response.status_code == 200
        return response.json()
    
    # Warmup
    for _ in range(benchmark_config["warmup_rounds"]):
        await make_health_request()
    
    # Benchmark
    start_time = time.time()
    results = []
    
    for _ in range(benchmark_config["rounds"]):
        request_start = time.time()
        await make_health_request()
        request_time = (time.time() - request_start) * 1000
        results.append(request_time)
    
    total_time = time.time() - start_time
    avg_response_time = sum(results) / len(results)
    max_response_time = max(results)
    
    # Performance assertions
    assert avg_response_time < benchmark_config["max_response_time_ms"]
    assert max_response_time < benchmark_config["max_response_time_ms"] * 2
    assert total_time < benchmark_config["timeout_seconds"]
    
    print(f"\nHealth endpoint performance:")
    print(f"  Average response time: {avg_response_time:.2f}ms")
    print(f"  Max response time: {max_response_time:.2f}ms")
    print(f"  Total requests: {benchmark_config['rounds']}")
    print(f"  Total time: {total_time:.2f}s")


@pytest.mark.benchmark
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_concurrent_performance(
    test_client: AsyncClient, 
    benchmark_config: Dict[str, Any]
):
    """Test health endpoint performance under concurrent load."""
    async def concurrent_requests():
        tasks = []
        for _ in range(benchmark_config["concurrent_requests"]):
            task = asyncio.create_task(test_client.get("/health"))
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed_time = (time.time() - start_time) * 1000
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        return elapsed_time
    
    # Warmup
    await concurrent_requests()
    
    # Benchmark multiple rounds of concurrent requests
    times = []
    for _ in range(10):
        elapsed = await concurrent_requests()
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    # Performance assertions
    assert avg_time < benchmark_config["max_response_time_ms"] * 2
    assert max_time < benchmark_config["max_response_time_ms"] * 3
    
    print(f"\nConcurrent health endpoint performance:")
    print(f"  Average time for {benchmark_config['concurrent_requests']} concurrent requests: {avg_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")


# ============================================================================
# Schema Validation Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_response_schema_validation(
    test_client: AsyncClient, 
    assert_json_schema
):
    """Test health endpoint response matches expected schema."""
    response = await test_client.get("/health")
    assert response.status_code == 200
    
    json_response = response.json()
    required_fields = [
        "status", "timestamp", "service", "version", 
        "checks", "uptime_seconds"
    ]
    
    assert_json_schema(json_response, required_fields)
    
    # Additional schema validation
    assert isinstance(json_response["status"], str)
    assert json_response["status"] in ["healthy", "unhealthy"]
    assert isinstance(json_response["service"], str)
    assert isinstance(json_response["version"], str)
    assert isinstance(json_response["checks"], dict)
    assert isinstance(json_response["uptime_seconds"], (int, float))


@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_content_type(test_client: AsyncClient):
    """Test health endpoint returns correct content type."""
    response = await test_client.get("/health")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


@pytest.mark.unit
@pytest.mark.health
@pytest.mark.asyncio
async def test_health_endpoint_cache_headers(test_client: AsyncClient):
    """Test health endpoint includes appropriate cache headers."""
    response = await test_client.get("/health")
    
    assert response.status_code == 200
    # Health checks should not be cached
    assert "cache-control" in response.headers or "Cache-Control" in response.headers