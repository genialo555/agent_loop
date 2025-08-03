import pytest
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient
import jwt

from inference.api import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="module")
def token():
    # For testing purposes, create a simple token without encryption
    expiration = datetime.now(timezone.utc) + timedelta(hours=24)
    payload = {
        "sub": "dev-user",
        "exp": expiration.timestamp(),
    }
    # Simple test token - in production this should use proper JWT secrets
    return "test-token-for-development"


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert json_response["service"] == "async-fastapi-example"
    assert "timestamp" in json_response


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    response = await client.get("/ready")
    if response.status_code == 503:
        # Service not ready - expected when dependencies are not available
        json_response = response.json()
        assert "detail" in json_response
    else:
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "ready"
        assert "checks" in json_response


@pytest.mark.asyncio
async def test_run_agent(client, token):
    # Test the actual /run-agent endpoint without authentication for now
    payload = {"instruction": "Analyze this test prompt and provide insights"}
    response = await client.post("/run-agent", json=payload)
    
    json_response = response.json()
    
    if response.status_code == 503:
        # Service not ready (Ollama not available) - this is expected in test environment
        assert json_response["success"] is False
        assert "error" in json_response
    else:
        assert response.status_code == 200
        assert json_response["success"] is True
        assert "result" in json_response
        assert "execution_time_ms" in json_response


@pytest.mark.asyncio
async def test_run_agent_with_options(client):
    """Test run-agent endpoint with custom options."""
    payload = {
        "instruction": "Generate a creative story",
        "use_ollama": True,
        "temperature": 0.9,
        "max_tokens": 512,
        "system_prompt": "You are a creative storyteller."
    }
    response = await client.post("/run-agent", json=payload)
    json_response = response.json()
    
    # Should handle the request gracefully even if Ollama is not available
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert json_response["success"] is True
        assert "inference_metrics" in json_response
    else:
        assert json_response["success"] is False


@pytest.mark.asyncio
async def test_ollama_health_endpoint(client):
    """Test Ollama health check endpoint."""
    response = await client.get("/ollama/health")
    
    # Expected to fail in test environment without Ollama
    if response.status_code == 503:
        json_response = response.json()
        assert "detail" in json_response
        assert "not available" in json_response["detail"]
    else:
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "healthy"
        assert json_response["service"] == "ollama"


@pytest.mark.asyncio
async def test_ollama_model_info_endpoint(client):
    """Test Ollama model info endpoint."""
    response = await client.get("/ollama/model-info")
    
    # Expected behavior when Ollama is not available
    if response.status_code == 503:
        json_response = response.json()
        assert "detail" in json_response
    else:
        assert response.status_code == 200
        json_response = response.json()
        assert "success" in json_response
        assert "model_info" in json_response


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Should contain some basic metrics
    content = response.text
    assert len(content) > 0


@pytest.mark.asyncio
async def test_run_agent_validation(client):
    """Test input validation for run-agent endpoint."""
    # Test with empty instruction
    payload = {"instruction": ""}
    response = await client.post("/run-agent", json=payload)
    assert response.status_code == 422  # Validation error
    
    # Test with too long instruction
    payload = {"instruction": "x" * 3000}
    response = await client.post("/run-agent", json=payload)
    assert response.status_code == 422  # Validation error
    
    # Test with invalid temperature
    payload = {
        "instruction": "Test prompt",
        "temperature": 2.0  # Should be between 0.0 and 1.0
    }
    response = await client.post("/run-agent", json=payload)
    assert response.status_code == 422  # Validation error

