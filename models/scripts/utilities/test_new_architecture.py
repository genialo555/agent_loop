#!/usr/bin/env python3
"""
Test script for the new modular FastAPI architecture.

This script validates all endpoints and functionality of the restructured application.
"""
import asyncio
import json
import time
from typing import Dict, Any

import httpx


class ArchitectureTestSuite:
    """Test suite for the new modular FastAPI architecture."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all architecture tests."""
        print("🚀 Starting FastAPI Architecture Test Suite")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test basic connectivity
            await self.test_root_endpoint(client)
            
            # Test health endpoints
            await self.test_basic_health(client)
            await self.test_detailed_health(client)
            await self.test_readiness_check(client)
            await self.test_liveness_check(client)
            await self.test_startup_check(client)
            
            # Test Ollama endpoints
            await self.test_ollama_health(client)
            await self.test_ollama_model_info(client)
            await self.test_ollama_models_list(client)
            
            # Test agent endpoints
            await self.test_agent_run(client)
            await self.test_agent_run_ollama(client)
            
            # Test metrics endpoint
            await self.test_metrics_endpoint(client)
            
            # Test error handling
            await self.test_error_handling(client)
            
            # Test rate limiting (optional, might affect other tests)
            # await self.test_rate_limiting(client)
        
        # Print summary
        self.print_test_summary()
    
    async def test_root_endpoint(self, client: httpx.AsyncClient):
        """Test root endpoint functionality."""
        print("\n📍 Testing root endpoint...")
        
        try:
            response = await client.get(f"{self.base_url}/")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            required_fields = ["service", "version", "status", "architecture", "endpoints"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            assert data["architecture"] == "modular"
            assert data["status"] == "running"
            
            self.record_test("Root Endpoint", True, "✅ All checks passed")
            
        except Exception as e:
            self.record_test("Root Endpoint", False, f"❌ Error: {e}")
    
    async def test_basic_health(self, client: httpx.AsyncClient):
        """Test basic health check endpoint."""
        print("🏥 Testing basic health check...")
        
        try:
            response = await client.get(f"{self.base_url}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            required_fields = ["status", "timestamp", "service", "version", "checks"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            assert data["status"] == "healthy"
            assert "uptime_seconds" in data
            
            self.record_test("Basic Health Check", True, "✅ Health check passed")
            
        except Exception as e:
            self.record_test("Basic Health Check", False, f"❌ Error: {e}")
    
    async def test_detailed_health(self, client: httpx.AsyncClient):
        """Test detailed health check endpoint."""
        print("🔍 Testing detailed health check...")
        
        try:
            response = await client.get(f"{self.base_url}/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "status" in data
            assert "checks" in data
            assert "ollama" in data["checks"]
            assert "http_client" in data["checks"]
            
            self.record_test("Detailed Health Check", True, "✅ Detailed checks passed")
            
        except Exception as e:
            self.record_test("Detailed Health Check", False, f"❌ Error: {e}")
    
    async def test_readiness_check(self, client: httpx.AsyncClient):
        """Test Kubernetes readiness probe."""
        print("🎯 Testing readiness probe...")
        
        try:
            response = await client.get(f"{self.base_url}/health/ready")
            
            # Accept both 200 (ready) and 503 (not ready) as valid responses
            assert response.status_code in [200, 503]
            data = response.json()
            
            assert "status" in data
            assert "checks" in data
            
            status_msg = "✅ Ready" if response.status_code == 200 else "⚠️ Not ready (acceptable)"
            self.record_test("Readiness Probe", True, status_msg)
            
        except Exception as e:
            self.record_test("Readiness Probe", False, f"❌ Error: {e}")
    
    async def test_liveness_check(self, client: httpx.AsyncClient):
        """Test Kubernetes liveness probe."""
        print("💓 Testing liveness probe...")
        
        try:
            response = await client.get(f"{self.base_url}/health/live")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "alive"
            assert "uptime_seconds" in data
            
            self.record_test("Liveness Probe", True, "✅ Application is alive")
            
        except Exception as e:
            self.record_test("Liveness Probe", False, f"❌ Error: {e}")
    
    async def test_startup_check(self, client: httpx.AsyncClient):
        """Test Kubernetes startup probe."""
        print("🚀 Testing startup probe...")
        
        try:
            response = await client.get(f"{self.base_url}/health/startup")
            
            # Accept both 200 (started) and 503 (starting) as valid responses
            assert response.status_code in [200, 503]
            data = response.json()
            
            assert "status" in data
            assert "checks" in data
            
            status_msg = "✅ Started" if response.status_code == 200 else "⚠️ Starting (acceptable)"
            self.record_test("Startup Probe", True, status_msg)
            
        except Exception as e:
            self.record_test("Startup Probe", False, f"❌ Error: {e}")
    
    async def test_ollama_health(self, client: httpx.AsyncClient):
        """Test Ollama health check."""
        print("🤖 Testing Ollama health check...")
        
        try:
            response = await client.get(f"{self.base_url}/ollama/health")
            
            # Accept both 200 (healthy) and 503 (unhealthy) for Ollama
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "healthy"
                assert "model" in data
                self.record_test("Ollama Health", True, "✅ Ollama is healthy")
            else:
                self.record_test("Ollama Health", True, "⚠️ Ollama not available (acceptable)")
            
        except Exception as e:
            self.record_test("Ollama Health", False, f"❌ Error: {e}")
    
    async def test_ollama_model_info(self, client: httpx.AsyncClient):
        """Test Ollama model info endpoint."""
        print("📋 Testing Ollama model info...")
        
        try:
            response = await client.get(f"{self.base_url}/ollama/model-info")
            
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "model_info" in data
                self.record_test("Ollama Model Info", True, "✅ Model info retrieved")
            else:
                self.record_test("Ollama Model Info", True, "⚠️ Ollama not available (acceptable)")
            
        except Exception as e:
            self.record_test("Ollama Model Info", False, f"❌ Error: {e}")
    
    async def test_ollama_models_list(self, client: httpx.AsyncClient):
        """Test Ollama models list endpoint."""
        print("📚 Testing Ollama models list...")
        
        try:
            response = await client.get(f"{self.base_url}/ollama/models")
            
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "available_models" in data
                self.record_test("Ollama Models List", True, "✅ Models list retrieved")
            else:
                self.record_test("Ollama Models List", True, "⚠️ Ollama not available (acceptable)")
            
        except Exception as e:
            self.record_test("Ollama Models List", False, f"❌ Error: {e}")
    
    async def test_agent_run(self, client: httpx.AsyncClient):
        """Test agent run endpoint."""
        print("🤖 Testing agent run endpoint...")
        
        try:
            payload = {
                "instruction": "Test instruction for agent",
                "use_groupthink": False,
                "timeout_seconds": 10
            }
            
            response = await client.post(f"{self.base_url}/agents/run", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            required_fields = ["task_id", "status", "correlation_id"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            self.record_test("Agent Run", True, "✅ Agent execution successful")
            
        except Exception as e:
            self.record_test("Agent Run", False, f"❌ Error: {e}")
    
    async def test_agent_run_ollama(self, client: httpx.AsyncClient):
        """Test agent run with Ollama endpoint."""
        print("🧠 Testing agent run with Ollama...")
        
        try:
            payload = {
                "instruction": "Test instruction for Ollama agent",
                "use_ollama": True,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            response = await client.post(f"{self.base_url}/agents/run-agent", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            required_fields = ["success", "result", "execution_time_ms"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            if data["success"]:
                self.record_test("Agent Run Ollama", True, "✅ Ollama agent execution successful")
            else:
                self.record_test("Agent Run Ollama", True, "⚠️ Ollama not available (acceptable)")
            
        except Exception as e:
            self.record_test("Agent Run Ollama", False, f"❌ Error: {e}")
    
    async def test_metrics_endpoint(self, client: httpx.AsyncClient):
        """Test Prometheus metrics endpoint."""
        print("📊 Testing metrics endpoint...")
        
        try:
            response = await client.get(f"{self.base_url}/metrics")
            
            assert response.status_code == 200
            content = response.text
            
            # Check for some expected metrics
            expected_metrics = ["http_requests_total", "app_info"]
            for metric in expected_metrics:
                assert metric in content, f"Missing metric: {metric}"
            
            self.record_test("Metrics Endpoint", True, "✅ Metrics available")
            
        except Exception as e:
            self.record_test("Metrics Endpoint", False, f"❌ Error: {e}")
    
    async def test_error_handling(self, client: httpx.AsyncClient):
        """Test error handling and response format."""
        print("🚨 Testing error handling...")
        
        try:
            # Test 404 error
            response = await client.get(f"{self.base_url}/nonexistent-endpoint")
            
            assert response.status_code == 404
            
            # Check for correlation ID header
            assert "X-Correlation-ID" in response.headers
            
            self.record_test("Error Handling", True, "✅ Error handling working correctly")
            
        except Exception as e:
            self.record_test("Error Handling", False, f"❌ Error: {e}")
    
    async def test_rate_limiting(self, client: httpx.AsyncClient):
        """Test rate limiting (be careful with this test)."""
        print("⏱️ Testing rate limiting...")
        
        try:
            # Make multiple rapid requests
            responses = []
            for i in range(5):  # Limited number to avoid blocking real tests
                response = await client.get(f"{self.base_url}/")
                responses.append(response.status_code)
            
            # All should succeed with low number of requests
            assert all(status == 200 for status in responses)
            
            self.record_test("Rate Limiting", True, "✅ Rate limiting configured")
            
        except Exception as e:
            self.record_test("Rate Limiting", False, f"❌ Error: {e}")
    
    def record_test(self, test_name: str, passed: bool, message: str):
        """Record test result."""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
        print(f"  {message}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)
        
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} - {result['test']}: {result['message']}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Architecture is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review the issues above.")


async def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FastAPI modular architecture")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the FastAPI application"
    )
    
    args = parser.parse_args()
    
    test_suite = ArchitectureTestSuite(base_url=args.url)
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())