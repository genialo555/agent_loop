#!/usr/bin/env python3
"""
Test script for Ollama integration with the agent API.

This script validates the Ollama integration by testing various scenarios
including health checks, model info, and different types of requests.
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaIntegrationTester:
    """Test suite for Ollama integration."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def test_api_health(self) -> bool:
        """Test basic API health."""
        try:
            response = await self.client.get(f"{self.api_base_url}/health")
            result = response.status_code == 200
            logger.info(f"API Health Check: {'PASS' if result else 'FAIL'}")
            return result
        except Exception as e:
            logger.error(f"API Health Check: FAIL - {e}")
            return False
    
    async def test_readiness_check(self) -> bool:
        """Test readiness check including Ollama."""
        try:
            response = await self.client.get(f"{self.api_base_url}/ready")
            if response.status_code == 200:
                data = response.json()
                ollama_ready = data.get("checks", {}).get("ollama", False)
                logger.info(f"Readiness Check (Ollama): {'PASS' if ollama_ready else 'FAIL'}")
                return ollama_ready
            else:
                logger.error(f"Readiness Check: FAIL - Status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Readiness Check: FAIL - {e}")
            return False
    
    async def test_ollama_health(self) -> bool:
        """Test dedicated Ollama health endpoint."""
        try:
            response = await self.client.get(f"{self.api_base_url}/ollama/health")
            result = response.status_code == 200
            if result:
                data = response.json()
                logger.info(f"Ollama Health: PASS - Model: {data.get('model', 'unknown')}")
            else:
                logger.error(f"Ollama Health: FAIL - Status {response.status_code}")
            return result
        except Exception as e:
            logger.error(f"Ollama Health: FAIL - {e}")
            return False
    
    async def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint."""
        try:
            response = await self.client.get(f"{self.api_base_url}/ollama/model-info")
            if response.status_code == 200:
                data = response.json()
                model_info = data.get("model_info", {})
                logger.info(f"Model Info: PASS - {model_info.get('name', 'unknown')}")
                return model_info
            else:
                logger.error(f"Model Info: FAIL - Status {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Model Info: FAIL - {e}")
            return {}
    
    async def test_basic_generation(self) -> bool:
        """Test basic text generation."""
        payload = {
            "instruction": "Hello! Please respond with a brief greeting.",
            "use_ollama": True,
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        try:
            start_time = time.time()
            response = await self.client.post(
                f"{self.api_base_url}/run-agent",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                model_used = data.get("model_used", "unknown")
                inference_time = data.get("inference_metrics", {}).get("inference_time_ms", 0)
                response_text = data.get("result", {}).get("response", "")
                
                total_time = (time.time() - start_time) * 1000
                
                logger.info(f"Basic Generation: {'PASS' if success else 'FAIL'}")
                logger.info(f"  Model: {model_used}")
                logger.info(f"  Inference Time: {inference_time:.2f}ms")
                logger.info(f"  Total Time: {total_time:.2f}ms")
                logger.info(f"  Response Length: {len(response_text)} chars")
                logger.info(f"  Response Preview: {response_text[:100]}...")
                
                return success
            else:
                logger.error(f"Basic Generation: FAIL - Status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Basic Generation: FAIL - {e}")
            return False
    
    async def test_browser_fallback(self) -> bool:
        """Test browser URL request with Ollama enhancement."""
        payload = {
            "instruction": "Open https://httpbin.org/json and analyze the content",
            "use_ollama": True,
            "temperature": 0.5,
            "max_tokens": 200,
            "system_prompt": "You are analyzing web content. Provide insights about the data structure."
        }
        
        try:
            start_time = time.time()
            response = await self.client.post(
                f"{self.api_base_url}/run-agent",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                result = data.get("result", {})
                enhanced_context = result.get("enhanced_context", False)
                inference_metrics = data.get("inference_metrics", {})
                urls_mentioned = inference_metrics.get("urls_mentioned", [])
                
                total_time = (time.time() - start_time) * 1000
                
                logger.info(f"Browser + Ollama: {'PASS' if success else 'FAIL'}")
                logger.info(f"  Enhanced Context: {enhanced_context}")
                logger.info(f"  URLs Detected: {urls_mentioned}")
                logger.info(f"  Total Time: {total_time:.2f}ms")
                
                return success
            else:
                logger.error(f"Browser + Ollama: FAIL - Status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Browser + Ollama: FAIL - {e}")
            return False
    
    async def test_performance_parameters(self) -> bool:
        """Test different performance parameters."""
        test_cases = [
            {"temperature": 0.1, "max_tokens": 50, "name": "Low temp, short"},
            {"temperature": 0.9, "max_tokens": 500, "name": "High temp, long"},
            {"temperature": 0.7, "max_tokens": 200, "name": "Balanced"},
        ]
        
        results = []
        
        for case in test_cases:
            payload = {
                "instruction": "Explain the concept of machine learning in simple terms.",
                "use_ollama": True,
                "temperature": case["temperature"],
                "max_tokens": case["max_tokens"]
            }
            
            try:
                start_time = time.time()
                response = await self.client.post(
                    f"{self.api_base_url}/run-agent",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    success = data.get("success", False)
                    inference_metrics = data.get("inference_metrics", {})
                    inference_time = inference_metrics.get("inference_time_ms", 0)
                    response_length = inference_metrics.get("response_length", 0)
                    
                    total_time = (time.time() - start_time) * 1000
                    
                    result = {
                        "case": case["name"],
                        "success": success,
                        "inference_time": inference_time,
                        "total_time": total_time,
                        "response_length": response_length,
                        "temperature": case["temperature"],
                        "max_tokens": case["max_tokens"]
                    }
                    
                    results.append(result)
                    logger.info(f"Performance Test ({case['name']}): {'PASS' if success else 'FAIL'}")
                    logger.info(f"  Inference: {inference_time:.2f}ms, Total: {total_time:.2f}ms, Length: {response_length}")
                
            except Exception as e:
                logger.error(f"Performance Test ({case['name']}): FAIL - {e}")
                results.append({"case": case["name"], "success": False, "error": str(e)})
        
        # Summary
        successful_tests = sum(1 for r in results if r.get("success", False))
        logger.info(f"Performance Tests Summary: {successful_tests}/{len(test_cases)} passed")
        
        return successful_tests > 0
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary."""
        logger.info("Starting Ollama Integration Test Suite")
        logger.info("=" * 50)
        
        test_results = {}
        
        # Basic connectivity tests
        test_results["api_health"] = await self.test_api_health()
        test_results["readiness_check"] = await self.test_readiness_check()
        test_results["ollama_health"] = await self.test_ollama_health()
        
        # Model info test
        model_info = await self.test_model_info()
        test_results["model_info"] = bool(model_info)
        
        # Generation tests
        test_results["basic_generation"] = await self.test_basic_generation()
        test_results["browser_fallback"] = await self.test_browser_fallback()
        test_results["performance_tests"] = await self.test_performance_parameters()
        
        # Summary
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        logger.info("=" * 50)
        logger.info(f"Test Suite Complete: {passed_tests}/{total_tests} tests passed")
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests * 100,
            "individual_results": test_results,
            "model_info": model_info
        }
        
        return summary
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Main test runner."""
    tester = OllamaIntegrationTester()
    
    try:
        summary = await tester.run_all_tests()
        
        # Save results to file
        with open("/tmp/ollama_integration_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("OLLAMA INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Results saved to: /tmp/ollama_integration_test_results.json")
        
        if summary['success_rate'] >= 80:
            print("\n✅ Integration tests PASSED - Ollama is ready for production!")
            exit_code = 0
        else:
            print("\n❌ Integration tests FAILED - Check Ollama configuration")
            exit_code = 1
        
        return exit_code
        
    finally:
        await tester.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)