#!/usr/bin/env python3
"""
Comprehensive Health Check Script for MLOps Pipeline

Performs deep health checks on all system components including:
- API endpoints and response validation
- Model inference performance
- Database connectivity and integrity
- External service dependencies
- Resource utilization monitoring
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import psutil
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class HealthCheckResult:
    """Represents the result of a single health check."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    details: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class HealthChecker:
    """
    Comprehensive health checker for MLOps infrastructure.
    
    Performs various types of health checks:
    - HTTP endpoint availability and response validation
    - Service performance benchmarking
    - Resource utilization monitoring
    - Database connectivity and query performance
    - Model inference validation
    """
    
    def __init__(self, timeout: int = 30, retry_interval: int = 5):
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.logger = logging.getLogger(__name__)
        self.results: List[HealthCheckResult] = []
        
    async def run_all_checks(self, endpoint: str) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        self.logger.info("Starting comprehensive health check...")
        
        checks = [
            self.check_api_health(endpoint),
            self.check_model_inference(endpoint),
            self.check_ollama_service(endpoint),
            self.check_database_health(endpoint),
            self.check_prometheus_metrics(endpoint),
            self.check_grafana_dashboard(),
            self.check_system_resources(),
            self.check_docker_health(),
        ]
        
        # Run checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.results.append(HealthCheckResult(
                    component=f"check_{i}",
                    status="unhealthy",
                    response_time_ms=0,
                    details={},
                    timestamp=datetime.utcnow().isoformat(),
                    error=str(result)
                ))
            elif isinstance(result, HealthCheckResult):
                self.results.append(result)
        
        return self._compile_report()
        
    async def check_api_health(self, endpoint: str) -> HealthCheckResult:
        """Check FastAPI application health."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Basic health check
                response = await client.get(f"{endpoint}/health")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    health_data = response.json()
                    
                    # Validate response structure
                    required_fields = ['status', 'service', 'timestamp']
                    missing_fields = [field for field in required_fields 
                                    if field not in health_data]
                    
                    status = "healthy" if not missing_fields else "degraded"
                    
                    return HealthCheckResult(
                        component="fastapi_health",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "response_data": health_data,
                            "missing_fields": missing_fields,
                            "http_status": response.status_code
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    return HealthCheckResult(
                        component="fastapi_health",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="fastapi_health",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=str(e)
            )
    
    async def check_model_inference(self, endpoint: str) -> HealthCheckResult:
        """Test model inference with sample requests."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                test_payload = {
                    "instruction": "Hello, this is a health check test. Please respond briefly.",
                    "use_ollama": True,
                    "temperature": 0.1,
                    "max_tokens": 50
                }
                
                response = await client.post(
                    f"{endpoint}/run-agent",
                    json=test_payload
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate inference response
                    is_successful = result.get('success', False)
                    has_result = bool(result.get('result', '').strip())
                    execution_time = result.get('execution_time_ms', 0)
                    
                    # Determine status based on performance thresholds
                    if is_successful and has_result:
                        if execution_time < 5000:  # < 5 seconds
                            status = "healthy"
                        elif execution_time < 15000:  # < 15 seconds
                            status = "degraded"
                        else:
                            status = "unhealthy"
                    else:
                        status = "unhealthy"
                    
                    return HealthCheckResult(
                        component="model_inference",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "inference_success": is_successful,
                            "has_output": has_result,
                            "model_execution_time_ms": execution_time,
                            "output_length": len(result.get('result', '')),
                            "response_quality": "valid" if has_result else "empty"
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    return HealthCheckResult(
                        component="model_inference",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Inference failed: HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="model_inference",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Inference check failed: {str(e)}"
            )
    
    async def check_ollama_service(self, endpoint: str) -> HealthCheckResult:
        """Check Ollama service health and model availability."""
        start_time = time.time()
        
        try:
            # Extract base URL for Ollama (assuming standard setup)
            base_url = endpoint.replace(":8000", ":11434")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check Ollama version endpoint
                response = await client.get(f"{base_url}/api/version")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    version_info = response.json()
                    
                    # Check model availability
                    models_response = await client.get(f"{base_url}/api/tags")
                    models_data = models_response.json() if models_response.status_code == 200 else {}
                    
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    has_gemma = any('gemma' in model.lower() for model in available_models)
                    
                    status = "healthy" if has_gemma else "degraded"
                    
                    return HealthCheckResult(
                        component="ollama_service",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "version": version_info.get('version', 'unknown'),
                            "available_models": available_models,
                            "has_gemma_model": has_gemma,
                            "model_count": len(available_models)
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    return HealthCheckResult(
                        component="ollama_service",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Ollama service unavailable: HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="ollama_service",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Ollama check failed: {str(e)}"
            )
    
    async def check_database_health(self, endpoint: str) -> HealthCheckResult:
        """Check database connectivity and basic operations."""
        start_time = time.time()
        
        try:
            # For this implementation, we'll check if the API can connect to its database
            # by using the ready endpoint which typically includes database checks
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{endpoint}/ready")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    ready_data = response.json()
                    
                    # Check for database-related information in ready response
                    db_status = ready_data.get('checks', {}).get('database', 'unknown')
                    
                    status = "healthy" if db_status == "healthy" else "degraded"
                    
                    return HealthCheckResult(
                        component="database",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "ready_status": ready_data.get('status', 'unknown'),
                            "database_check": db_status,
                            "checks": ready_data.get('checks', {})
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    return HealthCheckResult(
                        component="database",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Database health check failed: HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Database check failed: {str(e)}"
            )
    
    async def check_prometheus_metrics(self, endpoint: str) -> HealthCheckResult:
        """Check Prometheus metrics endpoint and validate metrics."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{endpoint}/metrics")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    metrics_text = response.text
                    
                    # Parse Prometheus metrics
                    try:
                        families = list(text_string_to_metric_families(metrics_text))
                        metric_names = [family.name for family in families]
                        
                        # Check for expected metrics
                        expected_metrics = [
                            'http_requests_total',
                            'http_request_duration_seconds',
                            'python_info'
                        ]
                        
                        present_metrics = [metric for metric in expected_metrics 
                                         if any(metric in name for name in metric_names)]
                        
                        status = "healthy" if len(present_metrics) >= 2 else "degraded"
                        
                        return HealthCheckResult(
                            component="prometheus_metrics",
                            status=status,
                            response_time_ms=response_time,
                            details={
                                "total_metrics": len(metric_names),
                                "expected_metrics_present": present_metrics,
                                "metrics_sample": metric_names[:10]
                            },
                            timestamp=datetime.utcnow().isoformat()
                        )
                        
                    except Exception as parse_error:
                        return HealthCheckResult(
                            component="prometheus_metrics",
                            status="degraded",
                            response_time_ms=response_time,
                            details={"metrics_length": len(metrics_text)},
                            timestamp=datetime.utcnow().isoformat(),
                            error=f"Metrics parsing failed: {str(parse_error)}"
                        )
                else:
                    return HealthCheckResult(
                        component="prometheus_metrics",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Metrics endpoint unavailable: HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="prometheus_metrics",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Metrics check failed: {str(e)}"
            )
    
    async def check_grafana_dashboard(self) -> HealthCheckResult:
        """Check Grafana dashboard availability."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get("http://localhost:3000/api/health")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    health_data = response.json()
                    
                    return HealthCheckResult(
                        component="grafana_dashboard",
                        status="healthy",
                        response_time_ms=response_time,
                        details={
                            "health_data": health_data,
                            "version": health_data.get('version', 'unknown')
                        },
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    return HealthCheckResult(
                        component="grafana_dashboard",
                        status="unhealthy",
                        response_time_ms=response_time,
                        details={"http_status": response.status_code},
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Grafana unavailable: HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                component="grafana_dashboard",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Grafana check failed: {str(e)}"
            )
    
    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = "unhealthy"
            elif cpu_percent > 75 or memory.percent > 75 or disk.percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                response_time_ms=response_time,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"System resource check failed: {str(e)}"
            )
    
    async def check_docker_health(self) -> HealthCheckResult:
        """Check Docker container health."""
        start_time = time.time()
        
        try:
            import subprocess
            
            # Check Docker daemon
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Get running containers
                containers_result = subprocess.run(
                    ['docker', 'ps', '--format', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                running_containers = []
                if containers_result.returncode == 0:
                    for line in containers_result.stdout.strip().split('\n'):
                        if line.strip():
                            try:
                                container_info = json.loads(line)
                                running_containers.append(container_info)
                            except json.JSONDecodeError:
                                pass
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    component="docker_health",
                    status="healthy",
                    response_time_ms=response_time,
                    details={
                        "docker_daemon": "running",
                        "running_containers": len(running_containers),
                        "container_names": [c.get('Names', 'unknown') for c in running_containers]
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                return HealthCheckResult(
                    component="docker_health",
                    status="unhealthy",
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={},
                    timestamp=datetime.utcnow().isoformat(),
                    error=f"Docker daemon not available: {result.stderr}"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="docker_health",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                details={},
                timestamp=datetime.utcnow().isoformat(),
                error=f"Docker check failed: {str(e)}"
            )
    
    def _compile_report(self) -> Dict[str, Any]:
        """Compile comprehensive health report."""
        healthy_count = sum(1 for r in self.results if r.status == "healthy")
        degraded_count = sum(1 for r in self.results if r.status == "degraded")
        unhealthy_count = sum(1 for r in self.results if r.status == "unhealthy")
        
        # Overall system status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate average response time
        avg_response_time = sum(r.response_time_ms for r in self.results) / len(self.results)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": len(self.results),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "average_response_time_ms": round(avg_response_time, 2)
            },
            "checks": [
                {
                    "component": r.component,
                    "status": r.status,
                    "response_time_ms": r.response_time_ms,
                    "details": r.details,
                    "timestamp": r.timestamp,
                    "error": r.error
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on health check results."""
        recommendations = []
        
        unhealthy_components = [r for r in self.results if r.status == "unhealthy"]
        degraded_components = [r for r in self.results if r.status == "degraded"]
        
        if unhealthy_components:
            recommendations.append(
                f"CRITICAL: {len(unhealthy_components)} component(s) are unhealthy. "
                f"Immediate attention required for: {', '.join(r.component for r in unhealthy_components)}"
            )
        
        if degraded_components:
            recommendations.append(
                f"WARNING: {len(degraded_components)} component(s) are degraded. "
                f"Monitor closely: {', '.join(r.component for r in degraded_components)}"
            )
        
        # Specific recommendations
        for result in self.results:
            if result.component == "model_inference" and result.status != "healthy":
                if result.details.get("model_execution_time_ms", 0) > 10000:
                    recommendations.append(
                        "Model inference is slow. Consider model optimization or resource scaling."
                    )
            
            if result.component == "system_resources" and result.status != "healthy":
                details = result.details
                if details.get("cpu_percent", 0) > 75:
                    recommendations.append(f"High CPU usage: {details['cpu_percent']:.1f}%")
                if details.get("memory_percent", 0) > 75:
                    recommendations.append(f"High memory usage: {details['memory_percent']:.1f}%")
                if details.get("disk_percent", 0) > 80:
                    recommendations.append(f"High disk usage: {details['disk_percent']:.1f}%")
        
        if not recommendations:
            recommendations.append("All systems operating normally. No action required.")
        
        return recommendations


async def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(description="Comprehensive MLOps Health Check")
    parser.add_argument("--endpoint", required=True, help="API endpoint to check")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--retry-interval", type=int, default=5, help="Retry interval in seconds")
    parser.add_argument("--output", help="Output file for health report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--exit-on-failure", action="store_true", help="Exit with error code if any check fails")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize health checker
        checker = HealthChecker(args.timeout, args.retry_interval)
        
        # Run all health checks
        report = await checker.run_all_checks(args.endpoint)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Health report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
        
        # Log summary
        status = report["overall_status"]
        summary = report["summary"]
        
        logger.info(f"Health check complete: {status.upper()}")
        logger.info(f"Checks: {summary['healthy']} healthy, {summary['degraded']} degraded, {summary['unhealthy']} unhealthy")
        
        # Exit with appropriate code
        if args.exit_on_failure and status != "healthy":
            logger.error("Health check failed!")
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())