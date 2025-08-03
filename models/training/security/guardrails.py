#!/usr/bin/env python3
"""
Security Guardrails and Sandboxing System

This module implements comprehensive security guardrails, sandboxing strategies,
and runtime protection mechanisms for safe dataset processing and model training
with multi-layered security controls.
"""

import os
import sys
import time
import json
import subprocess
import tempfile
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Optional, Union, Any, Set, Protocol, 
    Literal, TypeVar, Generic, Callable, ContextManager
)
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator
# Optional Docker import
try:
    import docker
    from docker.errors import DockerException
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DockerException = Exception
    DOCKER_AVAILABLE = False
import psutil

from .prompt_injection_analysis import SecurityThreat, ThreatCategory
from .advanced_detection import PromptInjectionDetectionSystem, DetectionResult

logger = structlog.get_logger(__name__)

# Type definitions
SecurityLevel = Literal["minimal", "standard", "strict", "maximum"]
SandboxType = Literal["process", "docker", "chroot", "virtual"]
T = TypeVar('T')


class SecurityViolationError(Exception):
    """Raised when a security guardrail is violated."""
    
    def __init__(self, message: str, violation_type: str, severity: str):
        self.message = message
        self.violation_type = violation_type
        self.severity = severity
        super().__init__(message)


class SandboxError(Exception):
    """Raised when sandboxing operations fail."""
    pass


@dataclass(frozen=True)
class SecurityPolicy:
    """Comprehensive security policy configuration."""
    
    # Input validation settings
    max_file_size: int = Field(default=10_000_000, description="Maximum file size in bytes")
    max_memory_usage: int = Field(default=1_000_000_000, description="Maximum memory usage in bytes")
    max_processing_time: int = Field(default=300, description="Maximum processing time in seconds")
    
    # Content filtering settings
    block_dangerous_functions: bool = Field(default=True, description="Block dangerous function calls")
    sanitize_html_content: bool = Field(default=True, description="Sanitize HTML/DOM content")
    filter_prompt_injections: bool = Field(default=True, description="Filter prompt injection attempts")
    
    # Sandboxing settings
    sandbox_type: SandboxType = Field(default="process", description="Type of sandboxing to use")
    enable_network_isolation: bool = Field(default=True, description="Isolate network access")
    enable_filesystem_isolation: bool = Field(default=True, description="Isolate filesystem access")
    
    # Monitoring settings
    log_security_events: bool = Field(default=True, description="Log all security events")
    alert_on_violations: bool = Field(default=True, description="Send alerts on security violations")
    track_resource_usage: bool = Field(default=True, description="Monitor resource consumption")
    
    # Execution limits
    allowed_file_extensions: Set[str] = Field(
        default_factory=lambda: {".json", ".jsonl", ".txt", ".html", ".png", ".jpg", ".jpeg"},
        description="Allowed file extensions for processing"
    )
    blocked_directories: Set[str] = Field(
        default_factory=lambda: {"/etc", "/sys", "/proc", "/dev", "/root", "/home"},
        description="Directories blocked from access"
    )
    blocked_commands: Set[str] = Field(
        default_factory=lambda: {"rm", "del", "format", "dd", "sudo", "su", "chmod", "chown"},
        description="System commands blocked from execution"
    )


class SecurityGuardrail(ABC):
    """Base class for security guardrails."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    @abstractmethod
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        """Check if the guardrail is violated."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the guardrail."""
        pass


class FileSizeGuardrail(SecurityGuardrail):
    """Guardrail to enforce file size limits."""
    
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        file_path = context.get("file_path")
        if not file_path or not isinstance(file_path, (str, Path)):
            return None
        
        try:
            file_path = Path(file_path)
            if file_path.exists():
                size = file_path.stat().st_size
                if size > self.policy.max_file_size:
                    return SecurityViolationError(
                        f"File size {size} exceeds limit {self.policy.max_file_size}",
                        "file_size_violation",
                        "high"
                    )
        except Exception as e:
            self.logger.warning("Error checking file size", error=str(e))
        
        return None
    
    def get_description(self) -> str:
        return f"Enforces maximum file size of {self.policy.max_file_size} bytes"


class FileExtensionGuardrail(SecurityGuardrail):
    """Guardrail to enforce allowed file extensions."""
    
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        file_path = context.get("file_path")
        if not file_path:
            return None
        
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension not in self.policy.allowed_file_extensions:
                return SecurityViolationError(
                    f"File extension '{extension}' not allowed",
                    "file_extension_violation",
                    "medium"
                )
        except Exception as e:
            self.logger.warning("Error checking file extension", error=str(e))
        
        return None
    
    def get_description(self) -> str:
        return f"Allows only file extensions: {', '.join(self.policy.allowed_file_extensions)}"


class ContentSecurityGuardrail(SecurityGuardrail):
    """Guardrail to check content for security threats."""
    
    def __init__(self, policy: SecurityPolicy):
        super().__init__(policy)
        self.detector = PromptInjectionDetectionSystem(strict_mode=True)
    
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        content = context.get("content")
        if not content:
            return None
        
        try:
            # Analyze content for security threats
            detection_result = self.detector.analyze_content(content)
            
            if detection_result.is_malicious and detection_result.risk_score > 80:
                return SecurityViolationError(
                    f"Content contains high-risk security threats (score: {detection_result.risk_score})",
                    "content_security_violation",
                    "critical"
                )
            elif detection_result.risk_score > 60:
                return SecurityViolationError(
                    f"Content contains moderate security threats (score: {detection_result.risk_score})",
                    "content_security_violation",
                    "medium"
                )
        except Exception as e:
            self.logger.warning("Error analyzing content security", error=str(e))
        
        return None
    
    def get_description(self) -> str:
        return "Analyzes content for prompt injection and other security threats"


class ResourceUsageGuardrail(SecurityGuardrail):
    """Guardrail to monitor and limit resource usage."""
    
    def __init__(self, policy: SecurityPolicy):
        super().__init__(policy)
        self.start_time = None
        self.process = psutil.Process()
    
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        try:
            # Check memory usage
            memory_info = self.process.memory_info()
            if memory_info.rss > self.policy.max_memory_usage:
                return SecurityViolationError(
                    f"Memory usage {memory_info.rss} exceeds limit {self.policy.max_memory_usage}",
                    "memory_usage_violation",
                    "high"
                )
            
            # Check processing time
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > self.policy.max_processing_time:
                    return SecurityViolationError(
                        f"Processing time {elapsed:.1f}s exceeds limit {self.policy.max_processing_time}s",
                        "time_limit_violation",
                        "high"
                    )
            else:
                self.start_time = time.time()
        
        except Exception as e:
            self.logger.warning("Error checking resource usage", error=str(e))
        
        return None
    
    def get_description(self) -> str:
        return f"Monitors memory ({self.policy.max_memory_usage} bytes) and time ({self.policy.max_processing_time}s) limits"


class FunctionCallGuardrail(SecurityGuardrail):
    """Guardrail to check for dangerous function calls."""
    
    DANGEROUS_FUNCTIONS = {
        # System execution
        "exec", "eval", "system", "subprocess", "shell", "os.system",
        # File operations
        "rm", "del", "unlink", "rmdir", "format", "fdisk", "dd",
        # Network operations
        "wget", "curl", "fetch", "requests.get", "urllib.request",
        # Process control
        "kill", "killall", "sudo", "su", "chmod", "chown",
    }
    
    def check(self, context: Dict[str, Any]) -> Optional[SecurityViolationError]:
        content = context.get("content")
        if not content:
            return None
        
        try:
            # Check for dangerous function calls in JSON content
            if isinstance(content, dict):
                violations = self._check_dict_content(content)
            else:
                violations = self._check_text_content(str(content))
            
            if violations:
                return SecurityViolationError(
                    f"Dangerous function calls detected: {', '.join(violations)}",
                    "dangerous_function_violation",
                    "critical"
                )
        except Exception as e:
            self.logger.warning("Error checking function calls", error=str(e))
        
        return None
    
    def _check_dict_content(self, content: Dict[str, Any]) -> List[str]:
        """Check dictionary content for dangerous function calls."""
        violations = []
        
        # Check function calls in structured data
        if "function" in content:
            func_name = content["function"].get("name", "")
            if any(dangerous in func_name.lower() for dangerous in self.DANGEROUS_FUNCTIONS):
                violations.append(func_name)
        
        if "tool_calls" in content:
            for call in content.get("tool_calls", []):
                if isinstance(call, dict) and "function" in call:
                    func_name = call["function"].get("name", "")
                    if any(dangerous in func_name.lower() for dangerous in self.DANGEROUS_FUNCTIONS):
                        violations.append(func_name)
        
        # Recursively check nested content
        for key, value in content.items():
            if isinstance(value, dict):
                violations.extend(self._check_dict_content(value))
            elif isinstance(value, str):
                violations.extend(self._check_text_content(value))
        
        return violations
    
    def _check_text_content(self, content: str) -> List[str]:
        """Check text content for dangerous function calls."""
        violations = []
        content_lower = content.lower()
        
        for dangerous_func in self.DANGEROUS_FUNCTIONS:
            if dangerous_func in content_lower:
                violations.append(dangerous_func)
        
        return violations
    
    def get_description(self) -> str:
        return f"Blocks dangerous function calls: {', '.join(list(self.DANGEROUS_FUNCTIONS)[:10])}..."


class SecurityGuardrailManager:
    """Manages multiple security guardrails."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.guardrails: List[SecurityGuardrail] = []
        self.logger = structlog.get_logger(__name__)
        
        # Initialize standard guardrails
        self._initialize_guardrails()
    
    def _initialize_guardrails(self):
        """Initialize standard security guardrails."""
        self.guardrails = [
            FileSizeGuardrail(self.policy),
            FileExtensionGuardrail(self.policy),
            ContentSecurityGuardrail(self.policy),
            ResourceUsageGuardrail(self.policy),
            FunctionCallGuardrail(self.policy),
        ]
    
    def add_guardrail(self, guardrail: SecurityGuardrail):
        """Add a custom guardrail."""
        self.guardrails.append(guardrail)
    
    def check_all(self, context: Dict[str, Any]) -> List[SecurityViolationError]:
        """Check all guardrails and return violations."""
        violations = []
        
        for guardrail in self.guardrails:
            try:
                violation = guardrail.check(context)
                if violation:
                    violations.append(violation)
                    
                    if self.policy.log_security_events:
                        self.logger.warning(
                            "Security guardrail violation",
                            guardrail=guardrail.__class__.__name__,
                            violation_type=violation.violation_type,
                            severity=violation.severity,
                            message=violation.message
                        )
            except Exception as e:
                self.logger.error(
                    "Error in security guardrail",
                    guardrail=guardrail.__class__.__name__,
                    error=str(e)
                )
        
        return violations
    
    def is_safe(self, context: Dict[str, Any]) -> bool:
        """Check if context is safe according to all guardrails."""
        violations = self.check_all(context)
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            return False
        
        # In strict mode, any high-severity violation fails
        if self.policy.sandbox_type in ["strict", "maximum"]:
            high_violations = [v for v in violations if v.severity in ["high", "critical"]]
            if high_violations:
                return False
        
        return True


class ProcessSandbox:
    """Process-level sandboxing using subprocess isolation."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = structlog.get_logger(__name__)
    
    @contextlib.contextmanager
    def execute(self, func: Callable, *args, **kwargs) -> ContextManager[Any]:
        """Execute function in process sandbox."""
        try:
            # Create isolated process environment
            env = os.environ.copy()
            
            # Restrict environment variables
            restricted_vars = ["PATH", "HOME", "USER", "SHELL"]
            for var in restricted_vars:
                if var in env:
                    env[var] = "/tmp"  # Restrict to safe directory
            
            # Execute with resource limits
            with self._resource_limits():
                result = func(*args, **kwargs)
                yield result
        
        except Exception as e:
            self.logger.error("Process sandbox execution failed", error=str(e))
            raise SandboxError(f"Sandbox execution failed: {e}")
    
    @contextlib.contextmanager
    def _resource_limits(self):
        """Apply resource limits."""
        import resource
        
        # Set memory limit
        if self.policy.max_memory_usage > 0:
            resource.setrlimit(resource.RLIMIT_AS, (self.policy.max_memory_usage, self.policy.max_memory_usage))
        
        # Set CPU time limit
        if self.policy.max_processing_time > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (self.policy.max_processing_time, self.policy.max_processing_time))
        
        try:
            yield
        finally:
            # Reset limits
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


class DockerSandbox:
    """Docker-based sandboxing for maximum isolation."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = structlog.get_logger(__name__)
        
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
            except DockerException as e:
                self.logger.warning("Docker not available", error=str(e))
                self.client = None
        else:
            self.logger.warning("Docker not installed - Docker sandbox not available")
            self.client = None
    
    @contextlib.contextmanager
    def execute(self, func: Callable, *args, **kwargs) -> ContextManager[Any]:
        """Execute function in Docker sandbox."""
        if not self.client:
            raise SandboxError("Docker not available for sandboxing")
        
        container = None
        try:
            # Create sandbox container
            container = self._create_sandbox_container()
            
            # Execute function in container
            result = self._execute_in_container(container, func, *args, **kwargs)
            yield result
        
        except Exception as e:
            self.logger.error("Docker sandbox execution failed", error=str(e))
            raise SandboxError(f"Docker sandbox failed: {e}")
        
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    self.logger.warning("Failed to cleanup container", error=str(e))
    
    def _create_sandbox_container(self):
        """Create isolated sandbox container."""
        # Use minimal Python image
        image = "python:3.11-alpine"
        
        # Container configuration
        config = {
            "image": image,
            "detach": True,
            "mem_limit": f"{self.policy.max_memory_usage}b",
            "network_disabled": self.policy.enable_network_isolation,
            "read_only": True,
            "security_opt": ["no-new-privileges:true"],
            "user": "nobody",
            "working_dir": "/tmp",
            "tmpfs": {"/tmp": "size=100m,noexec"},
        }
        
        if self.policy.enable_filesystem_isolation:
            config["volumes"] = {"/dev/null": {"bind": "/dev/null", "mode": "ro"}}
        
        return self.client.containers.create(**config)
    
    def _execute_in_container(self, container, func: Callable, *args, **kwargs):
        """Execute function inside container."""
        container.start()
        
        # Serialize function and arguments
        import pickle
        func_data = pickle.dumps((func, args, kwargs))
        
        # Create execution script
        script = f"""
import pickle
import sys
import json

try:
    func_data = {repr(func_data)}
    func, args, kwargs = pickle.loads(func_data)
    result = func(*args, **kwargs)
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
        
        # Execute in container
        exec_result = container.exec_run(
            ["python", "-c", script],
            user="nobody",
            workdir="/tmp"
        )
        
        # Parse result
        try:
            result_data = json.loads(exec_result.output.decode())
            if result_data["success"]:
                return result_data["result"]
            else:
                raise Exception(result_data["error"])
        except json.JSONDecodeError:
            raise SandboxError(f"Container execution failed: {exec_result.output.decode()}")


class SandboxManager:
    """Manages different types of sandboxes."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = structlog.get_logger(__name__)
        
        # Initialize appropriate sandbox
        if policy.sandbox_type == "docker":
            self.sandbox = DockerSandbox(policy)
        elif policy.sandbox_type == "process":
            self.sandbox = ProcessSandbox(policy)
        else:
            self.sandbox = ProcessSandbox(policy)  # Default fallback
    
    @contextlib.contextmanager
    def secure_execution(self, func: Callable, *args, **kwargs) -> ContextManager[Any]:
        """Execute function in secure sandbox environment."""
        try:
            with self.sandbox.execute(func, *args, **kwargs) as result:
                yield result
        except SandboxError:
            raise
        except Exception as e:
            self.logger.error("Sandbox execution error", error=str(e))
            raise SandboxError(f"Secure execution failed: {e}")


class SecureDatasetProcessor:
    """Main secure processor combining guardrails and sandboxing."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.guardrail_manager = SecurityGuardrailManager(policy)
        self.sandbox_manager = SandboxManager(policy)
        self.logger = structlog.get_logger(__name__)
    
    def process_content_safely(
        self, 
        content: Union[str, Dict[str, Any]], 
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Process content with full security protection."""
        
        # Create security context
        context = {
            "content": content,
            "file_path": file_path,
            "timestamp": time.time()
        }
        
        # Check guardrails
        violations = self.guardrail_manager.check_all(context)
        
        # Handle violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            error_msg = f"Critical security violations: {[v.message for v in critical_violations]}"
            self.logger.error("Processing blocked", violations=error_msg)
            raise SecurityViolationError(error_msg, "critical_violation", "critical")
        
        # Process in sandbox if there are any violations
        if violations or self.policy.sandbox_type != "minimal":
            return self._process_in_sandbox(content, context)
        else:
            return self._process_directly(content, context)
    
    def _process_in_sandbox(self, content: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content in secure sandbox."""
        def safe_processor(content_arg):
            # Safe processing logic here
            return {
                "processed_content": content_arg,
                "processing_method": "sandboxed",
                "security_context": context,
                "status": "success"
            }
        
        try:
            with self.sandbox_manager.secure_execution(safe_processor, content) as result:
                self.logger.info("Content processed in sandbox", content_hash=hash(str(content)))
                return result
        except Exception as e:
            self.logger.error("Sandbox processing failed", error=str(e))
            raise
    
    def _process_directly(self, content: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content directly (when safe)."""
        self.logger.info("Content processed directly", content_hash=hash(str(content)))
        return {
            "processed_content": content,
            "processing_method": "direct",
            "security_context": context,
            "status": "success"
        }


# Predefined security policies
class SecurityPolicies:
    """Predefined security policy configurations."""
    
    @staticmethod
    def minimal() -> SecurityPolicy:
        """Minimal security - for trusted content only."""
        return SecurityPolicy(
            max_file_size=50_000_000,  # 50MB
            max_memory_usage=2_000_000_000,  # 2GB
            max_processing_time=600,  # 10 minutes
            block_dangerous_functions=False,
            sanitize_html_content=False,
            filter_prompt_injections=False,
            sandbox_type="process",
            enable_network_isolation=False,
            enable_filesystem_isolation=False,
        )
    
    @staticmethod
    def standard() -> SecurityPolicy:
        """Standard security - balanced protection."""
        return SecurityPolicy(
            max_file_size=10_000_000,  # 10MB
            max_memory_usage=1_000_000_000,  # 1GB
            max_processing_time=300,  # 5 minutes
            block_dangerous_functions=True,
            sanitize_html_content=True,
            filter_prompt_injections=True,
            sandbox_type="process",
            enable_network_isolation=True,
            enable_filesystem_isolation=True,
        )
    
    @staticmethod
    def strict() -> SecurityPolicy:
        """Strict security - high protection."""
        return SecurityPolicy(
            max_file_size=5_000_000,  # 5MB
            max_memory_usage=500_000_000,  # 500MB
            max_processing_time=120,  # 2 minutes
            block_dangerous_functions=True,
            sanitize_html_content=True,
            filter_prompt_injections=True,
            sandbox_type="docker",
            enable_network_isolation=True,
            enable_filesystem_isolation=True,
        )
    
    @staticmethod
    def maximum() -> SecurityPolicy:
        """Maximum security - for untrusted content."""
        return SecurityPolicy(
            max_file_size=1_000_000,  # 1MB
            max_memory_usage=100_000_000,  # 100MB
            max_processing_time=60,  # 1 minute
            block_dangerous_functions=True,
            sanitize_html_content=True,
            filter_prompt_injections=True,
            sandbox_type="docker",
            enable_network_isolation=True,
            enable_filesystem_isolation=True,
            allowed_file_extensions={".json", ".jsonl", ".txt"},
            blocked_directories={"/", "/etc", "/sys", "/proc", "/dev", "/root", "/home", "/usr", "/var"},
        )


# Example usage and testing
def main():
    """Example usage of the security guardrails system."""
    
    # Create security policy
    policy = SecurityPolicies.strict()
    
    # Initialize secure processor
    processor = SecureDatasetProcessor(policy)
    
    # Test content
    test_content = {
        "function": {
            "name": "safe_function",
            "arguments": {"param": "value"}
        },
        "prompt": "This is a safe prompt for testing."
    }
    
    try:
        result = processor.process_content_safely(test_content)
        print("Processing successful:", result["status"])
        print("Method:", result["processing_method"])
    
    except SecurityViolationError as e:
        print(f"Security violation: {e.message}")
        print(f"Type: {e.violation_type}")
        print(f"Severity: {e.severity}")
    
    except Exception as e:
        print(f"Processing error: {e}")


if __name__ == "__main__":
    main()