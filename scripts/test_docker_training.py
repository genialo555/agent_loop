#!/usr/bin/env python3
"""
Test script for Docker Training Architecture - Sprint 2
Validates GPU setup, container configuration, and QLoRA pipeline readiness.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class DockerTrainingTester:
    """Test suite for Docker training infrastructure."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.project_root = Path(__file__).parent.parent
        
    def log_test(self, test_name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Log test result."""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
        print()
    
    def run_command(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute shell command and return success, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def test_docker_setup(self):
        """Test basic Docker setup and GPU support."""
        print("🐳 Testing Docker Setup...")
        
        # Test Docker availability
        success, stdout, stderr = self.run_command(["docker", "--version"])
        if not success:
            self.log_test("docker_availability", False, "Docker not installed or not accessible", 
                         {"error": stderr})
            return
        
        docker_version = stdout.strip()
        self.log_test("docker_availability", True, f"Docker available: {docker_version}")
        
        # Test Docker Compose
        success, stdout, stderr = self.run_command(["docker", "compose", "version"])
        if not success:
            self.log_test("docker_compose", False, "Docker Compose v2 not available", 
                         {"error": stderr})
            return
        
        compose_version = stdout.strip()
        self.log_test("docker_compose", True, f"Docker Compose available: {compose_version}")
        
        # Test GPU Support
        success, stdout, stderr = self.run_command(["docker", "run", "--rm", "--gpus", "all", 
                                                   "nvidia/cuda:12.6.0-base-ubuntu22.04", "nvidia-smi"])
        if not success:
            self.log_test("gpu_support", False, "GPU support not available in Docker", 
                         {"error": stderr, "hint": "Install nvidia-container-toolkit"})
        else:
            gpu_info = stdout.strip().split('\n')[0] if stdout.strip() else "GPU detected"
            self.log_test("gpu_support", True, f"GPU support working: {gpu_info}")
    
    def test_training_files(self):
        """Test training configuration files exist and are valid."""
        print("📁 Testing Training Files...")
        
        required_files = [
            "docker-compose.training.yml",
            "training/qlora_finetune.py", 
            "training/qlora_config.py",
            "requirements-training.txt",
            ".dockerignore"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_test(f"file_{file_path.replace('/', '_')}", True, 
                             f"Required file exists: {file_path}")
            else:
                self.log_test(f"file_{file_path.replace('/', '_')}", False, 
                             f"Required file missing: {file_path}")
        
        # Test docker-compose.training.yml syntax
        success, _, stderr = self.run_command(["docker", "compose", "-f", "docker-compose.training.yml", "config"])
        if success:
            self.log_test("compose_training_syntax", True, "docker-compose.training.yml syntax valid")
        else:
            self.log_test("compose_training_syntax", False, "docker-compose.training.yml syntax invalid", 
                         {"error": stderr})
    
    def test_volume_directories(self):
        """Test volume directory structure."""
        print("📂 Testing Volume Structure...")
        
        required_dirs = [
            "models",
            "models/gguf", 
            "models/.cache",
            "model_checkpoints",
            "logs",
            "backups/training"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_test(f"dir_{dir_path.replace('/', '_')}", True, 
                             f"Volume directory exists: {dir_path}")
            else:
                # Create missing directories
                full_path.mkdir(parents=True, exist_ok=True)
                self.log_test(f"dir_{dir_path.replace('/', '_')}", True, 
                             f"Volume directory created: {dir_path}")
    
    def test_dockerfile_stages(self):
        """Test Dockerfile multi-stage build."""
        print("🏗️ Testing Dockerfile Stages...")
        
        # Test training stage build
        success, stdout, stderr = self.run_command([
            "docker", "build", "--target", "training", "-t", "agent-loop-training-test", "."
        ], timeout=300)  # 5 minute timeout for builds
        
        if success:
            self.log_test("dockerfile_training_stage", True, "Training stage builds successfully")
            
            # Cleanup test image
            self.run_command(["docker", "rmi", "agent-loop-training-test"])
        else:
            self.log_test("dockerfile_training_stage", False, "Training stage build failed", 
                         {"error": stderr})
    
    def test_makefile_commands(self):
        """Test Makefile training commands exist."""
        print("📋 Testing Makefile Commands...")
        
        # Read Makefile and check for training commands
        makefile_path = self.project_root / "Makefile"
        if not makefile_path.exists():
            self.log_test("makefile_exists", False, "Makefile not found")
            return
        
        makefile_content = makefile_path.read_text()
        
        required_commands = [
            "train-docker",
            "train-dev", 
            "train-stop",
            "train-status",
            "train-gemma-2b",
            "train-gemma-9b"
        ]
        
        for command in required_commands:
            if f"{command}:" in makefile_content:
                self.log_test(f"makefile_{command.replace('-', '_')}", True, 
                             f"Makefile command exists: {command}")
            else:
                self.log_test(f"makefile_{command.replace('-', '_')}", False, 
                             f"Makefile command missing: {command}")
    
    def test_dependencies(self):
        """Test training dependencies in requirements-training.txt."""
        print("📦 Testing Training Dependencies...")
        
        req_file = self.project_root / "requirements-training.txt"
        if not req_file.exists():
            self.log_test("requirements_training", False, "requirements-training.txt not found")
            return
        
        content = req_file.read_text()
        
        critical_deps = [
            "torch",
            "transformers", 
            "peft",
            "bitsandbytes",
            "trl",
            "wandb"
        ]
        
        for dep in critical_deps:
            if dep in content:
                # Extract version if specified
                lines = [line for line in content.split('\n') if line.startswith(dep)]
                version_info = lines[0] if lines else dep
                self.log_test(f"dep_{dep}", True, f"Dependency found: {version_info}")
            else:
                self.log_test(f"dep_{dep}", False, f"Critical dependency missing: {dep}")
    
    def test_training_config(self):
        """Test QLoRA configuration validity."""
        print("⚙️ Testing QLoRA Configuration...")
        
        try:
            # Import and validate config (this tests Python syntax)
            sys.path.append(str(self.project_root))
            from training.qlora_config import QLoRAConfig, GEMMA_2B_CONFIG, GEMMA_9B_CONFIG
            
            # Test default config
            config = QLoRAConfig()
            model_info = config.get_model_info()
            
            self.log_test("qlora_config_import", True, "QLoRA configuration imports successfully")
            self.log_test("qlora_config_valid", True, "QLoRA configuration valid", 
                         {"model": model_info.get("model_name"), 
                          "quantization": model_info.get("quantization")})
            
            # Test predefined configs
            configs_to_test = {
                "gemma_2b": GEMMA_2B_CONFIG,
                "gemma_9b": GEMMA_9B_CONFIG
            }
            
            for name, config in configs_to_test.items():
                try:
                    bnb_config = config.get_bnb_config()
                    lora_config = config.get_lora_config()
                    training_args = config.get_training_args("/tmp/test")
                    
                    self.log_test(f"config_{name}", True, f"Predefined config {name} valid")
                except Exception as e:
                    self.log_test(f"config_{name}", False, f"Predefined config {name} invalid: {e}")
                    
        except ImportError as e:
            self.log_test("qlora_config_import", False, f"Cannot import QLoRA config: {e}")
        except Exception as e:
            self.log_test("qlora_config_error", False, f"QLoRA config error: {e}")
    
    def test_integration_dry_run(self):
        """Test training pipeline dry run (without actual training)."""
        print("🧪 Testing Training Pipeline Integration...")
        
        # Test training script dry run
        success, stdout, stderr = self.run_command([
            "python", "training/qlora_finetune.py", 
            "--dry-run",
            "--model-config", "gemma-2b",
            "--data", "/tmp/fake-data",
            "--no-wandb"
        ], timeout=60)
        
        if success:
            self.log_test("training_dry_run", True, "Training pipeline dry run successful")
        else:
            self.log_test("training_dry_run", False, "Training pipeline dry run failed", 
                         {"error": stderr})
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("🏁 DOCKER TRAINING ARCHITECTURE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["passed"]])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Test Summary:")
        print(f"   Total: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📈 Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n🔍 Failed Tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"   • {result['test']}: {result['message']}")
                    if result.get("details", {}).get("error"):
                        print(f"     Error: {result['details']['error']}")
        
        # Save detailed report
        report_file = self.project_root / "test_results_docker_training.json"
        with open(report_file, "w") as f:
            json.dump({
                "summary": {
                    "total": total_tests,
                    "passed": passed_tests, 
                    "failed": failed_tests,
                    "success_rate": passed_tests/total_tests*100
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": self.results
            }, f, indent=2)
            
        print(f"\n📋 Detailed report saved: {report_file}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! Docker training architecture is ready.")
            return True
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Review and fix issues before training.")
            return False
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 Starting Docker Training Architecture Tests\n")
        
        self.test_docker_setup()
        self.test_training_files()
        self.test_volume_directories()
        self.test_dockerfile_stages()
        self.test_makefile_commands()
        self.test_dependencies()
        self.test_training_config()
        self.test_integration_dry_run()
        
        return self.generate_report()


def main():
    """Main test execution."""
    tester = DockerTrainingTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()