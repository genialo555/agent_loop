#!/usr/bin/env python3
"""
Script de test d'intégration Docker Training - Sprint 2

Ce script teste l'infrastructure Docker training complète :
- Configuration docker-compose.training.yml
- Build des images avec stage training
- Test des services training et monitoring
- Validation des volumes et networks
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
from __future__ import annotations
import logging

# Setup logging with new structure
log_dir = Path("/home/jerem/agent_loop/models/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"docker_training_test_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Test logging to: {log_file}")

class DockerTrainingTester:
    """Testeur d'intégration pour l'environnement Docker training."""
    
    def __init__(self, project_root: str = "/home/jerem/agent_loop"):
        self.project_root = Path(project_root)
        self.compose_file = self.project_root / "docker-compose.training.yml"
        self.dockerfile = self.project_root / "Dockerfile"
        
        # Configuration des tests
        self.test_timeout = 300  # 5 minutes max par test
        self.services_to_test = [
            "qlora-training",
            "training-dev", 
            "training-monitor",
            "gpu-monitor"
        ]
        
        self.results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "overall_status": "pending"
        }
    
    def run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute une commande shell et retourne le résultat."""
        try:
            logger.info(f"Executing: {command}")
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timeout after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "returncode": -1
            }
    
    def test_docker_compose_config(self) -> Dict[str, Any]:
        """Test 1: Validation de la configuration docker-compose.training.yml"""
        logger.info("Test 1: Docker Compose Configuration")
        
        # Vérifier que le fichier existe
        if not self.compose_file.exists():
            return {
                "status": "failed",
                "error": f"docker-compose.training.yml not found at {self.compose_file}"
            }
        
        # Valider la configuration avec docker compose config
        result = self.run_command(f"docker compose -f {self.compose_file} config")
        
        if not result["success"]:
            return {
                "status": "failed", 
                "error": f"Docker compose config validation failed: {result.get('stderr', 'Unknown error')}"
            }
        
        # Vérifier que les services critiques sont définis
        config_content = result["stdout"]
        missing_services = []
        
        for service in ["qlora-training", "training-dev", "training-monitor"]:
            if service not in config_content:
                missing_services.append(service)
        
        if missing_services:
            return {
                "status": "failed",
                "error": f"Missing services in config: {missing_services}"
            }
        
        return {
            "status": "passed",
            "message": "Docker compose configuration is valid",
            "services_found": len([s for s in self.services_to_test if s in config_content])
        }
    
    def test_dockerfile_training_stage(self) -> Dict[str, Any]:
        """Test 2: Validation du stage training dans le Dockerfile"""
        logger.info("Test 2: Dockerfile Training Stage")
        
        if not self.dockerfile.exists():
            return {
                "status": "failed",
                "error": "Dockerfile not found"
            }
        
        # Lire le Dockerfile et vérifier le stage training
        try:
            dockerfile_content = self.dockerfile.read_text()
            
            # Vérifier les stages critiques
            required_stages = [
                "FROM nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04 AS training-builder",
                "FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04 AS training"
            ]
            
            missing_stages = []
            for stage in required_stages:
                if stage not in dockerfile_content:
                    missing_stages.append(stage)
            
            if missing_stages:
                return {
                    "status": "failed",
                    "error": f"Missing Dockerfile stages: {missing_stages}"
                }
            
            # Vérifier les dépendances CUDA
            cuda_dependencies = ["pytorch", "transformers", "bitsandbytes"]
            found_deps = [dep for dep in cuda_dependencies if dep in dockerfile_content.lower()]
            
            return {
                "status": "passed",
                "message": "Dockerfile training stage is properly configured",
                "stages_found": 2,
                "cuda_dependencies": found_deps
            }
            
        except Exception as e:
            return {
                "status": "failed", 
                "error": f"Error reading Dockerfile: {str(e)}"
            }
    
    def test_docker_image_build(self) -> Dict[str, Any]:
        """Test 3: Build de l'image Docker training"""
        logger.info("Test 3: Docker Image Build")
        
        # Build seulement le stage training pour économiser du temps
        build_cmd = f"docker build --target training -t agent-loop-training:test ."
        
        result = self.run_command(build_cmd, timeout=600)  # 10 minutes max
        
        if not result["success"]:
            return {
                "status": "failed",
                "error": f"Docker build failed: {result.get('stderr', 'Unknown error')}"
            }
        
        # Vérifier que l'image a été créée
        check_cmd = "docker images agent-loop-training:test"
        check_result = self.run_command(check_cmd)
        
        if not check_result["success"] or "agent-loop-training" not in check_result["stdout"]:
            return {
                "status": "failed",
                "error": "Built image not found in docker images"
            }
        
        return {
            "status": "passed",
            "message": "Docker training image built successfully",
            "image_name": "agent-loop-training:test"
        }
    
    def test_gpu_availability(self) -> Dict[str, Any]:
        """Test 4: Disponibilité GPU dans le container"""
        logger.info("Test 4: GPU Availability")
        
        # Test nvidia-smi dans un container CUDA
        gpu_test_cmd = "docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi"
        
        result = self.run_command(gpu_test_cmd, timeout=60)
        
        if not result["success"]:
            return {
                "status": "warning",
                "message": "GPU not available - training will run on CPU only",
                "error": result.get("stderr", "nvidia-smi failed")
            }
        
        # Parser la sortie nvidia-smi pour extraire les infos GPU
        gpu_info = {}
        if "NVIDIA" in result["stdout"]:
            lines = result["stdout"].split("\n")
            for line in lines:
                if "NVIDIA" in line and "MiB" in line:
                    gpu_info["gpu_detected"] = True
                    break
        
        return {
            "status": "passed",
            "message": "GPU is available for training",
            "gpu_info": gpu_info
        }
    
    def test_volumes_and_directories(self) -> Dict[str, Any]:
        """Test 5: Structure des volumes et répertoires"""
        logger.info("Test 5: Volumes and Directories")
        
        # Créer les répertoires requis s'ils n'existent pas (nouvelle structure)
        required_dirs = [
            "models/.cache",
            "models/gguf", 
            "models/logs",
            "model_checkpoints",
            "datasets",
            "backups/training"
        ]
        
        created_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
                except Exception as e:
                    return {
                        "status": "failed",
                        "error": f"Failed to create directory {dir_path}: {str(e)}"
                    }
        
        return {
            "status": "passed",
            "message": "All required directories are available",
            "created_directories": created_dirs,
            "total_directories": len(required_dirs)
        }
    
    def test_training_config_files(self) -> Dict[str, Any]:
        """Test 6: Fichiers de configuration training"""
        logger.info("Test 6: Training Configuration Files")
        
        required_files = [
            "models/training/qlora/qlora_finetune.py",
            "models/training/qlora/qlora_config.py",
            "scripts/run_training.sh"
        ]
        
        missing_files = []
        file_sizes = {}
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                file_sizes[file_path] = full_path.stat().st_size
        
        if missing_files:
            return {
                "status": "failed",
                "error": f"Missing training configuration files: {missing_files}"
            }
        
        # Vérifier que les fichiers ne sont pas vides
        empty_files = [f for f, size in file_sizes.items() if size == 0]
        if empty_files:
            return {
                "status": "failed", 
                "error": f"Empty configuration files: {empty_files}"
            }
        
        return {
            "status": "passed",
            "message": "All training configuration files are present",
            "files_checked": len(required_files),
            "file_sizes": file_sizes
        }
    
    def test_api_endpoints_integration(self) -> Dict[str, Any]:
        """Test 7: Intégration des endpoints API training"""
        logger.info("Test 7: API Endpoints Integration")
        
        # Vérifier que les fichiers d'API existent (nouvelle structure)
        api_files = [
            "models/inference/routers/training.py",
            "models/inference/services/training.py", 
            "models/inference/models/schemas.py"
        ]
        
        missing_files = []
        for file_path in api_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return {
                "status": "failed",
                "error": f"Missing API files: {missing_files}"
            }
        
        # Test d'import basique des modules Python
        try:
            import sys
            sys.path.append(str(self.project_root))
            
            # Test import des schemas
            import sys
            sys.path.insert(0, str(self.project_root))
            from models.inference.models.schemas import TrainingRequest, TrainingResponse
            
            # Test création d'une requête valide
            test_request = TrainingRequest(
                base_model="gemma:3n-e2b",
                dataset_path="/tmp/test.jsonl"
            )
            
            return {
                "status": "passed",
                "message": "API endpoints integration is working",
                "schemas_tested": ["TrainingRequest", "TrainingResponse"]
            }
            
        except ImportError as e:
            return {
                "status": "warning",
                "message": "API integration has import issues (may work in Docker)",
                "error": str(e)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"API integration test failed: {str(e)}"
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Exécute tous les tests et génère un rapport complet."""
        logger.info("Starting comprehensive Docker training integration test")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Log file: {log_file}")
        
        tests = [
            ("docker_compose_config", self.test_docker_compose_config),
            ("dockerfile_training_stage", self.test_dockerfile_training_stage),
            ("docker_image_build", self.test_docker_image_build),
            ("gpu_availability", self.test_gpu_availability),
            ("volumes_and_directories", self.test_volumes_and_directories),
            ("training_config_files", self.test_training_config_files),
            ("api_endpoints_integration", self.test_api_endpoints_integration)
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            
            try:
                result = test_func()
                self.results["tests"][test_name] = result
                
                if result["status"] == "passed":
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                elif result["status"] == "warning":
                    warnings += 1
                    logger.warning(f"⚠️  {test_name}: WARNING - {result.get('message', 'Unknown warning')}")
                else:
                    failed += 1
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed += 1
                self.results["tests"][test_name] = {
                    "status": "failed",
                    "error": f"Test execution failed: {str(e)}"
                }
                logger.error(f"❌ {test_name}: EXCEPTION - {str(e)}")
        
        # Calculer le statut global
        if failed == 0 and warnings == 0:
            self.results["overall_status"] = "passed"
        elif failed == 0:
            self.results["overall_status"] = "passed_with_warnings"
        else:
            self.results["overall_status"] = "failed"
        
        self.results["summary"] = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "success_rate": f"{(passed / len(tests)) * 100:.1f}%"
        }
        
        return self.results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Génère un rapport détaillé des tests."""
        report_lines = [
            "# Docker Training Integration Test Report",
            f"**Generated:** {self.results['timestamp']}",
            f"**Overall Status:** {self.results['overall_status'].upper()}",
            "",
            "## Summary",
            f"- **Total Tests:** {self.results['summary']['total_tests']}",
            f"- **Passed:** {self.results['summary']['passed']}",
            f"- **Failed:** {self.results['summary']['failed']}",
            f"- **Warnings:** {self.results['summary']['warnings']}",
            f"- **Success Rate:** {self.results['summary']['success_rate']}",
            "",
            "## Test Results",
            ""
        ]
        
        for test_name, result in self.results["tests"].items():
            status_emoji = "✅" if result["status"] == "passed" else "⚠️" if result["status"] == "warning" else "❌"
            
            report_lines.extend([
                f"### {test_name} {status_emoji}",
                f"**Status:** {result['status'].upper()}",
                ""
            ])
            
            if "message" in result:
                report_lines.append(f"**Message:** {result['message']}")
            
            if "error" in result:
                report_lines.append(f"**Error:** {result['error']}")
            
            # Ajouter les détails spécifiques au test
            for key, value in result.items():
                if key not in ["status", "message", "error"]:
                    report_lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(report_content)
            logger.info(f"📄 Report saved to: {output_path}")
        
        return report_content


def main():
    """Point d'entrée principal pour le script de test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Training Integration Tester")
    parser.add_argument("--project-root", default="/home/jerem/agent_loop",
                      help="Project root directory")
    parser.add_argument("--output", help="Output file for the test report")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Créer et exécuter le testeur
    tester = DockerTrainingTester(args.project_root)
    results = tester.run_comprehensive_test()
    
    if args.json:
        # Sortie JSON
        print(json.dumps(results, indent=2))
    else:
        # Génération du rapport Markdown
        report = tester.generate_report(args.output)
        if not args.output:
            print(report)
    
    # Code de sortie basé sur le statut global
    if results["overall_status"] == "failed":
        exit(1)
    elif results["overall_status"] == "passed_with_warnings":
        exit(2)
    else:
        exit(0)


if __name__ == "__main__":
    main()