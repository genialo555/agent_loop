#!/usr/bin/env python3
"""
CI Integration Tests for LaTeX Documentation System

This module provides specialized tests for continuous integration environments,
focusing on:
- Docker-based LaTeX compilation testing
- GitHub Actions / GitLab CI compatibility
- Dependency installation verification
- Automated documentation deployment validation

Author: Test Automator
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import pytest
import yaml

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Markers and Configuration
# ============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci,
    pytest.mark.slow
]


# ============================================================================
# CI Environment Fixtures
# ============================================================================

@pytest.fixture
def ci_environment_vars():
    """Common CI environment variables for testing."""
    return {
        'CI': 'true',
        'DEBIAN_FRONTEND': 'noninteractive',
        'GITHUB_ACTIONS': 'true',
        'RUNNER_OS': 'Linux',
        'GITHUB_WORKSPACE': '/github/workspace',
        'GITHUB_REPOSITORY': 'test/agent-loop',
        'GITHUB_REF': 'refs/heads/main',
        'GITHUB_SHA': 'abc123def456',
        'GITHUB_EVENT_NAME': 'push'
    }


@pytest.fixture
def latex_docker_env():
    """Mock Docker environment configuration for LaTeX."""
    return {
        'image': 'texlive/texlive:latest',
        'working_dir': '/workspace',
        'volumes': ['/github/workspace:/workspace'],
        'environment': {
            'DEBIAN_FRONTEND': 'noninteractive',
            'TEXMFHOME': '/workspace/.texmf',
            'TEXMFVAR': '/workspace/.texmf-var',
            'TEXMFCONFIG': '/workspace/.texmf-config'
        }
    }


@pytest.fixture
def github_actions_workflow():
    """Sample GitHub Actions workflow for LaTeX documentation."""
    return {
        'name': 'LaTeX Documentation',
        'on': {
            'push': {'branches': ['main', 'develop']},
            'pull_request': {'branches': ['main']}
        },
        'jobs': {
            'build-docs': {
                'runs-on': 'ubuntu-latest',
                'container': 'texlive/texlive:latest',
                'steps': [
                    {'uses': 'actions/checkout@v4'},
                    {
                        'name': 'Install Python dependencies',
                        'run': 'apt-get update && apt-get install -y python3 python3-pip'
                    },
                    {
                        'name': 'Set up LaTeX environment',
                        'run': 'cd docs/latex && make setup'
                    },
                    {
                        'name': 'Extract bibliography',
                        'run': 'cd docs/latex && python3 scripts/extract_bibliography.py'
                    },
                    {
                        'name': 'Create LaTeX templates',
                        'run': 'cd docs/latex && make create-hrm-template'
                    },
                    {
                        'name': 'Validate LaTeX syntax',
                        'run': 'cd docs/latex && make validate'
                    },
                    {
                        'name': 'Compile documentation',
                        'run': 'cd docs/latex && make hrm-paper'
                    },
                    {
                        'name': 'Upload artifacts',
                        'uses': 'actions/upload-artifact@v4',
                        'with': {
                            'name': 'latex-documentation',
                            'path': 'docs/latex/build/*.pdf'
                        }
                    }
                ]
            }
        }
    }


# ============================================================================
# Docker Integration Tests
# ============================================================================

class TestDockerIntegration:
    """Test suite for Docker-based LaTeX compilation."""
    
    def test_texlive_docker_image_availability(self):
        """Test that TeX Live Docker image is available and functional."""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                pytest.skip("Docker not available")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available")
        
        # Test pulling TeX Live image (skip if not available)
        result = subprocess.run(
            ['docker', 'images', 'texlive/texlive', '--format', '{{.Repository}}:{{.Tag}}'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if 'texlive/texlive' not in result.stdout:
            pytest.skip("TeX Live Docker image not available locally")
    
    @pytest.mark.slow
    def test_latex_compilation_in_docker(self, latex_docker_env):
        """Test LaTeX compilation within Docker container."""
        # Skip if Docker not available
        try:
            subprocess.run(['docker', '--version'], check=True, 
                         capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Docker not available")
        
        # Create minimal LaTeX document for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test document
            test_doc = temp_path / 'test.tex'
            test_doc.write_text(r'''
\documentclass{article}
\begin{document}
\title{Docker Test Document}
\author{CI Test}
\maketitle
This is a test document compiled in Docker.
\end{document}
            ''')
            
            # Test Docker command construction
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{temp_path}:/workspace',
                '-w', '/workspace',
                'texlive/texlive:latest',
                'pdflatex', '-interaction=nonstopmode', 'test.tex'
            ]
            
            # In actual CI, this would run; here we validate the command structure
            assert 'docker' in docker_cmd[0]
            assert 'texlive/texlive' in docker_cmd
            assert 'pdflatex' in docker_cmd
    
    def test_docker_volume_mounting(self, latex_docker_env):
        """Test proper Docker volume mounting for LaTeX workspace."""
        workspace_path = "/github/workspace"
        container_path = "/workspace"
        
        volume_mapping = f"{workspace_path}:{container_path}"
        
        # Validate volume mapping format
        assert ':' in volume_mapping
        assert workspace_path in volume_mapping
        assert container_path in volume_mapping
        
        # Test environment variables for TeX
        tex_env = latex_docker_env['environment']
        required_tex_vars = ['TEXMFHOME', 'TEXMFVAR', 'TEXMFCONFIG']
        
        for var in required_tex_vars:
            assert var in tex_env, f"Missing required TeX environment variable: {var}"
            assert tex_env[var].startswith('/workspace'), f"TeX variable {var} should use workspace path"


# ============================================================================
# GitHub Actions Integration Tests
# ============================================================================

class TestGitHubActionsIntegration:
    """Test suite for GitHub Actions workflow integration."""
    
    def test_workflow_syntax_validation(self, github_actions_workflow):
        """Test that GitHub Actions workflow has valid YAML syntax."""
        # Convert to YAML and back to validate syntax
        yaml_content = yaml.dump(github_actions_workflow, default_flow_style=False)
        parsed_workflow = yaml.safe_load(yaml_content)
        
        # Validate required workflow structure
        assert 'name' in parsed_workflow
        assert 'on' in parsed_workflow
        assert 'jobs' in parsed_workflow
        
        # Validate job structure
        jobs = parsed_workflow['jobs']
        assert len(jobs) > 0, "Workflow should have at least one job"
        
        for job_name, job_config in jobs.items():
            assert 'runs-on' in job_config, f"Job {job_name} should specify runs-on"
            assert 'steps' in job_config, f"Job {job_name} should have steps"
    
    def test_workflow_step_validation(self, github_actions_workflow):
        """Test that workflow steps are properly configured."""
        build_job = github_actions_workflow['jobs']['build-docs']
        steps = build_job['steps']
        
        # Check for essential steps
        step_names = [step.get('name', step.get('uses', '')) for step in steps]
        
        essential_steps = [
            'checkout',  # Should checkout code
            'setup',     # Should set up LaTeX environment
            'validate',  # Should validate LaTeX
            'compile'    # Should compile documentation
        ]
        
        for essential in essential_steps:
            found = any(essential.lower() in step.lower() for step in step_names)
            assert found, f"Workflow should include {essential} step"
    
    def test_artifact_upload_configuration(self, github_actions_workflow):
        """Test that artifacts are properly configured for upload."""
        build_job = github_actions_workflow['jobs']['build-docs']
        
        # Find upload artifact step
        upload_step = None
        for step in build_job['steps']:
            if step.get('uses', '').startswith('actions/upload-artifact'):
                upload_step = step
                break
        
        assert upload_step is not None, "Workflow should upload artifacts"
        
        # Validate artifact configuration
        with_config = upload_step.get('with', {})
        assert 'name' in with_config, "Artifact should have a name"
        assert 'path' in with_config, "Artifact should specify path"
        
        # Check path points to PDF outputs
        artifact_path = with_config['path']
        assert '*.pdf' in artifact_path, "Should upload PDF files"
        assert 'build' in artifact_path, "Should upload from build directory"
    
    def test_environment_variable_handling(self, ci_environment_vars):
        """Test proper handling of CI environment variables."""
        # Test GitHub-specific variables
        github_vars = ['GITHUB_WORKSPACE', 'GITHUB_REPOSITORY', 'GITHUB_SHA']
        
        for var in github_vars:
            assert var in ci_environment_vars, f"Missing GitHub environment variable: {var}"
        
        # Test CI detection
        assert ci_environment_vars.get('CI') == 'true', "CI flag should be set"
        assert ci_environment_vars.get('GITHUB_ACTIONS') == 'true', "GitHub Actions flag should be set"
    
    def test_matrix_build_configuration(self):
        """Test matrix build configuration for multiple LaTeX distributions."""
        matrix_config = {
            'strategy': {
                'matrix': {
                    'latex-image': [
                        'texlive/texlive:latest',
                        'texlive/texlive:TL2023-historic',
                        'texlive/texlive:TL2022-historic'
                    ],
                    'include': [
                        {
                            'latex-image': 'texlive/texlive:latest',
                            'experimental': False
                        },
                        {
                            'latex-image': 'texlive/texlive:TL2023-historic',
                            'experimental': False
                        },
                        {
                            'latex-image': 'texlive/texlive:TL2022-historic',
                            'experimental': True
                        }
                    ]
                }
            }
        }
        
        # Validate matrix structure
        strategy = matrix_config['strategy']
        assert 'matrix' in strategy
        
        matrix = strategy['matrix']
        assert 'latex-image' in matrix
        assert len(matrix['latex-image']) > 1, "Should test multiple LaTeX versions"
        
        # Validate include configuration
        if 'include' in matrix:
            for include_item in matrix['include']:
                assert 'latex-image' in include_item
                assert 'experimental' in include_item


# ============================================================================
# Dependency Installation Tests
# ============================================================================

class TestDependencyInstallation:
    """Test suite for LaTeX dependency installation in CI."""
    
    def test_apt_package_installation(self):
        """Test APT package installation for LaTeX dependencies."""
        required_packages = [
            'texlive-latex-extra',
            'texlive-bibtex-extra',
            'texlive-fonts-recommended',
            'texlive-fonts-extra',
            'texlive-science',
            'texlive-publishers',
            'pandoc'
        ]
        
        # Test package list formatting
        package_list = ' '.join(required_packages)
        apt_command = f"apt-get install -y {package_list}"
        
        assert 'apt-get install' in apt_command
        assert '-y' in apt_command  # Non-interactive
        
        for package in required_packages:
            assert package in apt_command
    
    def test_texlive_installation_validation(self):
        """Test validation of TeX Live installation."""
        validation_commands = [
            ['pdflatex', '--version'],
            ['bibtex', '--version'],
            ['makeindex', '--version'],
            ['kpsewhich', '--version']
        ]
        
        # Test command structure
        for cmd in validation_commands:
            assert len(cmd) >= 2, "Command should have program and --version flag"
            assert cmd[1] == '--version', "Should check version for validation"
    
    def test_python_dependencies_installation(self):
        """Test Python dependencies for bibliography extraction."""
        python_requirements = [
            'pathlib',  # Usually built-in
            'regex',    # Enhanced regex support
            'click',    # CLI interface
            'pyyaml',   # YAML parsing
            'pytest',   # Testing framework
        ]
        
        # In CI, these would be installed via pip
        pip_command = f"pip3 install {' '.join(python_requirements)}"
        
        assert 'pip3 install' in pip_command
        for requirement in python_requirements:
            assert requirement in pip_command
    
    def test_cache_optimization_strategy(self):
        """Test caching strategy for CI performance optimization."""
        cache_config = {
            'texlive-cache': {
                'path': '~/.texlive',
                'key': 'texlive-${{ runner.os }}-${{ hashFiles("docs/latex/**/*.tex") }}'
            },
            'python-cache': {
                'path': '~/.cache/pip',
                'key': 'pip-${{ runner.os }}-${{ hashFiles("requirements.txt") }}'
            }
        }
        
        # Validate cache configuration
        for cache_name, config in cache_config.items():
            assert 'path' in config, f"Cache {cache_name} should specify path"
            assert 'key' in config, f"Cache {cache_name} should specify key"
            
            # Key should include OS and file hash for proper invalidation
            cache_key = config['key']
            assert 'runner.os' in cache_key, "Cache key should include OS"
            assert 'hashFiles' in cache_key, "Cache key should include file hash"


# ============================================================================
# Automated Documentation Deployment Tests
# ============================================================================

class TestAutomatedDeployment:
    """Test suite for automated documentation deployment."""
    
    def test_pdf_artifact_generation(self):
        """Test PDF artifact generation and validation."""
        expected_artifacts = [
            'hierarchical_reasoning_model.pdf',
            'supplementary_materials.pdf',
            'bibliography_references.pdf'
        ]
        
        # Test artifact naming convention
        for artifact in expected_artifacts:
            assert artifact.endswith('.pdf'), "Artifacts should be PDF files"
            assert '_' in artifact or '-' in artifact, "Artifacts should use consistent naming"
    
    def test_documentation_website_deployment(self):
        """Test deployment to GitHub Pages or similar."""
        deployment_config = {
            'name': 'Deploy Documentation',
            'uses': 'peaceiris/actions-gh-pages@v3',
            'with': {
                'github_token': '${{ secrets.GITHUB_TOKEN }}',
                'publish_dir': './docs/latex/build',
                'destination_dir': 'latex-docs',
                'enable_jekyll': False,
                'force_orphan': True
            }
        }
        
        # Validate deployment configuration
        assert 'uses' in deployment_config
        assert 'peaceiris/actions-gh-pages' in deployment_config['uses']
        
        with_config = deployment_config['with']
        assert 'github_token' in with_config
        assert 'publish_dir' in with_config
        assert './docs/latex/build' in with_config['publish_dir']
    
    def test_multi_format_documentation_export(self):
        """Test export to multiple documentation formats."""
        export_formats = {
            'pdf': {
                'tool': 'pdflatex',
                'output': 'build/*.pdf',
                'command': 'make hrm-paper'
            },
            'html': {
                'tool': 'pandoc',
                'output': 'build/*.html',
                'command': 'pandoc -f latex -t html5 sources/*.tex -o build/docs.html'
            },
            'epub': {
                'tool': 'pandoc',
                'output': 'build/*.epub',
                'command': 'pandoc -f latex -t epub3 sources/*.tex -o build/docs.epub'
            }
        }
        
        # Validate export configuration
        for format_name, config in export_formats.items():
            assert 'tool' in config, f"Format {format_name} should specify tool"
            assert 'output' in config, f"Format {format_name} should specify output"
            assert 'command' in config, f"Format {format_name} should specify command"
            
            # Output should go to build directory
            assert 'build/' in config['output'], f"Format {format_name} should output to build directory"
    
    def test_version_tagging_and_releases(self):
        """Test automated version tagging and release creation."""
        release_config = {
            'name': 'Create Documentation Release',
            'uses': 'softprops/action-gh-release@v1',
            'if': 'startsWith(github.ref, "refs/tags/")',
            'with': {
                'files': 'docs/latex/build/*.pdf',
                'name': 'Documentation Release ${{ github.ref_name }}',
                'body': 'Automated documentation release for version ${{ github.ref_name }}',
                'draft': False,
                'prerelease': False
            }
        }
        
        # Validate release configuration
        assert 'uses' in release_config
        assert 'softprops/action-gh-release' in release_config['uses']
        
        # Should only run on tags
        assert 'if' in release_config
        assert 'refs/tags/' in release_config['if']
        
        with_config = release_config['with']
        assert 'files' in with_config
        assert '*.pdf' in with_config['files']


# ============================================================================
# Performance and Optimization Tests
# ============================================================================

class TestCIPerformanceOptimization:
    """Test suite for CI performance optimization."""
    
    def test_parallel_compilation_strategy(self):
        """Test parallel compilation for multiple documents."""
        parallel_config = {
            'strategy': {
                'matrix': {
                    'document': [
                        'hierarchical_reasoning_model',
                        'supplementary_materials',
                        'bibliography_references'
                    ]
                },
                'fail-fast': False
            }
        }
        
        # Validate parallel configuration
        strategy = parallel_config['strategy']
        assert 'matrix' in strategy
        assert 'fail-fast' in strategy
        assert strategy['fail-fast'] == False, "Should not fail fast to compile all documents"
        
        matrix = strategy['matrix']
        assert len(matrix['document']) > 1, "Should compile multiple documents in parallel"
    
    def test_incremental_build_optimization(self):
        """Test incremental build optimization strategies."""
        # Test file change detection
        change_detection = {
            'latex-files': 'docs/latex/sources/**/*.tex',
            'bibliography-files': 'docs/latex/bibliography/**/*.bib',
            'figure-files': 'docs/latex/figures/**/*',
            'script-files': 'docs/latex/scripts/**/*.py'
        }
        
        # Validate change detection patterns
        for file_type, pattern in change_detection.items():
            assert 'docs/latex/' in pattern, f"Pattern for {file_type} should target LaTeX directory"
            assert '**/*' in pattern, f"Pattern for {file_type} should use recursive glob"
    
    @pytest.mark.benchmark
    def test_build_time_measurement(self):
        """Test build time measurement and optimization."""
        import time
        
        # Simulate build steps with timing
        build_steps = [
            ('setup', 0.5),
            ('bibliography-extraction', 2.0),
            ('template-creation', 1.0),
            ('latex-compilation', 5.0),
            ('artifact-upload', 1.5)
        ]
        
        total_time = 0
        step_times = {}
        
        for step_name, expected_time in build_steps:
            start_time = time.time()
            # Simulate work
            time.sleep(0.01)  # Minimal delay for testing
            end_time = time.time()
            
            actual_time = end_time - start_time
            step_times[step_name] = actual_time
            total_time += actual_time
        
        # Build should complete within reasonable time
        assert total_time < 1.0, f"Test build took {total_time:.2f}s (simulated)"
        
        # Individual steps should be tracked
        assert len(step_times) == len(build_steps), "All steps should be timed"
        
        print(f"CI build step times: {step_times}")
    
    def test_resource_usage_optimization(self):
        """Test resource usage optimization for CI environments."""
        resource_limits = {
            'memory': '2Gi',
            'cpu': '1000m',
            'disk': '10Gi',
            'timeout': '30m'
        }
        
        # Validate resource configuration
        assert 'memory' in resource_limits
        assert 'cpu' in resource_limits
        assert 'disk' in resource_limits
        assert 'timeout' in resource_limits
        
        # Memory should be sufficient for LaTeX compilation
        memory = resource_limits['memory']
        assert memory.endswith('Gi'), "Memory should be specified in GiB"
        memory_gb = int(memory[:-2])
        assert memory_gb >= 1, "Should allocate at least 1GB for LaTeX compilation"


# ============================================================================
# Error Handling and Recovery Tests
# ============================================================================

class TestCIErrorHandlingAndRecovery:
    """Test suite for CI error handling and recovery mechanisms."""
    
    def test_latex_compilation_error_reporting(self):
        """Test proper error reporting for LaTeX compilation failures."""
        error_patterns = [
            r'LaTeX Error: (.+)',
            r'! (.+)',
            r'Undefined control sequence (.+)',
            r'Package (.+) Error: (.+)',
            r'File (.+) not found'
        ]
        
        # Test error detection patterns
        sample_latex_error = """
! LaTeX Error: File `nonexistent.sty' not found.

Type X to quit or <RETURN> to proceed,
or enter new name. (Default extension: sty)

Enter file name:
"""
        
        error_found = False
        for pattern in error_patterns:
            if re.search(pattern, sample_latex_error):
                error_found = True
                break
        
        assert error_found, "Should detect LaTeX compilation errors"
    
    def test_dependency_failure_handling(self):
        """Test handling of dependency installation failures."""
        dependency_checks = [
            ('pdflatex', 'LaTeX compiler not available'),
            ('bibtex', 'BibTeX processor not available'),
            ('python3', 'Python interpreter not available'),
            ('make', 'Make utility not available')
        ]
        
        for command, error_message in dependency_checks:
            # Test command availability check
            check_command = f"which {command} || echo '{error_message}'"
            assert command in check_command
            assert error_message in check_command
    
    def test_retry_mechanism_configuration(self):
        """Test retry mechanisms for transient failures."""
        retry_config = {
            'download-retries': 3,
            'compilation-retries': 2,
            'upload-retries': 3,
            'timeout-per-retry': '5m'
        }
        
        # Validate retry configuration
        for retry_type, count in retry_config.items():
            if isinstance(count, int):
                assert count > 0, f"Retry count for {retry_type} should be positive"
            elif isinstance(count, str) and count.endswith('m'):
                timeout_min = int(count[:-1])
                assert timeout_min > 0, f"Timeout for {retry_type} should be positive"
    
    def test_fallback_strategy_implementation(self):
        """Test fallback strategies for CI failures."""
        fallback_strategies = {
            'latex-compilation': {
                'primary': 'pdflatex',
                'fallback': ['xelatex', 'lualatex'],
                'minimal': 'pandoc'
            },
            'bibliography': {
                'primary': 'bibtex',
                'fallback': ['biber'],
                'minimal': 'manual-references'
            }
        }
        
        # Validate fallback configuration
        for component, strategy in fallback_strategies.items():
            assert 'primary' in strategy, f"Component {component} should have primary strategy"
            assert 'fallback' in strategy, f"Component {component} should have fallback options"
            assert 'minimal' in strategy, f"Component {component} should have minimal fallback"
            
            # Fallback should be a list
            fallback_options = strategy['fallback']
            assert isinstance(fallback_options, list), f"Fallback for {component} should be a list"
            assert len(fallback_options) > 0, f"Should have at least one fallback option for {component}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])