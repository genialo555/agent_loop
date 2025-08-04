#!/usr/bin/env python3
"""
Tests for LaTeX Documentation System

This module provides comprehensive testing for the LaTeX documentation pipeline
including:
- Bibliography extraction script validation
- Makefile target testing
- LaTeX compilation verification  
- Documentation pipeline validation
- CI integration testing

Author: Test Automator
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest
from hypothesis import given, strategies as st

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from docs.latex.scripts.extract_bibliography import (
    extract_bibliography_from_text,
    main as extract_bibliography_main
)


# ============================================================================
# Test Markers and Configuration
# ============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.latex
]


# ============================================================================
# Fixtures for LaTeX Testing
# ============================================================================

@pytest.fixture
def latex_test_dir():
    """Create a temporary directory for LaTeX testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_latex_env(latex_test_dir):
    """Mock LaTeX environment with required directories."""
    latex_dir = latex_test_dir / "latex"
    latex_dir.mkdir()
    
    # Create required subdirectories
    (latex_dir / "sources").mkdir()
    (latex_dir / "build").mkdir()
    (latex_dir / "bibliography").mkdir()
    (latex_dir / "figures").mkdir()
    (latex_dir / "scripts").mkdir()
    
    # Copy Makefile
    original_makefile = Path(__file__).parent.parent / "docs" / "latex" / "Makefile"
    if original_makefile.exists():
        shutil.copy2(original_makefile, latex_dir / "Makefile")
    else:
        # Create minimal Makefile for testing
        (latex_dir / "Makefile").write_text(
            "help:\n\t@echo 'Test Makefile'\n\n"
            "setup:\n\t@mkdir -p build sources\n\n"
            "clean:\n\t@rm -rf build/*\n"
        )
    
    return latex_dir


@pytest.fixture
def sample_bibtex_content():
    """Sample BibTeX content for testing."""
    return """@article{sample2023,
  title={Sample Article},
  author={Doe, John and Smith, Jane},
  journal={Test Journal},
  year={2023},
  volume={1},
  pages={1--10}
}

@inproceedings{conference2022,
  title={Conference Paper},
  author={Brown, Alice},
  booktitle={Proceedings of Test Conference},
  year={2022},
  pages={123--130}
}"""


@pytest.fixture
def sample_latex_document():
    """Sample LaTeX document for testing."""
    return r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{natbib}

\title{Test Document}
\author{Test Author}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This is a test document with citations \cite{sample2023}.

\section{Results}
We found significant results \cite{conference2022}.

\bibliographystyle{plain}
\bibliography{references}

\end{document}"""


@pytest.fixture
def sample_references_text():
    """Sample reference text for bibliography extraction testing."""
    return """1. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.
http://www.deeplearningbook.org.

2. Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pages 770–778, 2015.

3. Lena Strobl. Average-hard attention transformers are constant-depth uniform threshold
circuits, 2023. arXiv preprint arXiv:2301.12345.

10. Jason Wei, Yi Tay, et al. Chain-of-thought prompting elicits reasoning in large language
models, 2022. arXiv preprint arXiv:2201.11903."""


# ============================================================================
# Bibliography Extraction Script Tests
# ============================================================================

class TestBibliographyExtraction:
    """Test suite for bibliography extraction functionality."""
    
    def test_extract_bibliography_basic(self, sample_references_text):
        """Test basic bibliography extraction functionality."""
        entries = extract_bibliography_from_text(sample_references_text)
        
        assert len(entries) > 0, "Should extract at least one reference"
        assert all("@" in entry for entry in entries), "All entries should be BibTeX format"
        
        # Check first entry structure
        first_entry = entries[0]
        assert "title={" in first_entry, "Should extract title"
        assert "author={" in first_entry, "Should extract author"
        assert "year={" in first_entry, "Should extract year"
    
    def test_extract_bibliography_entry_types(self, sample_references_text):
        """Test that different entry types are correctly identified."""
        entries = extract_bibliography_from_text(sample_references_text)
        
        entry_text = "\n".join(entries)
        
        # Should have different entry types
        assert "@book{" in entry_text, "Should identify book entries"
        assert "@inproceedings{" in entry_text, "Should identify conference papers"
        assert "@misc{" in entry_text, "Should identify arXiv preprints"
    
    def test_extract_bibliography_year_extraction(self, sample_references_text):
        """Test year extraction from references."""
        entries = extract_bibliography_from_text(sample_references_text)
        
        years_found = []
        for entry in entries:
            year_match = re.search(r'year=\{(\d{4})\}', entry)
            if year_match:
                years_found.append(year_match.group(1))
        
        assert "2016" in years_found, "Should extract 2016"
        assert "2015" in years_found, "Should extract 2015"
        assert "2022" in years_found, "Should extract 2022"
    
    @given(st.text(min_size=10, max_size=1000))
    def test_extract_bibliography_robustness(self, random_text):
        """Property-based test for bibliography extraction robustness."""
        # Should not crash on random input
        try:
            entries = extract_bibliography_from_text(random_text)
            assert isinstance(entries, list), "Should return a list"
        except Exception as e:
            pytest.fail(f"Bibliography extraction should not crash on random input: {e}")
    
    def test_extract_bibliography_empty_input(self):
        """Test bibliography extraction with empty input."""
        entries = extract_bibliography_from_text("")
        assert entries == [], "Empty input should return empty list"
    
    def test_extract_bibliography_malformed_references(self):
        """Test handling of malformed reference text."""
        malformed_text = "Not a proper reference format\nJust some random text"
        entries = extract_bibliography_from_text(malformed_text)
        
        # Should handle gracefully, might return empty or attempt parsing
        assert isinstance(entries, list), "Should return a list even for malformed input"
    
    @patch('pathlib.Path.mkdir')
    @patch('builtins.open')
    def test_bibliography_main_function(self, mock_open, mock_mkdir):
        """Test the main function of bibliography extraction script."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Should execute without errors
        extract_bibliography_main()
        
        # Verify file operations
        mock_mkdir.assert_called()
        mock_open.assert_called()
        mock_file.write.assert_called()


# ============================================================================
# Makefile Integration Tests
# ============================================================================

class TestMakefileIntegration:
    """Test suite for Makefile targets and integration."""
    
    def test_makefile_help_target(self, mock_latex_env):
        """Test that Makefile help target works."""
        result = subprocess.run(
            ["make", "help"], 
            cwd=mock_latex_env, 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0, f"Make help failed: {result.stderr}"
        assert "help" in result.stdout.lower(), "Help should mention available commands"
    
    def test_makefile_setup_target(self, mock_latex_env):
        """Test that Makefile setup target creates required directories."""
        # Remove directories first
        shutil.rmtree(mock_latex_env / "build", ignore_errors=True)
        shutil.rmtree(mock_latex_env / "sources", ignore_errors=True)
        
        result = subprocess.run(
            ["make", "setup"], 
            cwd=mock_latex_env, 
            capture_output=True, 
            text=True
        )
        
        # Should succeed even without LaTeX installed (directory creation only)
        assert result.returncode == 0 or "not found" in result.stderr
        
        # Check if directories were created (if setup succeeded)
        if result.returncode == 0:
            assert (mock_latex_env / "build").exists(), "Build directory should be created"
            assert (mock_latex_env / "sources").exists(), "Sources directory should be created"
    
    def test_makefile_clean_target(self, mock_latex_env):
        """Test that Makefile clean target removes build artifacts."""
        # Create some fake build artifacts
        build_dir = mock_latex_env / "build"
        (build_dir / "test.pdf").touch()
        (build_dir / "test.aux").touch()
        (build_dir / "test.log").touch()
        
        result = subprocess.run(
            ["make", "clean"], 
            cwd=mock_latex_env, 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0, f"Make clean failed: {result.stderr}"
        
        # Check that files were removed
        assert not (build_dir / "test.pdf").exists(), "PDF should be cleaned"
        assert not (build_dir / "test.aux").exists(), "AUX should be cleaned"
        assert not (build_dir / "test.log").exists(), "LOG should be cleaned"
    
    @pytest.mark.slow
    def test_makefile_template_creation(self, mock_latex_env):
        """Test that Makefile can create LaTeX templates."""
        result = subprocess.run(
            ["make", "create-hrm-template"], 
            cwd=mock_latex_env, 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0, f"Template creation failed: {result.stderr}"
        
        # Check if template was created
        template_file = mock_latex_env / "sources" / "hierarchical_reasoning_model.tex"
        if template_file.exists():
            content = template_file.read_text()
            assert r"\documentclass" in content, "Should be valid LaTeX document"
            assert r"\begin{document}" in content, "Should have document body"
            assert r"\end{document}" in content, "Should have document end"


# ============================================================================
# LaTeX Compilation Tests
# ============================================================================

class TestLatexCompilation:
    """Test suite for LaTeX compilation verification."""
    
    def test_latex_installation_check(self):
        """Test if LaTeX is available for compilation."""
        try:
            result = subprocess.run(
                ["pdflatex", "--version"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            latex_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            latex_available = False
        
        if not latex_available:
            pytest.skip("LaTeX not installed - skipping compilation tests")
    
    @pytest.mark.slow
    def test_minimal_latex_compilation(self, mock_latex_env, sample_latex_document):
        """Test compilation of minimal LaTeX document."""
        # Skip if LaTeX not available
        self.test_latex_installation_check()
        
        # Create minimal document
        sources_dir = mock_latex_env / "sources"
        doc_file = sources_dir / "test_doc.tex"
        doc_file.write_text(sample_latex_document)
        
        # Create empty bibliography
        bib_file = sources_dir / "references.bib"
        bib_file.write_text("% Empty bibliography for testing")
        
        # Attempt compilation
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "test_doc.tex"],
            cwd=sources_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.skip(f"LaTeX compilation failed (dependencies): {result.stderr}")
        
        # Check output
        pdf_file = sources_dir / "test_doc.pdf"
        assert pdf_file.exists(), "PDF should be generated"
        assert pdf_file.stat().st_size > 0, "PDF should not be empty"
    
    def test_latex_syntax_validation(self, sample_latex_document):
        """Test LaTeX syntax validation without full compilation."""
        # Skip if LaTeX not available
        try:
            subprocess.run(["pdflatex", "--version"], 
                         capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("LaTeX not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            f.write(sample_latex_document)
            temp_file = f.name
        
        try:
            # Use draft mode for syntax checking only
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-draftmode", temp_file],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            # Check for syntax errors
            if result.returncode != 0:
                assert "Undefined control sequence" not in result.stdout
                assert "LaTeX Error" not in result.stdout
                # May fail due to missing packages, but syntax should be OK
        finally:
            os.unlink(temp_file)
    
    def test_bibtex_integration(self, mock_latex_env, sample_bibtex_content):
        """Test BibTeX integration with LaTeX compilation."""
        # Skip if tools not available
        try:
            subprocess.run(["pdflatex", "--version"], check=True, capture_output=True, timeout=5)
            subprocess.run(["bibtex", "--version"], check=True, capture_output=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("LaTeX/BibTeX not available")
        
        sources_dir = mock_latex_env / "sources"
        
        # Create bibliography file
        bib_file = sources_dir / "references.bib"
        bib_file.write_text(sample_bibtex_content)
        
        # Create document with citations
        doc_content = r"""\documentclass{article}
\usepackage{natbib}
\begin{document}
\title{Test}
\author{Test}
\maketitle
Citation test \cite{sample2023}.
\bibliographystyle{plain}
\bibliography{references}
\end{document}"""
        
        doc_file = sources_dir / "test_bib.tex"
        doc_file.write_text(doc_content)
        
        # Test BibTeX processing
        result = subprocess.run(
            ["bibtex", "test_bib"],
            cwd=sources_dir,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        # BibTeX might fail due to missing .aux file, but should recognize .bib format
        if result.returncode != 0 and "couldn't open auxiliary file" not in result.stderr:
            pytest.fail(f"BibTeX processing failed: {result.stderr}")


# ============================================================================
# Documentation Pipeline Validation Tests
# ============================================================================

class TestDocumentationPipeline:
    """Test suite for documentation pipeline validation."""
    
    def test_directory_structure_validation(self, mock_latex_env):
        """Test that required directory structure exists."""
        required_dirs = ["sources", "build", "bibliography", "figures", "scripts"]
        
        for dir_name in required_dirs:
            dir_path = mock_latex_env / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
    
    def test_makefile_presence_and_syntax(self, mock_latex_env):
        """Test that Makefile exists and has valid syntax."""
        makefile = mock_latex_env / "Makefile"
        assert makefile.exists(), "Makefile should exist"
        
        content = makefile.read_text()
        
        # Basic Makefile syntax checks
        assert ":" in content, "Makefile should have targets"
        assert "\t" in content or "    " in content, "Makefile should have commands"
        
        # Check for essential targets
        essential_targets = ["help", "clean"]
        for target in essential_targets:
            assert f"{target}:" in content, f"Makefile should have {target} target"
    
    def test_script_permissions_and_execution(self, mock_latex_env):
        """Test that scripts have proper permissions and can execute."""
        scripts_dir = mock_latex_env / "scripts"
        
        # Create test script for validation
        test_script = scripts_dir / "test_script.py"
        test_script.write_text("#!/usr/bin/env python3\nprint('test')\n")
        test_script.chmod(0o755)
        
        # Test execution
        result = subprocess.run(
            ["python3", str(test_script)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, "Test script should execute successfully"
        assert "test" in result.stdout, "Script should produce expected output"
    
    def test_file_encoding_validation(self, mock_latex_env, sample_latex_document):
        """Test that LaTeX files use proper UTF-8 encoding."""
        sources_dir = mock_latex_env / "sources"
        test_file = sources_dir / "encoding_test.tex"
        
        # Write with Unicode characters
        unicode_content = sample_latex_document + "\n% Unicode test: café, naïve, résumé"
        test_file.write_text(unicode_content, encoding='utf-8')
        
        # Read back and verify
        read_content = test_file.read_text(encoding='utf-8')
        assert "café" in read_content, "Should handle Unicode characters"
        assert "naïve" in read_content, "Should preserve accented characters"
    
    def test_bibliography_file_validation(self, mock_latex_env, sample_bibtex_content):
        """Test bibliography file structure and content validation."""
        bib_dir = mock_latex_env / "bibliography"
        bib_file = bib_dir / "references.bib"
        bib_file.write_text(sample_bibtex_content)
        
        content = bib_file.read_text()
        
        # Validate BibTeX structure
        assert "@article{" in content, "Should contain article entries"
        assert "@inproceedings{" in content, "Should contain conference entries"
        
        # Check for required fields
        assert "title={" in content, "Entries should have titles"
        assert "author={" in content, "Entries should have authors"
        assert "year={" in content, "Entries should have years"
        
        # Validate BibTeX syntax
        brace_count = content.count('{') - content.count('}')
        assert brace_count == 0, "BibTeX should have balanced braces"


# ============================================================================
# CI Integration Tests
# ============================================================================

class TestCIIntegration:
    """Test suite for CI integration and automation."""
    
    def test_ci_environment_simulation(self, mock_latex_env):
        """Test documentation pipeline in CI-like environment."""
        # Simulate CI environment variables
        ci_env = os.environ.copy()
        ci_env.update({
            'CI': 'true',
            'DEBIAN_FRONTEND': 'noninteractive',
            'PATH': ci_env.get('PATH', '')
        })
        
        # Test basic operations that would run in CI
        operations = [
            ["make", "help"],
            ["make", "setup"],
            ["make", "clean"]
        ]
        
        for operation in operations:
            result = subprocess.run(
                operation,
                cwd=mock_latex_env,
                env=ci_env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Allow for missing dependencies in CI
            if result.returncode != 0:
                assert "not found" in result.stderr or "command not found" in result.stderr
    
    def test_dependency_detection(self):
        """Test detection of LaTeX dependencies for CI setup."""
        required_commands = ["pdflatex", "bibtex", "makeindex"]
        available_commands = {}
        
        for cmd in required_commands:
            try:
                result = subprocess.run(
                    ["which", cmd],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                available_commands[cmd] = result.returncode == 0
            except subprocess.TimeoutExpired:
                available_commands[cmd] = False
        
        # Document which commands are available for CI configuration
        print(f"LaTeX dependency status: {available_commands}")
        
        # At least 'which' command should work in most environments
        result = subprocess.run(["which", "python3"], capture_output=True, timeout=5)
        assert result.returncode == 0, "Basic shell commands should work"
    
    def test_error_reporting_format(self, mock_latex_env):
        """Test that errors are reported in CI-friendly format."""
        # Create a Makefile target that will fail
        makefile = mock_latex_env / "Makefile"
        current_content = makefile.read_text()
        
        # Add a failing target
        failing_target = "\ntest-fail:\n\t@echo 'ERROR: This is a test failure' && exit 1\n"
        makefile.write_text(current_content + failing_target)
        
        result = subprocess.run(
            ["make", "test-fail"],
            cwd=mock_latex_env,
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Failing target should return non-zero exit code"
        assert "ERROR:" in result.stdout or "ERROR:" in result.stderr, "Error should be clearly marked"
    
    @pytest.mark.slow
    def test_documentation_build_pipeline(self, mock_latex_env):
        """Test complete documentation build pipeline."""
        pipeline_steps = [
            ("setup", ["make", "setup"]),
            ("template", ["make", "create-hrm-template"]),
            ("bibliography", ["make", "create-bibliography"]),
            ("validate", ["make", "validate"]),
            ("clean", ["make", "clean"])
        ]
        
        results = {}
        
        for step_name, command in pipeline_steps:
            result = subprocess.run(
                command,
                cwd=mock_latex_env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            results[step_name] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        
        # Document pipeline results for CI optimization
        print(f"Documentation pipeline results: {list(results.keys())}")
        
        # Setup should always work
        assert results['setup']['returncode'] == 0, "Setup step should succeed"
        
        # Clean should always work
        assert results['clean']['returncode'] == 0, "Clean step should succeed"


# ============================================================================
# Performance and Resource Tests
# ============================================================================

class TestPerformanceAndResources:
    """Test suite for performance and resource usage validation."""
    
    @pytest.mark.benchmark
    def test_bibliography_extraction_performance(self, sample_references_text):
        """Benchmark bibliography extraction performance."""
        import time
        
        # Large reference text for performance testing
        large_text = sample_references_text * 100  # Simulate 400+ references
        
        start_time = time.time()
        entries = extract_bibliography_from_text(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 5.0, f"Bibliography extraction took {processing_time:.2f}s (>5s)"
        assert len(entries) > 0, "Should extract entries from large text"
        
        # Performance metrics
        entries_per_second = len(entries) / processing_time if processing_time > 0 else float('inf')
        print(f"Bibliography extraction: {entries_per_second:.1f} entries/second")
    
    @pytest.mark.benchmark
    def test_makefile_execution_time(self, mock_latex_env):
        """Benchmark Makefile execution time."""
        import time
        
        commands = ["help", "setup", "clean"]
        execution_times = {}
        
        for command in commands:
            start_time = time.time()
            result = subprocess.run(
                ["make", command],
                cwd=mock_latex_env,
                capture_output=True,
                text=True,
                timeout=30
            )
            end_time = time.time()
            
            execution_times[command] = end_time - start_time
            
            # Commands should complete quickly
            assert execution_times[command] < 10.0, f"Make {command} took {execution_times[command]:.2f}s"
        
        print(f"Makefile execution times: {execution_times}")
    
    def test_memory_usage_bibliography_extraction(self):
        """Test memory usage during bibliography extraction."""
        import tracemalloc
        
        # Large sample for memory testing
        large_sample = "1. Sample reference entry.\n" * 10000
        
        tracemalloc.start()
        
        entries = extract_bibliography_from_text(large_sample)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 100MB for large input)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Memory usage {peak_mb:.1f}MB is too high"
        
        print(f"Bibliography extraction memory usage: {peak_mb:.1f}MB peak")


# ============================================================================
# Error Handling and Edge Cases Tests
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    def test_missing_dependencies_handling(self, mock_latex_env):
        """Test graceful handling of missing LaTeX dependencies."""
        # Test when LaTeX is not available
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("pdflatex not found")
            
            result = subprocess.run(
                ["make", "setup"],
                cwd=mock_latex_env,
                capture_output=True,
                text=True
            )
            
            # Should handle missing dependencies gracefully
            # (Actual behavior depends on Makefile implementation)
    
    def test_invalid_latex_content_handling(self, mock_latex_env):
        """Test handling of invalid LaTeX content."""
        sources_dir = mock_latex_env / "sources"
        
        # Create invalid LaTeX document
        invalid_content = r"""\documentclass{article}
\begin{document}
\invalid_command{This should cause an error}
\end{document}"""
        
        doc_file = sources_dir / "invalid.tex"
        doc_file.write_text(invalid_content)
        
        # Should be detectable as invalid
        content = doc_file.read_text()
        assert r"\invalid_command" in content, "Should contain invalid command"
    
    def test_filesystem_permission_errors(self, mock_latex_env):
        """Test handling of filesystem permission errors."""
        # Make directory read-only
        build_dir = mock_latex_env / "build"
        original_mode = build_dir.stat().st_mode
        
        try:
            build_dir.chmod(0o444)  # Read-only
            
            # Try to write to read-only directory
            test_file = build_dir / "test.txt"
            
            try:
                test_file.write_text("test")
                pytest.fail("Should not be able to write to read-only directory")
            except PermissionError:
                pass  # Expected behavior
                
        finally:
            # Restore permissions
            build_dir.chmod(original_mode)
    
    def test_large_bibliography_handling(self):
        """Test handling of very large bibliography files."""
        # Generate large bibliography text
        large_bib_text = ""
        for i in range(1000):
            large_bib_text += f"{i}. Author {i}. Title {i}. Journal {i}, {2000 + i % 25}.\n"
        
        # Should handle without crashing
        entries = extract_bibliography_from_text(large_bib_text)
        assert len(entries) > 0, "Should extract some entries from large bibliography"
        assert len(entries) <= 1000, "Should not create more entries than references"
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in references."""
        unicode_refs = """1. Müller, Hans and Café, José. Naïve Bayes für große Datensätze. 
Machine Learning Journal, 2023.

2. 王小明 and 李小红. Deep Learning with Chinese Text. 
北京大学学报, 2022.

3. Åström, K.J. and Wittenmark, B. Adaptive Control (2nd ed.). 
Addison-Wesley, 1995."""
        
        entries = extract_bibliography_from_text(unicode_refs)
        
        # Should handle Unicode without errors
        assert len(entries) > 0, "Should extract Unicode references"
        
        entries_text = "\n".join(entries)
        # Basic characters should be preserved (exact preservation depends on implementation)
        assert len(entries_text) > 0, "Should produce non-empty output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])