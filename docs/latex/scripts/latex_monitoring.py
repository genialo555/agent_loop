#!/usr/bin/env python3
"""
LaTeX Compilation Monitoring Script
Provides Prometheus metrics and structured logging for LaTeX builds.
"""

import os
import time
import subprocess
import json
import logging
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.testing.LogCapture,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus Metrics
LATEX_BUILD_DURATION = Histogram(
    'latex_build_duration_seconds',
    'Time spent compiling LaTeX documents',
    labelnames=['document', 'stage', 'status']
)

LATEX_BUILD_SUCCESS = Counter(
    'latex_build_success_total',
    'Number of successful LaTeX builds',
    labelnames=['document', 'type']
)

LATEX_BUILD_FAILURE = Counter(
    'latex_build_failure_total', 
    'Number of failed LaTeX builds',
    labelnames=['document', 'type', 'error_type']
)

LATEX_WARNINGS = Counter(
    'latex_warnings_total',
    'Number of LaTeX warnings during compilation',
    labelnames=['document', 'warning_type']
)

LATEX_ERRORS = Counter(
    'latex_errors_total',
    'Number of LaTeX errors during compilation', 
    labelnames=['document', 'error_type']
)

LATEX_PAGES_GENERATED = Gauge(
    'latex_pages_generated_total',
    'Number of pages in generated PDF',
    labelnames=['document']
)

LATEX_BIBTEX_REFERENCES = Gauge(
    'latex_bibtex_references_total',
    'Number of bibliography references processed',
    labelnames=['document']
)

LATEX_FILE_SIZE_BYTES = Gauge(
    'latex_output_file_size_bytes',
    'Size of generated PDF in bytes',
    labelnames=['document']
)

class LaTeXMonitor:
    """Monitor LaTeX compilation with metrics and structured logging."""
    
    def __init__(self, base_path: Path, correlation_id: Optional[str] = None):
        self.base_path = Path(base_path)
        self.correlation_id = correlation_id or f"latex-{int(time.time())}"
        self.start_time = time.time()
        
        # Bind correlation ID to logger context
        structlog.contextvars.bind_contextvars(
            correlation_id=self.correlation_id,
            service="latex-compiler",
            component="documentation"
        )
        
    def monitor_compilation(self, document: str, makefile_target: str) -> bool:
        """Monitor a complete LaTeX compilation process."""
        logger.info(
            "Starting LaTeX compilation",
            document=document,
            target=makefile_target,
            base_path=str(self.base_path)
        )
        
        try:
            # Stage 1: Environment validation
            with LATEX_BUILD_DURATION.labels(
                document=document, stage="validation", status="running"
            ).time():
                self._validate_environment()
            
            # Stage 2: Template preparation
            with LATEX_BUILD_DURATION.labels(
                document=document, stage="template", status="running"
            ).time():
                self._prepare_templates(document)
            
            # Stage 3: Bibliography processing
            with LATEX_BUILD_DURATION.labels(
                document=document, stage="bibliography", status="running" 
            ).time():
                bibtex_count = self._process_bibliography(document)
                LATEX_BIBTEX_REFERENCES.labels(document=document).set(bibtex_count)
            
            # Stage 4: Main compilation
            with LATEX_BUILD_DURATION.labels(
                document=document, stage="compilation", status="running"
            ).time():
                success = self._run_compilation(document, makefile_target)
            
            if success:
                # Stage 5: Output validation and metrics
                with LATEX_BUILD_DURATION.labels(
                    document=document, stage="validation", status="success"
                ).time():
                    self._collect_output_metrics(document)
                
                LATEX_BUILD_SUCCESS.labels(document=document, type="full").inc()
                logger.info("LaTeX compilation completed successfully", document=document)
                return True
            else:
                LATEX_BUILD_FAILURE.labels(
                    document=document, type="full", error_type="compilation"
                ).inc()
                return False
                
        except Exception as e:
            LATEX_BUILD_FAILURE.labels(
                document=document, type="full", error_type=type(e).__name__
            ).inc()
            logger.error("LaTeX compilation failed", document=document, error=str(e))
            return False
    
    def _validate_environment(self):
        """Validate LaTeX environment and dependencies."""
        required_tools = ['pdflatex', 'bibtex']
        
        for tool in required_tools:
            if not self._check_command_exists(tool):
                raise RuntimeError(f"Required tool '{tool}' not found in PATH")
        
        logger.info("LaTeX environment validated", tools=required_tools)
    
    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH."""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _prepare_templates(self, document: str):
        """Prepare LaTeX templates and validate sources."""
        sources_dir = self.base_path / "sources"
        tex_file = sources_dir / f"{document}.tex"
        
        if not tex_file.exists():
            logger.warning("LaTeX source file not found", 
                         file=str(tex_file), document=document)
            raise FileNotFoundError(f"LaTeX source not found: {tex_file}")
        
        # Count template complexity metrics
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        sections = content.count('\\section{')
        figures = content.count('\\includegraphics')
        tables = content.count('\\begin{table}')
        
        logger.info("Template analysis completed",
                   document=document,
                   sections=sections,
                   figures=figures, 
                   tables=tables,
                   file_size_bytes=len(content.encode('utf-8')))
    
    def _process_bibliography(self, document: str) -> int:
        """Process bibliography and return reference count."""
        bib_dir = self.base_path / "bibliography"
        bib_file = bib_dir / "references.bib"
        
        if not bib_file.exists():
            logger.info("No bibliography file found", document=document)
            return 0
        
        # Count bibliography entries
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple count of @article, @book, @inproceedings, etc.
        import re
        entries = re.findall(r'@\w+\{', content)
        ref_count = len(entries)
        
        logger.info("Bibliography processed",
                   document=document,
                   references=ref_count,
                   bib_file=str(bib_file))
        
        return ref_count
    
    def _run_compilation(self, document: str, target: str) -> bool:
        """Run the actual LaTeX compilation via Makefile."""
        try:
            # Change to LaTeX directory
            original_cwd = os.getcwd()
            os.chdir(self.base_path)
            
            # Run make with the specified target
            result = subprocess.run(
                ['make', target],
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
            )
            
            # Parse compilation output for warnings and errors
            self._parse_latex_output(document, result.stdout, result.stderr)
            
            success = result.returncode == 0
            
            logger.info("Make command completed",
                       document=document,
                       target=target,
                       returncode=result.returncode,
                       stdout_lines=len(result.stdout.splitlines()),
                       stderr_lines=len(result.stderr.splitlines()))
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("LaTeX compilation timed out", document=document, target=target)
            LATEX_BUILD_FAILURE.labels(
                document=document, type="compilation", error_type="timeout"
            ).inc()
            return False
        except Exception as e:
            logger.error("Compilation command failed", 
                        document=document, target=target, error=str(e))
            return False
        finally:
            os.chdir(original_cwd)
    
    def _parse_latex_output(self, document: str, stdout: str, stderr: str):
        """Parse LaTeX output for warnings and errors."""
        
        # Common LaTeX warning patterns
        warning_patterns = {
            'overfull_hbox': r'Overfull \\hbox',
            'underfull_hbox': r'Underfull \\hbox', 
            'citation_undefined': r'Citation .* undefined',
            'reference_undefined': r'Reference .* undefined',
            'font_warning': r'Font shape .* undefined'
        }
        
        # Common LaTeX error patterns  
        error_patterns = {
            'missing_file': r'File .* not found',
            'undefined_control': r'Undefined control sequence',
            'missing_dollar': r'Missing \$ inserted',
            'runaway_argument': r'Runaway argument',
            'emergency_stop': r'Emergency stop'
        }
        
        output_text = stdout + stderr
        
        # Count warnings
        for warning_type, pattern in warning_patterns.items():
            import re
            matches = re.findall(pattern, output_text, re.IGNORECASE)
            if matches:
                LATEX_WARNINGS.labels(
                    document=document, warning_type=warning_type
                ).inc(len(matches))
                logger.warning("LaTeX warnings detected",
                             document=document,
                             warning_type=warning_type,
                             count=len(matches))
        
        # Count errors
        for error_type, pattern in error_patterns.items():
            matches = re.findall(pattern, output_text, re.IGNORECASE)
            if matches:
                LATEX_ERRORS.labels(
                    document=document, error_type=error_type
                ).inc(len(matches))
                logger.error("LaTeX errors detected",
                           document=document,
                           error_type=error_type,
                           count=len(matches))
    
    def _collect_output_metrics(self, document: str):
        """Collect metrics from the generated PDF output."""
        build_dir = self.base_path / "build"
        pdf_file = build_dir / f"{document}.pdf"
        
        if not pdf_file.exists():
            logger.warning("Generated PDF not found", 
                         document=document, 
                         expected_path=str(pdf_file))
            return
        
        # File size
        file_size = pdf_file.stat().st_size
        LATEX_FILE_SIZE_BYTES.labels(document=document).set(file_size)
        
        # Try to get page count using pdfinfo if available
        page_count = self._get_pdf_page_count(pdf_file)
        if page_count > 0:
            LATEX_PAGES_GENERATED.labels(document=document).set(page_count)
        
        logger.info("Output metrics collected",
                   document=document,
                   file_size_bytes=file_size,
                   pages=page_count,
                   output_file=str(pdf_file))
    
    def _get_pdf_page_count(self, pdf_file: Path) -> int:
        """Get page count from PDF file."""
        try:
            # Try using pdfinfo command
            result = subprocess.run(
                ['pdfinfo', str(pdf_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import re
                match = re.search(r'Pages:\s+(\d+)', result.stdout)
                if match:
                    return int(match.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback: try using PyPDF2 if available
        try:
            import PyPDF2
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return len(reader.pages)
        except ImportError:
            logger.warning("pdfinfo and PyPDF2 not available for page counting")
        except Exception as e:
            logger.warning("Could not count PDF pages", error=str(e))
        
        return 0

def main():
    """Main entry point for LaTeX monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor LaTeX compilation")
    parser.add_argument("--document", required=True, help="Document name to compile")
    parser.add_argument("--target", default="all", help="Makefile target")
    parser.add_argument("--latex-dir", default=".", help="LaTeX directory path")
    parser.add_argument("--metrics-port", type=int, default=9091, 
                       help="Prometheus metrics port")
    parser.add_argument("--correlation-id", help="Correlation ID for tracing")
    
    args = parser.parse_args()
    
    # Start Prometheus metrics server
    start_http_server(args.metrics_port)
    logger.info("Prometheus metrics server started", port=args.metrics_port)
    
    # Initialize monitor
    monitor = LaTeXMonitor(
        base_path=Path(args.latex_dir),
        correlation_id=args.correlation_id
    )
    
    # Run compilation monitoring
    success = monitor.monitor_compilation(args.document, args.target)
    
    # Exit with appropriate code
    exit(0 if success else 1)

if __name__ == "__main__":
    main()