#!/usr/bin/env python3
"""
Secure Data Loading System with Comprehensive Type Safety

This module provides secure, type-safe data loading mechanisms for multi-modal
agent training datasets with comprehensive security validation, sanitization,
and monitoring integration.
"""

import json
import hashlib
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Optional, Union, Any, Set, Protocol, 
    Literal, TypeVar, Generic, Callable, Iterator, AsyncIterator
)
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import StrictStr, StrictInt, StrictBool, constr, conint
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading

from .prompt_injection_analysis import (
    SecurityAnalyzer, SecurityAnalyzerFactory, SecurityAnalysisResult,
    SecureDatasetConfig, SecureDatasetProcessor
)
from .input_validation import (
    SecureInputValidator, ValidationReport, BatchValidator,
    ToolBenchDatasetEntry, WebArenaContentValidator, AgentInstructEntry
)
from .advanced_detection import PromptInjectionDetectionSystem, DetectionResult
from .guardrails import SecurityGuardrailManager, SecurityPolicies, SecurityPolicy
from .monitoring import SecurityMonitor, AlertType

logger = structlog.get_logger(__name__)

# Type definitions
DatasetType = Literal["toolbench", "webarena", "agentinstruct", "react", "miniowb", "browsergym"]
LoadingMode = Literal["strict", "permissive", "sanitize"]
ProcessingStage = Literal["raw", "validated", "sanitized", "filtered", "processed"]
T = TypeVar('T')


class DataLoadingError(Exception):
    """Raised when secure data loading fails."""
    
    def __init__(self, message: str, file_path: Optional[Path] = None, error_code: Optional[str] = None):
        self.message = message
        self.file_path = file_path
        self.error_code = error_code
        super().__init__(message)


@dataclass(frozen=True)
class DatasetEntry:
    """Secure dataset entry with comprehensive metadata."""
    
    entry_id: str
    dataset_type: DatasetType
    content: Union[Dict[str, Any], str]
    security_analysis: SecurityAnalysisResult
    validation_report: ValidationReport
    processing_stage: ProcessingStage
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    content_hash: Optional[str] = None
    created_timestamp: Optional[float] = None


@dataclass(frozen=True)
class LoadingStatistics:
    """Statistics for data loading operations."""
    
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_entries: int = 0
    valid_entries: int = 0
    sanitized_entries: int = 0
    filtered_entries: int = 0
    blocked_entries: int = 0
    processing_time: float = 0.0
    security_violations: int = 0
    errors: List[str] = field(default_factory=list)


class SecureDatasetSchema(BaseModel):
    """Base schema for secure dataset validation."""
    
    class Config:
        extra = "forbid"  # Reject extra fields
        validate_assignment = True
        use_enum_values = True
    
    entry_id: constr(min_length=1, max_length=100) = Field(..., description="Unique entry identifier")
    dataset_type: DatasetType = Field(..., description="Type of dataset")
    content_hash: constr(min_length=32, max_length=64) = Field(..., description="Content hash for integrity")
    source_info: Dict[str, str] = Field(default_factory=dict, description="Source file information")
    
    @validator('entry_id')
    def validate_entry_id(cls, v):
        """Validate entry ID format."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Entry ID must be alphanumeric with hyphens/underscores only")
        return v


class ToolBenchSecureEntry(SecureDatasetSchema):
    """Secure schema for ToolBench entries."""
    
    dataset_type: Literal["toolbench"] = "toolbench"
    content: ToolBenchDatasetEntry = Field(..., description="Validated ToolBench content")
    
    @validator('content')
    def validate_toolbench_content(cls, v):
        """Additional validation for ToolBench content."""
        # Ensure no dangerous function calls
        if hasattr(v, 'function') and v.function:
            dangerous_patterns = ['exec', 'eval', 'system', 'subprocess', 'shell']
            func_name = v.function.name.lower()
            if any(pattern in func_name for pattern in dangerous_patterns):
                raise ValueError(f"Dangerous function call detected: {v.function.name}")
        return v


class WebArenaSecureEntry(SecureDatasetSchema):
    """Secure schema for WebArena entries."""
    
    dataset_type: Literal["webarena"] = "webarena"
    content: WebArenaContentValidator = Field(..., description="Validated WebArena content")
    
    @validator('content')
    def validate_webarena_content(cls, v):
        """Additional validation for WebArena content."""
        # Check DOM content size
        if hasattr(v, 'dom') and v.dom and len(v.dom) > 500000:  # 500KB limit
            raise ValueError("DOM content too large")
        return v


class AgentInstructSecureEntry(SecureDatasetSchema):
    """Secure schema for AgentInstruct entries."""
    
    dataset_type: Literal["agentinstruct"] = "agentinstruct"
    content: AgentInstructEntry = Field(..., description="Validated AgentInstruct content")


class SecureDataLoader(ABC):
    """Abstract base class for secure data loaders."""
    
    def __init__(
        self,
        dataset_type: DatasetType,
        security_policy: SecurityPolicy,
        loading_mode: LoadingMode = "strict"
    ):
        self.dataset_type = dataset_type
        self.security_policy = security_policy
        self.loading_mode = loading_mode
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize security components
        self.input_validator = SecureInputValidator(strict_mode=(loading_mode == "strict"))
        self.detection_system = PromptInjectionDetectionSystem(strict_mode=(loading_mode == "strict"))
        self.guardrail_manager = SecurityGuardrailManager(security_policy)
        
        # Statistics tracking
        self.stats = LoadingStatistics()
        self._stats_lock = threading.Lock()
    
    @abstractmethod
    def load_file(self, file_path: Path) -> Iterator[DatasetEntry]:
        """Load and validate entries from a single file."""
        pass
    
    @abstractmethod
    def validate_entry(self, raw_entry: Dict[str, Any]) -> ValidationReport:
        """Validate a single entry."""
        pass
    
    def process_entry_securely(
        self, 
        raw_entry: Dict[str, Any], 
        source_file: Optional[str] = None
    ) -> Optional[DatasetEntry]:
        """Process a single entry with comprehensive security checks."""
        
        try:
            # Generate entry ID and content hash
            entry_id = self._generate_entry_id(raw_entry)
            content_hash = self._calculate_content_hash(raw_entry)
            
            # Validate entry structure
            validation_report = self.validate_entry(raw_entry)
            
            if not validation_report.is_valid and self.loading_mode == "strict":
                self._update_stats(blocked_entries=1)
                self.logger.warning(
                    "Entry blocked in strict mode",
                    entry_id=entry_id,
                    errors=validation_report.errors
                )
                return None
            
            # Use sanitized content if available
            processed_content = validation_report.sanitized_content or raw_entry
            
            # Security analysis
            security_analysis = self._analyze_security(processed_content)
            
            # Check security guardrails
            guardrail_context = {
                "content": processed_content,
                "file_path": source_file,
                "entry_id": entry_id
            }
            
            violations = self.guardrail_manager.check_all(guardrail_context)
            if violations:
                critical_violations = [v for v in violations if v.severity == "critical"]
                if critical_violations and self.loading_mode != "permissive":
                    self._update_stats(blocked_entries=1, security_violations=len(violations))
                    self.logger.warning(
                        "Entry blocked by security guardrails",
                        entry_id=entry_id,
                        violations=[v.message for v in critical_violations]
                    )
                    return None
            
            # Determine processing stage
            processing_stage = self._determine_processing_stage(
                validation_report, security_analysis, violations
            )
            
            # Create secure dataset entry
            dataset_entry = DatasetEntry(
                entry_id=entry_id,
                dataset_type=self.dataset_type,
                content=processed_content,
                security_analysis=security_analysis,
                validation_report=validation_report,
                processing_stage=processing_stage,
                metadata={
                    "loading_mode": self.loading_mode,
                    "guardrail_violations": len(violations),
                    "has_security_warnings": security_analysis.overall_risk != "safe"
                },
                source_file=source_file,
                content_hash=content_hash,
                created_timestamp=time.time()
            )
            
            # Update statistics
            self._update_stats_for_entry(dataset_entry)
            
            return dataset_entry
            
        except Exception as e:
            self._update_stats(failed_entries=1)
            self.logger.error(
                "Entry processing failed",
                entry_id=entry_id if 'entry_id' in locals() else "unknown",
                error=str(e),
                source_file=source_file
            )
            
            if self.loading_mode == "strict":
                raise DataLoadingError(f"Entry processing failed: {e}", error_code="PROCESSING_ERROR")
            
            return None
    
    def _generate_entry_id(self, entry: Dict[str, Any]) -> str:
        """Generate unique entry ID."""
        # Use existing ID if available
        if "id" in entry:
            return str(entry["id"])
        
        # Generate from content hash
        content_str = json.dumps(entry, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        return f"{self.dataset_type}_{content_hash[:16]}"
    
    def _calculate_content_hash(self, content: Union[str, Dict[str, Any]]) -> str:
        """Calculate SHA-256 hash of content."""
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _analyze_security(self, content: Union[str, Dict[str, Any]]) -> SecurityAnalysisResult:
        """Perform comprehensive security analysis."""
        analyzer = SecurityAnalyzerFactory.create_analyzer(
            self.dataset_type, 
            strict_mode=(self.loading_mode == "strict")
        )
        return analyzer.analyze(content)
    
    def _determine_processing_stage(
        self,
        validation_report: ValidationReport,
        security_analysis: SecurityAnalysisResult,
        violations: List
    ) -> ProcessingStage:
        """Determine the processing stage based on analysis results."""
        
        if violations:
            return "filtered"
        elif security_analysis.overall_risk in ["dangerous", "blocked"]:
            return "filtered"
        elif validation_report.validation_result == "sanitized":
            return "sanitized"
        elif validation_report.validation_result == "valid":
            return "validated"
        else:
            return "raw"
    
    def _update_stats(self, **kwargs):
        """Thread-safe statistics update."""
        with self._stats_lock:
            for key, value in kwargs.items():
                if hasattr(self.stats, key):
                    current_value = getattr(self.stats, key)
                    if isinstance(current_value, int):
                        setattr(self.stats, key, current_value + value)
                    elif isinstance(current_value, list):
                        current_value.extend(value if isinstance(value, list) else [value])
    
    def _update_stats_for_entry(self, entry: DatasetEntry):
        """Update statistics based on processed entry."""
        stats_update = {"total_entries": 1}
        
        if entry.processing_stage == "validated":
            stats_update["valid_entries"] = 1
        elif entry.processing_stage == "sanitized":
            stats_update["sanitized_entries"] = 1
        elif entry.processing_stage == "filtered":
            stats_update["filtered_entries"] = 1
        
        if entry.security_analysis.overall_risk != "safe":
            stats_update["security_violations"] = 1
        
        self._update_stats(**stats_update)


class ToolBenchSecureLoader(SecureDataLoader):
    """Secure loader for ToolBench datasets."""
    
    def __init__(self, security_policy: SecurityPolicy, loading_mode: LoadingMode = "strict"):
        super().__init__("toolbench", security_policy, loading_mode)
    
    def load_file(self, file_path: Path) -> Iterator[DatasetEntry]:
        """Load ToolBench entries from JSONL file."""
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}", file_path=file_path)
        
        self.logger.info("Loading ToolBench file", file_path=str(file_path))
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        raw_entry = json.loads(line)
                        entry = self.process_entry_securely(raw_entry, str(file_path))
                        
                        if entry:
                            yield entry
                        
                    except json.JSONDecodeError as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"JSON decode error at line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
                    
                    except Exception as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"Error processing line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
        
        except Exception as e:
            self._update_stats(failed_files=1)
            raise DataLoadingError(f"Failed to load file: {e}", file_path=file_path)
    
    def validate_entry(self, raw_entry: Dict[str, Any]) -> ValidationReport:
        """Validate ToolBench entry."""
        return self.input_validator.validate_toolbench_entry(raw_entry)


class WebArenaSecureLoader(SecureDataLoader):
    """Secure loader for WebArena datasets."""
    
    def __init__(self, security_policy: SecurityPolicy, loading_mode: LoadingMode = "strict"):
        super().__init__("webarena", security_policy, loading_mode)
    
    def load_file(self, file_path: Path) -> Iterator[DatasetEntry]:
        """Load WebArena entries from JSONL file."""
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}", file_path=file_path)
        
        self.logger.info("Loading WebArena file", file_path=str(file_path))
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        raw_entry = json.loads(line)
                        entry = self.process_entry_securely(raw_entry, str(file_path))
                        
                        if entry:
                            yield entry
                        
                    except json.JSONDecodeError as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"JSON decode error at line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
                    
                    except Exception as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"Error processing line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
        
        except Exception as e:
            self._update_stats(failed_files=1)
            raise DataLoadingError(f"Failed to load file: {e}", file_path=file_path)
    
    def validate_entry(self, raw_entry: Dict[str, Any]) -> ValidationReport:
        """Validate WebArena entry."""
        return self.input_validator.validate_webarena_entry(raw_entry)


class AgentInstructSecureLoader(SecureDataLoader):
    """Secure loader for AgentInstruct datasets."""
    
    def __init__(self, security_policy: SecurityPolicy, loading_mode: LoadingMode = "strict"):
        super().__init__("agentinstruct", security_policy, loading_mode)
    
    def load_file(self, file_path: Path) -> Iterator[DatasetEntry]:
        """Load AgentInstruct entries from JSONL file."""
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}", file_path=file_path)
        
        self.logger.info("Loading AgentInstruct file", file_path=str(file_path))
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        raw_entry = json.loads(line)
                        entry = self.process_entry_securely(raw_entry, str(file_path))
                        
                        if entry:
                            yield entry
                        
                    except json.JSONDecodeError as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"JSON decode error at line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
                    
                    except Exception as e:
                        self._update_stats(failed_entries=1)
                        error_msg = f"Error processing line {line_num}: {e}"
                        self.logger.error(error_msg, file_path=str(file_path))
                        
                        if self.loading_mode == "strict":
                            raise DataLoadingError(error_msg, file_path=file_path)
        
        except Exception as e:
            self._update_stats(failed_files=1)
            raise DataLoadingError(f"Failed to load file: {e}", file_path=file_path)
    
    def validate_entry(self, raw_entry: Dict[str, Any]) -> ValidationReport:
        """Validate AgentInstruct entry."""
        return self.input_validator.validate_agentinstruct_entry(raw_entry)


class SecureDataLoaderFactory:
    """Factory for creating secure data loaders."""
    
    _loaders = {
        "toolbench": ToolBenchSecureLoader,
        "webarena": WebArenaSecureLoader,
        "agentinstruct": AgentInstructSecureLoader,
        "react": AgentInstructSecureLoader,  # Uses same validation as AgentInstruct
        "miniowb": WebArenaSecureLoader,  # Uses same validation as WebArena
        "browsergym": WebArenaSecureLoader,  # Uses same validation as WebArena
    }
    
    @classmethod
    def create_loader(
        cls, 
        dataset_type: DatasetType, 
        security_policy: SecurityPolicy,
        loading_mode: LoadingMode = "strict"
    ) -> SecureDataLoader:
        """Create appropriate secure loader for dataset type."""
        loader_class = cls._loaders.get(dataset_type)
        if not loader_class:
            raise ValueError(f"No secure loader available for dataset type: {dataset_type}")
        return loader_class(security_policy, loading_mode)


class SecureDatasetManager:
    """High-level manager for secure multi-dataset loading and processing."""
    
    def __init__(
        self,
        security_policy: SecurityPolicy,
        loading_mode: LoadingMode = "strict",
        enable_monitoring: bool = True
    ):
        self.security_policy = security_policy
        self.loading_mode = loading_mode
        self.enable_monitoring = enable_monitoring
        self.logger = structlog.get_logger(__name__)
        
        # Initialize monitoring if enabled
        if enable_monitoring:
            monitor_config = {
                "monitoring_interval": 300,
                "event_rate_threshold": 50,
                "alerting": {"email": {"enabled": False}, "webhook": {"enabled": False}}
            }
            self.security_monitor = SecurityMonitor(monitor_config)
        else:
            self.security_monitor = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def load_dataset_files(
        self,
        dataset_configs: List[Dict[str, Any]],
        max_entries_per_file: Optional[int] = None
    ) -> Iterator[DatasetEntry]:
        """Load multiple dataset files with comprehensive security."""
        
        total_files = len(dataset_configs)
        processed_files = 0
        
        self.logger.info(
            "Starting secure dataset loading",
            total_files=total_files,
            loading_mode=self.loading_mode,
            security_policy=type(self.security_policy).__name__
        )
        
        for config in dataset_configs:
            try:
                dataset_type = config["dataset_type"]
                file_path = Path(config["file_path"])
                
                # Create appropriate loader
                loader = SecureDataLoaderFactory.create_loader(
                    dataset_type, 
                    self.security_policy, 
                    self.loading_mode
                )
                
                # Load entries from file
                entry_count = 0
                for entry in loader.load_file(file_path):
                    yield entry
                    entry_count += 1
                    
                    # Check entry limit
                    if max_entries_per_file and entry_count >= max_entries_per_file:
                        break
                    
                    # Record security events if monitoring enabled
                    if self.security_monitor:
                        self._record_security_events(entry)
                
                processed_files += 1
                
                self.logger.info(
                    "File loading completed",
                    file_path=str(file_path),
                    dataset_type=dataset_type,
                    entries_loaded=entry_count,
                    progress=f"{processed_files}/{total_files}"
                )
                
            except Exception as e:
                self.logger.error(
                    "File loading failed",
                    file_path=config.get("file_path", "unknown"),
                    dataset_type=config.get("dataset_type", "unknown"),
                    error=str(e)
                )
                
                if self.loading_mode == "strict":
                    raise DataLoadingError(f"Dataset loading failed: {e}")
    
    async def load_dataset_files_async(
        self,
        dataset_configs: List[Dict[str, Any]],
        max_entries_per_file: Optional[int] = None
    ) -> AsyncIterator[DatasetEntry]:
        """Asynchronously load multiple dataset files."""
        
        async def load_file_async(config):
            """Load single file asynchronously."""
            dataset_type = config["dataset_type"]
            file_path = Path(config["file_path"])
            
            # Create loader in thread pool
            loop = asyncio.get_event_loop()
            loader = await loop.run_in_executor(
                self.executor,
                SecureDataLoaderFactory.create_loader,
                dataset_type,
                self.security_policy,
                self.loading_mode
            )
            
            # Load entries
            entry_count = 0
            async for entry in self._load_file_entries_async(loader, file_path):
                yield entry
                entry_count += 1
                
                if max_entries_per_file and entry_count >= max_entries_per_file:
                    break
        
        # Process files concurrently
        tasks = [load_file_async(config) for config in dataset_configs]
        
        for task in asyncio.as_completed(tasks):
            async for entry in await task:
                yield entry
    
    async def _load_file_entries_async(
        self, 
        loader: SecureDataLoader, 
        file_path: Path
    ) -> AsyncIterator[DatasetEntry]:
        """Load entries from file asynchronously."""
        
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}", file_path=file_path)
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            async for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    raw_entry = json.loads(line)
                    
                    # Process in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    entry = await loop.run_in_executor(
                        self.executor,
                        loader.process_entry_securely,
                        raw_entry,
                        str(file_path)
                    )
                    
                    if entry:
                        yield entry
                
                except json.JSONDecodeError as e:
                    self.logger.error(
                        "JSON decode error",
                        file_path=str(file_path),
                        line_num=line_num,
                        error=str(e)
                    )
                    
                    if self.loading_mode == "strict":
                        raise DataLoadingError(f"JSON decode error: {e}", file_path=file_path)
                
                except Exception as e:
                    self.logger.error(
                        "Entry processing error",
                        file_path=str(file_path),
                        line_num=line_num,
                        error=str(e)
                    )
                    
                    if self.loading_mode == "strict":
                        raise DataLoadingError(f"Entry processing error: {e}", file_path=file_path)
    
    def _record_security_events(self, entry: DatasetEntry):
        """Record security events for monitoring."""
        if not self.security_monitor:
            return
        
        # Record security analysis results
        if entry.security_analysis.overall_risk in ["dangerous", "blocked"]:
            self.security_monitor.record_security_event(
                event_type=AlertType.PROMPT_INJECTION,
                severity="critical" if entry.security_analysis.overall_risk == "blocked" else "warning",
                source=entry.source_file or "unknown",
                message=f"Security threat detected in {entry.dataset_type} entry",
                details={
                    "entry_id": entry.entry_id,
                    "threat_count": len(entry.security_analysis.threats),
                    "overall_risk": entry.security_analysis.overall_risk
                },
                content_hash=entry.content_hash
            )
        
        # Record validation issues
        if not entry.validation_report.is_valid:
            self.security_monitor.record_security_event(
                event_type=AlertType.SECURITY_VIOLATION,
                severity="warning",
                source=entry.source_file or "unknown",
                message=f"Validation failed for {entry.dataset_type} entry",
                details={
                    "entry_id": entry.entry_id,
                    "validation_errors": entry.validation_report.errors,
                    "validation_result": entry.validation_report.validation_result
                },
                content_hash=entry.content_hash
            )
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        # This would aggregate statistics from all loaders
        # Implementation depends on how you track loader instances
        return {
            "loading_mode": self.loading_mode,
            "security_policy": type(self.security_policy).__name__,
            "monitoring_enabled": self.enable_monitoring,
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Shutdown the dataset manager and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.security_monitor:
            self.security_monitor.shutdown()


# Example usage and configuration
def create_example_dataset_configs() -> List[Dict[str, Any]]:
    """Create example dataset configurations."""
    return [
        {
            "dataset_type": "toolbench",
            "file_path": "/path/to/toolbench.jsonl",
            "description": "ToolBench function calling dataset"
        },
        {
            "dataset_type": "webarena",
            "file_path": "/path/to/webarena.jsonl",
            "description": "WebArena web interaction dataset"
        },
        {
            "dataset_type": "agentinstruct",
            "file_path": "/path/to/agentinstruct.jsonl",
            "description": "AgentInstruct instruction dataset"
        }
    ]


def main():
    """Example usage of the secure data loading system."""
    
    # Create security policy
    security_policy = SecurityPolicies.strict()
    
    # Initialize secure dataset manager
    manager = SecureDatasetManager(
        security_policy=security_policy,
        loading_mode="sanitize",  # Allow sanitization
        enable_monitoring=True
    )
    
    # Example dataset configurations
    dataset_configs = create_example_dataset_configs()
    
    try:
        # Load datasets securely
        entries_loaded = 0
        for entry in manager.load_dataset_files(dataset_configs, max_entries_per_file=100):
            entries_loaded += 1
            print(f"Loaded entry {entry.entry_id} from {entry.dataset_type} dataset")
            
            if entries_loaded >= 10:  # Limit for example
                break
        
        print(f"Successfully loaded {entries_loaded} entries")
        
        # Get statistics
        stats = manager.get_loading_statistics()
        print(f"Loading statistics: {stats}")
    
    except DataLoadingError as e:
        print(f"Data loading failed: {e.message}")
        if e.file_path:
            print(f"File: {e.file_path}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        manager.shutdown()


if __name__ == "__main__":
    import time
    main()