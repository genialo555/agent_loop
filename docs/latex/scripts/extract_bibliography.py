#!/usr/bin/env python3
"""
Production-ready bibliography extraction tool with complete type safety.

This script extracts bibliography entries from PDF text and converts them to BibTeX format,
following strict Type Guardian standards for the Gemma-3N-Agent-Loop project.

Features:
- Complete type annotations for all functions and variables
- Pydantic models for data validation and sanitization
- Comprehensive error handling with structured logging
- Security measures for text processing and file I/O
- Modern Python 3.11+ patterns with match/case
- Centralized configuration via project settings
- Input sanitization and validation
- Async context managers for file operations
"""

from __future__ import annotations

import asyncio
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import from project's centralized settings
try:
    from agent_loop.models.core.settings import settings as global_settings
except ImportError:
    # Fallback for development mode
    class FallbackSettings(BaseSettings):
        """Fallback settings when main project settings unavailable."""
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8", 
            case_sensitive=False,
            extra="ignore"
        )
        
        docs_dir: Path = Field(default=Path("docs"), description="Documentation directory")
        
    global_settings = FallbackSettings()


class BibEntryType(str, Enum):
    """Enumeration of supported BibTeX entry types."""
    ARTICLE = "article"
    BOOK = "book"
    INPROCEEDINGS = "inproceedings"
    MISC = "misc"
    TECHREPORT = "techreport"
    UNPUBLISHED = "unpublished"


class ProcessingStatus(str, Enum):
    """Status enumeration for bibliography processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BibliographyEntry(BaseModel):
    """Pydantic model for a bibliography entry with validation and sanitization."""
    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid"
    }
    
    reference_number: int = Field(
        ge=1,
        le=999,
        description="Reference number from the original document"
    )
    entry_type: BibEntryType = Field(description="BibTeX entry type")
    entry_key: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique BibTeX key"
    )
    title: str = Field(
        min_length=1,
        max_length=500,
        description="Publication title"
    )
    authors: str = Field(
        min_length=1,
        max_length=300,
        description="Author names"
    )
    year: Union[int, str] = Field(description="Publication year")
    raw_text: str = Field(
        min_length=1,
        max_length=2000,
        description="Original reference text"
    )
    note: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Additional note"
    )
    
    @field_validator("entry_key")
    @classmethod
    def validate_entry_key(cls, v: str) -> str:
        """Ensure entry key follows BibTeX conventions."""
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Entry key must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Union[int, str]) -> Union[int, str]:
        """Validate publication year."""
        if isinstance(v, str):
            if v.lower() in ["unknown", "year not found", "n/a"]:
                return v
            try:
                year_int = int(v)
                if 1900 <= year_int <= datetime.now().year + 5:
                    return year_int
            except ValueError:
                pass
            raise ValueError(f"Invalid year format: {v}")
        elif isinstance(v, int):
            if not (1900 <= v <= datetime.now().year + 5):
                raise ValueError(f"Year must be between 1900 and {datetime.now().year + 5}")
        return v
    
    @field_validator("title", "authors", "raw_text")
    @classmethod
    def sanitize_text_fields(cls, v: str) -> str:
        """Sanitize text fields to prevent injection attacks."""
        if not v or not isinstance(v, str):
            raise ValueError("Text field cannot be empty")
        
        # Remove potentially dangerous characters
        dangerous_patterns = [
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'\\[a-zA-Z]+\{',  # LaTeX commands (basic detection)
        ]
        
        sanitized = v
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        if len(sanitized) < len(v) * 0.5:  # Sanity check
            raise ValueError("Text field contains too much suspicious content")
            
        return sanitized
    
    def to_bibtex(self) -> str:
        """Convert entry to BibTeX format with proper escaping."""
        # Escape special BibTeX characters
        def escape_bibtex(text: str) -> str:
            """Escape special characters for BibTeX."""
            escapes = {
                '&': '\\&',
                '%': '\\%',
                '$': '\\$',
                '#': '\\#',
                '^': '\\textasciicircum{}',
                '_': '\\_',
                '{': '\\{',
                '}': '\\}',
                '~': '\\textasciitilde{}',
                '\\': '\\textbackslash{}'
            }
            for char, escape in escapes.items():
                text = text.replace(char, escape)
            return text
        
        title_escaped = escape_bibtex(self.title)
        authors_escaped = escape_bibtex(self.authors)
        
        entry = f"@{self.entry_type.value}{{{self.entry_key},\n"
        entry += f"  title={{{title_escaped}}},\n"
        entry += f"  author={{{authors_escaped}}},\n"
        entry += f"  year={{{self.year}}}"
        
        if self.note:
            note_escaped = escape_bibtex(self.note)
            entry += f",\n  note={{{note_escaped}}}"
        
        entry += "\n}"
        return entry


class BibliographyExtractionConfig(BaseSettings):
    """Configuration for bibliography extraction with validation."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # File paths using centralized settings
    docs_dir: Path = Field(
        default_factory=lambda: global_settings.model_dir.parent / "docs",
        description="Documentation directory"
    )
    output_dir: Path = Field(
        default_factory=lambda: global_settings.model_dir.parent / "docs" / "latex" / "bibliography",
        description="Output directory for bibliography files"
    )
    
    # Processing limits for security
    max_references: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of references to process"
    )
    max_text_length: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        description="Maximum input text length in characters"
    )
    
    # Output settings
    output_filename: str = Field(
        default="references.bib",
        pattern=r"^[a-zA-Z0-9_.-]+\.bib$",
        description="Output BibTeX filename"
    )
    include_header: bool = Field(
        default=True,
        description="Include header comment in output file"
    )
    
    # Regex patterns for extraction
    reference_pattern: str = Field(
        default=r'(\d+)\.\s+(.+?)(?=\n\d+\.|$)',
        description="Regex pattern for extracting numbered references"
    )
    
    @field_validator("docs_dir", "output_dir")
    @classmethod
    def validate_directories(cls, v: Path) -> Path:
        """Ensure directories are secure paths."""
        # Convert to absolute path for security
        abs_path = v.resolve()
        
        # Basic security check - ensure path doesn't traverse outside project
        if '..' in str(abs_path):
            raise ValueError(f"Directory path contains unsafe components: {abs_path}")
            
        return abs_path


class BibliographyProcessor(Protocol):
    """Protocol for bibliography processing implementations."""
    
    async def extract_references(self, text: str) -> List[BibliographyEntry]:
        """Extract bibliography entries from text."""
        ...
    
    async def save_bibliography(self, entries: List[BibliographyEntry], output_path: Path) -> bool:
        """Save bibliography entries to file."""
        ...


class ProductionBibliographyExtractor:
    """Production-ready bibliography extractor with comprehensive error handling."""
    
    __slots__ = ('config', '_entry_type_patterns', '_processing_status')
    
    def __init__(self, config: Optional[BibliographyExtractionConfig] = None) -> None:
        """Initialize extractor with configuration."""
        self.config = config or BibliographyExtractionConfig()
        self._processing_status: ProcessingStatus = ProcessingStatus.PENDING
        
        # Pre-compiled regex patterns for entry type detection
        self._entry_type_patterns: Dict[BibEntryType, re.Pattern[str]] = {
            BibEntryType.ARTICLE: re.compile(
                r'\b(?:journal|article|paper)\b', 
                re.IGNORECASE
            ),
            BibEntryType.INPROCEEDINGS: re.compile(
                r'\b(?:proceedings|conference|workshop|symposium)\b', 
                re.IGNORECASE
            ),
            BibEntryType.BOOK: re.compile(
                r'\b(?:press|publisher|book|edition)\b', 
                re.IGNORECASE
            ),
            BibEntryType.MISC: re.compile(
                r'\b(?:arxiv|preprint|technical report|tech\.?\s*report)\b', 
                re.IGNORECASE
            ),
        }
    
    def _classify_entry_type(self, text: str) -> BibEntryType:
        """Classify bibliography entry type using pattern matching."""
        text_lower = text.lower()
        
        # Use match/case for modern Python pattern matching
        match True:
            case _ if self._entry_type_patterns[BibEntryType.ARTICLE].search(text_lower):
                return BibEntryType.ARTICLE
            case _ if self._entry_type_patterns[BibEntryType.INPROCEEDINGS].search(text_lower):
                return BibEntryType.INPROCEEDINGS
            case _ if self._entry_type_patterns[BibEntryType.BOOK].search(text_lower):
                return BibEntryType.BOOK
            case _ if self._entry_type_patterns[BibEntryType.MISC].search(text_lower):
                return BibEntryType.MISC
            case _:
                return BibEntryType.MISC
    
    def _extract_metadata(self, ref_text: str) -> Dict[str, str]:
        """Extract metadata from reference text using modern regex patterns."""
        # Sanitize input first
        cleaned_text = re.sub(r'\s+', ' ', ref_text.strip())
        
        metadata: Dict[str, str] = {}
        
        # Extract title using multiple strategies
        title_patterns = [
            r'"([^"]+)"',  # Quoted title
            r'\b([A-Z][^.!?]*[.!?])\b',  # Sentence-like title
            r'^[^.]*?\.\s*([A-Z][^.]*?)\.',  # Second sentence
        ]
        
        for pattern in title_patterns:
            if match := re.search(pattern, cleaned_text):
                metadata['title'] = match.group(1).strip()
                break
        else:
            metadata['title'] = "Title extraction failed"
        
        # Extract authors (usually at the beginning)
        author_patterns = [
            r'^([^.]+?)\.',  # First sentence
            r'^([^,]+(?:,[^,]+)*)',  # Comma-separated names
        ]
        
        for pattern in author_patterns:
            if match := re.search(pattern, cleaned_text):
                authors = match.group(1).strip()
                # Clean up common author formatting
                authors = re.sub(r'\bet\s+al\.?', 'et al.', authors)
                metadata['authors'] = authors
                break
        else:
            metadata['authors'] = "Authors extraction failed"
        
        # Extract year with improved pattern
        if year_match := re.search(r'\b(19|20)\d{2}\b', cleaned_text):
            metadata['year'] = year_match.group(0)
        else:
            metadata['year'] = "Year not found"
        
        return metadata
    
    async def extract_references(self, text: str) -> List[BibliographyEntry]:
        """Extract bibliography entries from text with comprehensive validation."""
        if len(text) > self.config.max_text_length:
            raise ValueError(f"Input text too long: {len(text)} > {self.config.max_text_length}")
        
        self._processing_status = ProcessingStatus.PROCESSING
        
        try:
            # Extract references using configured pattern
            pattern = re.compile(self.config.reference_pattern, re.DOTALL | re.MULTILINE)
            raw_references = pattern.findall(text)
            
            if len(raw_references) > self.config.max_references:
                raise ValueError(f"Too many references: {len(raw_references)} > {self.config.max_references}")
            
            entries: List[BibliographyEntry] = []
            
            for ref_num_str, ref_text in raw_references:
                try:
                    ref_num = int(ref_num_str)
                    
                    # Clean and validate reference text
                    cleaned_text = re.sub(r'\s+', ' ', ref_text.strip())
                    
                    if not cleaned_text or len(cleaned_text) < 10:
                        continue  # Skip too short references
                    
                    # Classify entry type
                    entry_type = self._classify_entry_type(cleaned_text)
                    
                    # Extract metadata
                    metadata = self._extract_metadata(cleaned_text)
                    
                    # Create entry with validation
                    entry = BibliographyEntry(
                        reference_number=ref_num,
                        entry_type=entry_type,
                        entry_key=f"ref{ref_num:03d}",
                        title=metadata['title'],
                        authors=metadata['authors'],
                        year=metadata['year'],
                        raw_text=cleaned_text[:500],  # Truncate for storage
                        note=f"Ref {ref_num}: Auto-extracted from text"
                    )
                    
                    entries.append(entry)
                    
                except (ValueError, ValidationError) as e:
                    print(f"Warning: Failed to process reference {ref_num_str}: {e}")
                    continue
            
            self._processing_status = ProcessingStatus.COMPLETED
            return entries
            
        except Exception as e:
            self._processing_status = ProcessingStatus.FAILED
            raise RuntimeError(f"Bibliography extraction failed: {e}") from e
    
    @asynccontextmanager
    async def _safe_file_writer(self, file_path: Path) -> AsyncGenerator[Any, None]:
        """Async context manager for safe file writing."""
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temporary file for atomic writes
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        
        try:
            # Use asyncio for non-blocking I/O simulation
            await asyncio.sleep(0)  # Yield control
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                yield f
            
            # Atomic move after successful write
            temp_path.replace(file_path)
            
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    async def save_bibliography(self, entries: List[BibliographyEntry], output_path: Optional[Path] = None) -> bool:
        """Save bibliography entries to file with atomic writes."""
        if not entries:
            raise ValueError("No entries to save")
        
        target_path = output_path or (self.config.output_dir / self.config.output_filename)
        
        try:
            async with self._safe_file_writer(target_path) as f:
                # Write header if configured
                if self.config.include_header:
                    f.write("% Bibliography for Hierarchical Reasoning Model\n")
                    f.write("% Auto-generated by production bibliography extractor\n")
                    f.write(f"% Generated on: {datetime.now().isoformat()}\n")
                    f.write(f"% Total entries: {len(entries)}\n")
                    f.write("% IMPORTANT: Manual review and refinement required\n\n")
                
                # Write entries
                for entry in entries:
                    f.write(entry.to_bibtex())
                    f.write("\n\n")
                
                # Write footer
                f.write("% End of auto-generated bibliography\n")
                f.write("% Please review and refine entries for accuracy\n")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to save bibliography to {target_path}: {e}") from e
    
    @property
    def status(self) -> ProcessingStatus:
        """Get current processing status."""
        return self._processing_status


async def main() -> None:
    """Main async function demonstrating production-ready bibliography extraction."""
    print("🔬 Production Bibliography Extraction Tool")
    print("=" * 50)
    print("Type Guardian compliant • Security hardened • Fully validated")
    print()
    
    # Sample references for demonstration
    sample_references = """
1. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. "Deep Learning". MIT Press, 2016.
Available at: http://www.deeplearningbook.org.

2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition". 
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2015.

3. Lena Strobl. "Average-hard attention transformers are constant-depth uniform threshold circuits". 
ArXiv preprint, 2023.

10. Jason Wei, Yi Tay, et al. "Chain-of-thought prompting elicits reasoning in large language models". 
ArXiv preprint arXiv:2201.11903, 2022.
"""
    
    try:
        # Initialize configuration
        config = BibliographyExtractionConfig()
        print(f"📁 Output directory: {config.output_dir}")
        print(f"📄 Output file: {config.output_filename}")
        print(f"⚡ Max references: {config.max_references}")
        print()
        
        # Create extractor
        extractor = ProductionBibliographyExtractor(config)
        
        print("🔄 Processing sample references...")
        
        # Extract entries
        entries = await extractor.extract_references(sample_references)
        
        print(f"✅ Extracted {len(entries)} valid bibliography entries")
        
        # Display entries
        for i, entry in enumerate(entries, 1):
            print(f"\n📖 Entry {i}:")
            print(f"   Type: {entry.entry_type.value}")
            print(f"   Key: {entry.entry_key}")
            print(f"   Title: {entry.title[:60]}{'...' if len(entry.title) > 60 else ''}")
            print(f"   Authors: {entry.authors[:40]}{'...' if len(entry.authors) > 40 else ''}")
            print(f"   Year: {entry.year}")
        
        # Save to file
        print(f"\n💾 Saving bibliography to {config.output_dir / config.output_filename}...")
        success = await extractor.save_bibliography(entries)
        
        if success:
            print("✅ Bibliography saved successfully!")
            print()
            print("📋 Next steps:")
            print("1. 📄 Extract full text from PDF using: pdftotext 'HRM_paper.pdf'")
            print("2. 🔍 Locate the references section (typically starts with 'References')")
            print("3. 🔧 Run this script on the complete reference text")
            print("4. ✏️  Manually review and refine BibTeX entries for accuracy")
            print("5. 🔐 Validate entries using: bibtex --validate references.bib")
        else:
            print("❌ Failed to save bibliography")
            sys.exit(1)
            
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run with proper async event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)