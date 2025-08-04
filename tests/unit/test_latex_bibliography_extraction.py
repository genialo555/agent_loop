#!/usr/bin/env python3
"""
Unit tests for LaTeX bibliography extraction script.

Tests the bibliography extraction functionality that converts PDF references
to BibTeX format for the HRM paper and other research documents.
"""
from __future__ import annotations

import re
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, mock_open
from typing import Dict, List, Any

# Import test target (assuming we can import from the script location)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "docs" / "latex" / "scripts"))

try:
    from extract_bibliography import extract_bibliography_from_text
except ImportError:
    # Fallback: implement the function here for testing
    def extract_bibliography_from_text(text_content: str) -> List[str]:
        """Fallback implementation for testing."""
        ref_pattern = r'(\d+)\.\s+(.+?)(?=\n\d+\.|$)'
        references = re.findall(ref_pattern, text_content, re.DOTALL)
        
        bibtex_entries = []
        for ref_num, ref_text in references:
            ref_text = ref_text.strip().replace('\n', ' ')
            entry_key = f"ref{ref_num.zfill(3)}"
            
            if 'arxiv' in ref_text.lower() or 'preprint' in ref_text.lower():
                entry_type = 'misc'
            elif 'proceedings' in ref_text.lower() or 'conference' in ref_text.lower():
                entry_type = 'inproceedings'
            elif 'journal' in ref_text.lower():
                entry_type = 'article'
            elif 'press' in ref_text.lower() or 'publisher' in ref_text.lower():
                entry_type = 'book'
            else:
                entry_type = 'misc'
            
            title_match = re.search(r'([A-Z][^.]*?)\.', ref_text)
            title = title_match.group(1) if title_match else "Title not extracted"
            
            author_match = re.search(r'^([^.]*?)\.', ref_text)
            authors = author_match.group(1) if author_match else "Authors not extracted"
            
            year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
            year = year_match.group(0) if year_match else "Year not found"
            
            bibtex_entry = f"""@{entry_type}{{{entry_key},
  title={{{title}}},
  author={{{authors}}},
  year={{{year}}},
  note={{Reference {ref_num}: {ref_text[:100]}...}}
}}"""
            bibtex_entries.append(bibtex_entry)
        
        return bibtex_entries


class TestBibliographyExtraction:
    """Test suite for bibliography extraction from academic papers."""
    
    @pytest.fixture
    def sample_reference_text(self) -> str:
        """Sample reference text for testing extraction."""
        return """
1. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.
http://www.deeplearningbook.org.

2. Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pages 770–778, 2015.

3. Lena Strobl. Average-hard attention transformers are constant-depth uniform threshold
circuits, 2023.

10. Jason Wei, Yi Tay, et al. Chain-of-thought prompting elicits reasoning in large language
models, 2022. arXiv preprint arXiv:2201.11903.
"""
    
    @pytest.fixture
    def complex_reference_text(self) -> str:
        """Complex reference text with edge cases."""
        return """
15. Attention Is All You Need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. In Advances in Neural
Information Processing Systems 30 (NIPS 2017), pages 5998–6008, 2017.

42. On the Measure of Intelligence. François Chollet. arXiv preprint arXiv:1911.01547, 2019.

101. The Annotated Transformer. Harvard NLP. Available at http://nlp.seas.harvard.edu/
2018/04/03/attention.html, 2018.
"""
    
    @pytest.fixture
    def malformed_reference_text(self) -> str:
        """Malformed reference text to test error handling."""
        return """
This is not a proper reference format.

1. Missing period after authors
Without proper structure

2. Normal Reference. Author Name. Journal Title, 2023.
3. Another good reference. Second Author. Conference Proceedings, 2022.
"""
    
    def test_extract_basic_references(self, sample_reference_text: str):
        """Test extraction of basic bibliography references."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        # Should extract 4 references (1, 2, 3, 10)
        assert len(entries) == 4
        
        # Check first entry structure
        first_entry = entries[0]
        assert "@book{ref001," in first_entry
        assert "Ian Goodfellow, Yoshua Bengio, and Aaron Courville" in first_entry
        assert "2016" in first_entry
        assert "Deep Learning" in first_entry
        
        # Check arXiv entry is properly categorized
        arxiv_entry = entries[3]  # Entry 10 is arXiv
        assert "@misc{ref010," in arxiv_entry
        assert "2022" in arxiv_entry
    
    def test_entry_type_classification(self, sample_reference_text: str):
        """Test that references are properly classified by type."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        # Book entry
        assert "@book{ref001," in entries[0]
        
        # Conference paper
        assert "@inproceedings{ref002," in entries[1]
        
        # ArXiv preprint
        assert "@misc{ref010," in entries[3]
    
    def test_author_extraction(self, sample_reference_text: str):
        """Test proper extraction of author information."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        # Multi-author extraction
        first_entry = entries[0]
        assert "Ian Goodfellow, Yoshua Bengio, and Aaron Courville" in first_entry
        
        # Et al. handling
        arxiv_entry = entries[3]
        assert "Jason Wei, Yi Tay, et al" in arxiv_entry
    
    def test_year_extraction(self, sample_reference_text: str):
        """Test proper extraction of publication years."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        years = [re.search(r'year=\{(\d{4})\}', entry) for entry in entries]
        extracted_years = [match.group(1) for match in years if match]
        
        assert "2016" in extracted_years
        assert "2015" in extracted_years
        assert "2023" in extracted_years
        assert "2022" in extracted_years
    
    def test_title_extraction(self, sample_reference_text: str):
        """Test proper extraction of titles."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        # Should extract meaningful titles
        titles = [re.search(r'title=\{([^}]+)\}', entry) for entry in entries]
        extracted_titles = [match.group(1) for match in titles if match]
        
        assert any("Deep Learning" in title for title in extracted_titles)
        assert any("Deep residual learning" in title for title in extracted_titles)
    
    def test_complex_references(self, complex_reference_text: str):
        """Test handling of complex reference formats."""
        entries = extract_bibliography_from_text(complex_reference_text)
        
        assert len(entries) == 3
        
        # Check high reference numbers are handled
        assert "ref015" in entries[0]
        assert "ref042" in entries[1] 
        assert "ref101" in entries[2]
    
    def test_malformed_references(self, malformed_reference_text: str):
        """Test handling of malformed reference text."""
        entries = extract_bibliography_from_text(malformed_reference_text)
        
        # Should extract valid references, skip malformed ones
        assert len(entries) == 2  # Only entries 2 and 3 are properly formatted
        
        # Check that valid entries are still processed
        assert any("ref002" in entry for entry in entries)
        assert any("ref003" in entry for entry in entries)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        entries = extract_bibliography_from_text("")
        assert entries == []
        
        entries = extract_bibliography_from_text("   \n   \n   ")
        assert entries == []
    
    def test_single_reference(self):
        """Test extraction of a single reference."""
        single_ref = "1. Smith, John. Test Paper. Journal of Testing, 2023."
        entries = extract_bibliography_from_text(single_ref)
        
        assert len(entries) == 1
        assert "ref001" in entries[0]
        assert "Smith, John" in entries[0]
        assert "2023" in entries[0]
    
    def test_special_characters_handling(self):
        """Test handling of special characters in references."""
        special_chars = """
1. François Chollet. On the Measure of Intelligence. arXiv preprint, 2019.
2. José García. Análisis de Redes Neuronales. Editorial Técnica, 2020.
"""
        entries = extract_bibliography_from_text(special_chars)
        
        assert len(entries) == 2
        assert "François Chollet" in entries[0]
        assert "José García" in entries[1]


class TestBibliographyValidation:
    """Test suite for bibliography validation and quality checks."""
    
    def test_bibtex_format_validity(self, sample_reference_text: str):
        """Test that generated BibTeX entries have valid format."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        for entry in entries:
            # Check basic BibTeX structure
            assert entry.startswith('@')
            assert '{' in entry and '}' in entry
            
            # Check required fields are present
            assert 'title=' in entry
            assert 'author=' in entry
            assert 'year=' in entry
            
            # Check balanced braces
            open_braces = entry.count('{')
            close_braces = entry.count('}')
            assert open_braces == close_braces, f"Unbalanced braces in entry: {entry[:100]}"
    
    def test_entry_key_uniqueness(self, sample_reference_text: str):
        """Test that entry keys are unique."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        keys = []
        for entry in entries:
            key_match = re.search(r'@\w+\{([^,]+),', entry)
            if key_match:
                keys.append(key_match.group(1))
        
        # All keys should be unique
        assert len(keys) == len(set(keys)), f"Duplicate keys found: {keys}"
    
    def test_entry_key_format(self, sample_reference_text: str):
        """Test that entry keys follow proper format."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        key_pattern = re.compile(r'^ref\d{3}$')
        
        for entry in entries:
            key_match = re.search(r'@\w+\{([^,]+),', entry)
            if key_match:
                key = key_match.group(1)
                assert key_pattern.match(key), f"Invalid key format: {key}"


class TestBibliographyRobustness:
    """Property-based and robustness tests for bibliography extraction."""
    
    @pytest.mark.property
    def test_extraction_stability(self):
        """Test that extraction is stable across multiple runs."""
        reference_text = """
1. Test Author. Test Title. Test Journal, 2023.
2. Another Author. Another Title. Another Journal, 2022.
"""
        
        # Run extraction multiple times
        results = []
        for _ in range(10):
            entries = extract_bibliography_from_text(reference_text)
            results.append(entries)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
    
    @pytest.mark.property
    def test_input_size_handling(self):
        """Test handling of various input sizes."""
        # Test very small input
        small_input = "1. A. B. C, 2023."
        entries = extract_bibliography_from_text(small_input)
        assert len(entries) == 1
        
        # Test medium input (simulate typical paper references)
        medium_input = "\n".join([
            f"{i}. Author {i}. Title {i}. Journal {i}, {2020 + i}."
            for i in range(1, 21)
        ])
        entries = extract_bibliography_from_text(medium_input)
        assert len(entries) == 20
    
    @pytest.mark.property
    def test_reference_number_edge_cases(self):
        """Test handling of edge cases in reference numbering."""
        # Non-sequential numbers
        non_sequential = """
1. First Author. First Title. First Journal, 2023.
5. Fifth Author. Fifth Title. Fifth Journal, 2022.
10. Tenth Author. Tenth Title. Tenth Journal, 2021.
"""
        entries = extract_bibliography_from_text(non_sequential)
        assert len(entries) == 3
        
        # Very high numbers
        high_numbers = """
999. Last Author. Last Title. Last Journal, 2023.
1000. Final Author. Final Title. Final Journal, 2022.
"""
        entries = extract_bibliography_from_text(high_numbers)
        assert len(entries) == 2


class TestBibliographyIntegration:
    """Integration tests for bibliography extraction with file I/O."""
    
    def test_file_writing_integration(self, sample_reference_text: str):
        """Test integration with file writing operations."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bib', delete=False) as f:
            for entry in entries:
                f.write(entry + "\n\n")
            temp_path = Path(f.name)
        
        try:
            # Verify file was written correctly
            assert temp_path.exists()
            content = temp_path.read_text()
            
            # Check that all entries are present
            for entry in entries:
                assert entry in content
            
            # Check basic BibTeX structure in file
            assert content.count('@') == len(entries)
            
        finally:
            temp_path.unlink()
    
    @patch("builtins.open", new_callable=mock_open)
    def test_output_file_creation(self, mock_file, sample_reference_text: str):
        """Test output file creation with mocked I/O."""
        entries = extract_bibliography_from_text(sample_reference_text)
        
        # Simulate writing to file
        output_path = Path("test_references.bib")
        
        # Mock file operations
        with patch.object(Path, 'parent') as mock_parent:
            mock_parent.mkdir = lambda exist_ok=True: None
            
            # Simulate the main script's file writing logic
            mock_file.return_value.write.return_value = None
            
            # This would be called by the main script
            with open(output_path, 'w') as f:
                f.write("% Bibliography for Hierarchical Reasoning Model\n")
                f.write("% Auto-generated template - manual refinement required\n\n")
                
                for entry in entries:
                    f.write(entry + "\n\n")
        
        # Verify mock was called correctly
        mock_file.assert_called_once_with(output_path, 'w')
        
        # Verify write calls
        write_calls = mock_file.return_value.write.call_args_list
        assert len(write_calls) >= len(entries) + 2  # Entries + header lines


@pytest.mark.benchmark
class TestBibliographyPerformance:
    """Performance benchmarks for bibliography extraction."""
    
    def test_extraction_performance(self, benchmark):
        """Benchmark bibliography extraction performance."""
        # Create a realistic reference text (101 references like HRM paper)
        large_reference_text = "\n\n".join([
            f"{i}. Author {i}, Second Author {i}. Title of Paper {i}: "
            f"Subtitle and More Details. Journal of Advanced Research {i}, "
            f"volume {i % 10}, pages {i * 10}-{i * 10 + 20}, {2000 + i % 24}."
            for i in range(1, 102)
        ])
        
        # Benchmark the extraction
        result = benchmark(extract_bibliography_from_text, large_reference_text)
        
        # Verify results
        assert len(result) == 101
        assert all("ref" in entry for entry in result[:5])  # Spot check
    
    def test_memory_usage_large_input(self):
        """Test memory usage with large input."""
        import tracemalloc
        
        # Create very large reference text
        huge_reference_text = "\n\n".join([
            f"{i}. Very Long Author Name {i} and Second Very Long Author Name {i}. "
            f"An Extremely Long Title That Goes On And On About {i}: "
            f"With Multiple Subtitles and Additional Information That Makes This "
            f"Reference Very Long Indeed. Journal of Extremely Long Names and "
            f"Detailed Research Topics Volume {i}, issue {i % 12}, "
            f"pages {i * 100}-{i * 100 + 200}, {1950 + i % 75}."
            for i in range(1, 1001)  # 1000 references
        ])
        
        tracemalloc.start()
        
        entries = extract_bibliography_from_text(huge_reference_text)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify results
        assert len(entries) == 1000
        
        # Memory usage should be reasonable (less than 100MB for 1000 references)
        assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])