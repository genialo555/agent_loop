"""Tests for groupthink inference module."""

import pytest
from inference.groupthink import generate


class TestGroupthink:
    """Test cases for groupthink module following TST001 pattern."""

    def test_generate_basic_functionality(self):
        """Test basic generate function functionality."""
        # Arrange
        instruction = "Test instruction"
        
        # Act
        result = generate(instruction)
        
        # Assert
        assert isinstance(result, str)
        assert instruction in result
        assert "Stub" in result

    def test_generate_with_custom_threads(self):
        """Test generate function with custom thread count."""
        # Arrange
        instruction = "Custom thread test"
        n_threads = 2
        
        # Act
        result = generate(instruction, n_threads)
        
        # Assert
        assert isinstance(result, str)
        assert instruction in result
        # Should return one of the generated answers (all same length in stub)
        assert "Answer" in result and "Stub" in result

    def test_generate_with_single_thread(self):
        """Test generate function with single thread."""
        # Arrange
        instruction = "Single thread test"
        n_threads = 1
        
        # Act
        result = generate(instruction, n_threads)
        
        # Assert
        assert isinstance(result, str)
        assert "Answer 1" in result

    def test_generate_with_zero_threads(self):
        """Test generate function with zero threads."""
        # Arrange
        instruction = "Zero thread test"
        n_threads = 0
        
        # Act & Assert
        # Should raise ValueError for empty list
        with pytest.raises(ValueError):
            generate(instruction, n_threads)

    @pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
    def test_generate_thread_scaling(self, n_threads):
        """Test generate function with different thread counts."""
        # Arrange
        instruction = "Thread scaling test"
        
        # Act
        result = generate(instruction, n_threads)
        
        # Assert
        assert isinstance(result, str)
        if n_threads > 0:
            assert instruction in result
            assert "Stub" in result

    def test_generate_empty_instruction(self):
        """Test generate function with empty instruction."""
        # Arrange
        instruction = ""
        
        # Act
        result = generate(instruction)
        
        # Assert
        assert isinstance(result, str)
        # Should still work with empty instruction

    def test_generate_long_instruction(self):
        """Test generate function with long instruction."""
        # Arrange
        instruction = "A" * 1000  # Very long instruction
        
        # Act
        result = generate(instruction)
        
        # Assert
        assert isinstance(result, str)
        assert instruction in result

    @pytest.mark.benchmark
    def test_generate_performance(self, benchmark):
        """Benchmark generate function performance (TST007)."""
        # Arrange
        instruction = "Performance test instruction"
        
        # Act & Assert
        result = benchmark(generate, instruction)
        assert isinstance(result, str)
        assert instruction in result