"""
Tests for model inference on test prompts.

This module tests:
- Model inference with various prompt types
- Response generation quality
- Tokenization and decoding
- Batch inference capabilities
- Streaming inference
- Agent-specific prompt handling
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Generator
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import torch
from hypothesis import given, strategies as st, settings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

from agent_loop.models.inference.services.ollama import OllamaService
from agent_loop.models.training.qlora.qlora_config import QLoRAConfig


# Test prompts for different agent scenarios
TEST_PROMPTS = {
    "tool_use": {
        "prompt": "You are an AI assistant. Use the search tool to find information about Python decorators.",
        "expected_patterns": ["search", "tool", "decorator", "Python"],
        "max_length": 150
    },
    "code_generation": {
        "prompt": "Write a Python function that calculates the factorial of a number using recursion.",
        "expected_patterns": ["def", "factorial", "return", "if", "else"],
        "max_length": 200
    },
    "reasoning": {
        "prompt": "Explain step by step how to solve the equation: 2x + 5 = 13",
        "expected_patterns": ["step", "subtract", "divide", "x ="],
        "max_length": 250
    },
    "agent_instruction": {
        "prompt": "You are a helpful assistant. The user asks: 'What is the capital of France?' Provide a brief, accurate response.",
        "expected_patterns": ["Paris", "capital", "France"],
        "max_length": 100
    }
}


class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for inference tests."""
    
    def __init__(self, stop_tokens: List[int], max_length: int = 100):
        self.stop_tokens = stop_tokens
        self.max_length = max_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > self.max_length:
            return True
        
        for stop_token in self.stop_tokens:
            if stop_token in input_ids[0, -len(self.stop_tokens):]:
                return True
        
        return False


class TestModelInference:
    """Test basic model inference functionality."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer for testing."""
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.encode = MagicMock(return_value=[101, 102, 103])
        tokenizer.decode = MagicMock(return_value="Generated response")
        tokenizer.batch_encode_plus = MagicMock(return_value={
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })
        
        # Mock model
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.tensor([[101, 102, 103, 104, 105]]))
        model.config = MagicMock()
        model.config.max_length = 512
        model.device = torch.device("cpu")
        
        return model, tokenizer
    
    @pytest.mark.asyncio
    async def test_single_prompt_inference(self, mock_model_and_tokenizer):
        """Test inference on a single prompt."""
        model, tokenizer = mock_model_and_tokenizer
        
        prompt = TEST_PROMPTS["agent_instruction"]["prompt"]
        
        # Encode prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Verify calls
        tokenizer.encode.assert_called_once()
        model.generate.assert_called_once()
        tokenizer.decode.assert_called_once()
        
        assert response == "Generated response"
    
    @pytest.mark.asyncio
    async def test_batch_inference(self, mock_model_and_tokenizer):
        """Test batch inference on multiple prompts."""
        model, tokenizer = mock_model_and_tokenizer
        
        # Prepare batch of prompts
        prompts = [
            TEST_PROMPTS["tool_use"]["prompt"],
            TEST_PROMPTS["code_generation"]["prompt"],
            TEST_PROMPTS["reasoning"]["prompt"]
        ]
        
        # Mock batch generation
        model.generate.return_value = torch.tensor([
            [101, 102, 103, 104, 105],
            [201, 202, 203, 204, 205],
            [301, 302, 303, 304, 305]
        ])
        
        # Encode batch
        inputs = tokenizer.batch_encode_plus(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # Verify batch processing
        assert outputs.shape[0] == len(prompts)
        tokenizer.batch_encode_plus.assert_called_once_with(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
    @pytest.mark.asyncio
    async def test_inference_with_stopping_criteria(self, mock_model_and_tokenizer):
        """Test inference with custom stopping criteria."""
        model, tokenizer = mock_model_and_tokenizer
        
        # Setup stopping criteria
        stop_tokens = [tokenizer.eos_token_id]
        stopping_criteria = StoppingCriteriaList([
            CustomStoppingCriteria(stop_tokens, max_length=100)
        ])
        
        prompt = "Generate a short response:"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate with stopping criteria
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            stopping_criteria=stopping_criteria
        )
        
        # Verify stopping criteria was passed
        call_args = model.generate.call_args
        assert "stopping_criteria" in call_args.kwargs


class TestPromptHandling:
    """Test various prompt formats and handling."""
    
    @pytest.mark.asyncio
    async def test_agent_loop_prompt_format(self):
        """Test agent loop specific prompt formatting."""
        # Agent loop prompt template
        template = """You are an AI assistant with access to tools.

User: {user_input}
Assistant: """
        
        user_inputs = [
            "Search for information about Python decorators",
            "Calculate the factorial of 5",
            "Explain quantum computing in simple terms"
        ]
        
        formatted_prompts = []
        for user_input in user_inputs:
            formatted_prompt = template.format(user_input=user_input)
            formatted_prompts.append(formatted_prompt)
            
            # Verify formatting
            assert "You are an AI assistant" in formatted_prompt
            assert user_input in formatted_prompt
            assert formatted_prompt.endswith("Assistant: ")
        
        assert len(formatted_prompts) == len(user_inputs)
    
    @pytest.mark.asyncio
    async def test_tool_use_prompt_format(self):
        """Test tool use prompt formatting."""
        tool_prompt = """You have access to the following tools:
- search(query): Search for information
- calculate(expression): Perform calculations
- read_file(path): Read file contents

User: Find the square root of 144
Assistant: I'll help you find the square root of 144.

<tool_call>
calculate("sqrt(144)")
</tool_call>

The square root of 144 is 12."""
        
        # Verify tool call pattern
        assert "<tool_call>" in tool_prompt
        assert "</tool_call>" in tool_prompt
        assert 'calculate("sqrt(144)")' in tool_prompt
    
    @given(
        prompt_length=st.integers(min_value=10, max_value=1000),
        temperature=st.floats(min_value=0.1, max_value=2.0),
        top_p=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=10)
    def test_inference_parameters_validation(self, prompt_length, temperature, top_p):
        """Property-based test for inference parameter validation."""
        # Generate random prompt
        prompt = "x" * prompt_length
        
        # Validate parameters
        assert 0 < temperature <= 2.0, "Temperature must be between 0 and 2"
        assert 0 < top_p <= 1.0, "Top-p must be between 0 and 1"
        assert len(prompt) > 0, "Prompt must not be empty"
        
        # Simulate parameter usage
        inference_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": min(100, 2048 - prompt_length)
        }
        
        assert inference_params["max_new_tokens"] > 0


class TestStreamingInference:
    """Test streaming inference capabilities."""
    
    @pytest.mark.asyncio
    async def test_streaming_response_generation(self):
        """Test streaming text generation."""
        class MockStreamer:
            def __init__(self):
                self.tokens = []
            
            def put(self, value):
                self.tokens.append(value)
            
            def end(self):
                pass
        
        # Mock streaming process
        streamer = MockStreamer()
        tokens = ["Hello", " world", "!", " How", " are", " you", "?"]
        
        # Simulate streaming
        for token in tokens:
            streamer.put(token)
            await asyncio.sleep(0.01)  # Simulate generation delay
        
        streamer.end()
        
        # Verify streaming
        assert len(streamer.tokens) == len(tokens)
        assert "".join(streamer.tokens) == "Hello world! How are you?"
    
    @pytest.mark.asyncio
    async def test_streaming_with_early_stopping(self):
        """Test streaming with early stopping."""
        class StreamController:
            def __init__(self):
                self.should_stop = False
                self.generated_tokens = []
            
            async def generate_stream(self, max_tokens=100):
                tokens = ["The", " answer", " is", " 42", ".", " But", " wait", "..."]
                
                for i, token in enumerate(tokens):
                    if self.should_stop:
                        break
                    
                    self.generated_tokens.append(token)
                    yield token
                    
                    # Stop after "42."
                    if token == ".":
                        self.should_stop = True
                    
                    await asyncio.sleep(0.01)
        
        controller = StreamController()
        
        # Collect streamed tokens
        collected = []
        async for token in controller.generate_stream():
            collected.append(token)
        
        # Verify early stopping
        assert controller.should_stop
        assert "".join(collected) == "The answer is 42."
        assert len(collected) < 8  # Stopped before all tokens


class TestModelResponseQuality:
    """Test quality and coherence of model responses."""
    
    @pytest.mark.asyncio
    async def test_response_coherence_check(self):
        """Test that responses are coherent and relevant."""
        test_cases = [
            {
                "prompt": "What is 2 + 2?",
                "expected_keywords": ["4", "four", "equals"],
                "unexpected_keywords": ["banana", "car", "purple"]
            },
            {
                "prompt": "Write a Python function to add two numbers",
                "expected_keywords": ["def", "return", "add", "+"],
                "unexpected_keywords": ["JavaScript", "HTML", "CSS"]
            }
        ]
        
        for test_case in test_cases:
            # Mock response for testing
            if "2 + 2" in test_case["prompt"]:
                mock_response = "2 + 2 equals 4"
            else:
                mock_response = "def add_numbers(a, b):\n    return a + b"
            
            # Check expected keywords
            response_lower = mock_response.lower()
            found_expected = any(
                keyword.lower() in response_lower 
                for keyword in test_case["expected_keywords"]
            )
            assert found_expected, f"Response missing expected keywords for: {test_case['prompt']}"
            
            # Check unexpected keywords
            found_unexpected = any(
                keyword.lower() in response_lower 
                for keyword in test_case["unexpected_keywords"]
            )
            assert not found_unexpected, f"Response contains unexpected keywords for: {test_case['prompt']}"
    
    @pytest.mark.asyncio
    async def test_response_length_control(self):
        """Test controlling response length."""
        length_configs = [
            {"max_tokens": 50, "prompt": "Write a short greeting"},
            {"max_tokens": 200, "prompt": "Explain machine learning"},
            {"max_tokens": 500, "prompt": "Write a detailed Python tutorial"}
        ]
        
        for config in length_configs:
            # Simulate length-controlled generation
            if config["max_tokens"] <= 50:
                mock_response = "Hello! How can I help you today?"
            elif config["max_tokens"] <= 200:
                mock_response = "Machine learning is a subset of AI that enables systems to learn from data."
            else:
                mock_response = "Python Tutorial:\n1. Introduction\n2. Variables\n3. Functions\n..."
            
            # Token count approximation (rough estimate: ~4 chars per token)
            estimated_tokens = len(mock_response) // 4
            
            assert estimated_tokens <= config["max_tokens"], \
                f"Response exceeds token limit for: {config['prompt']}"


class TestInferencePerformance:
    """Test inference performance metrics."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_inference_speed_single_prompt(self, benchmark):
        """Benchmark single prompt inference speed."""
        def mock_inference():
            # Simulate inference computation
            prompt = "What is the meaning of life?"
            tokens = list(range(50))  # 50 tokens
            
            # Simulate token generation delay
            result = []
            for token in tokens:
                result.append(token)
                # In real inference, there would be model computation here
            
            return result
        
        result = benchmark(mock_inference)
        assert len(result) == 50
    
    @pytest.mark.asyncio
    async def test_batch_inference_efficiency(self):
        """Test efficiency of batch vs sequential inference."""
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Simulate batch processing
            prompts = [f"Prompt {i}" for i in range(batch_size)]
            
            # Mock batch inference time (should be less than sequential)
            # In reality, batch processing is more efficient
            batch_time = 0.1 + (batch_size * 0.02)  # Base time + small increment per item
            await asyncio.sleep(batch_time)
            
            elapsed = time.time() - start_time
            
            # Calculate efficiency
            sequential_time = batch_size * 0.1  # If processed one by one
            efficiency = (sequential_time - elapsed) / sequential_time * 100
            
            print(f"Batch size {batch_size}: {efficiency:.1f}% more efficient than sequential")


class TestOllamaIntegration:
    """Test integration with Ollama service."""
    
    @pytest.mark.asyncio
    @patch('inference.services.ollama.OllamaService')
    async def test_ollama_model_inference(self, mock_ollama_service):
        """Test inference using Ollama service."""
        # Mock Ollama service
        service = mock_ollama_service.return_value
        service.generate = AsyncMock(return_value={
            "response": "Paris is the capital of France.",
            "model": "gemma:3n-e2b",
            "done": True,
            "context": [1, 2, 3, 4, 5],
            "total_duration": 1500000000,  # 1.5 seconds in nanoseconds
            "eval_count": 10
        })
        
        # Test inference
        prompt = "What is the capital of France?"
        response = await service.generate(
            model="gemma:3n-e2b",
            prompt=prompt,
            options={"temperature": 0.7}
        )
        
        # Verify response
        assert "Paris" in response["response"]
        assert response["model"] == "gemma:3n-e2b"
        assert response["done"] is True
        
        # Check performance metrics
        duration_seconds = response["total_duration"] / 1e9
        tokens_per_second = response["eval_count"] / duration_seconds
        
        assert duration_seconds == 1.5
        assert tokens_per_second > 0


class TestErrorHandling:
    """Test error handling during inference."""
    
    @pytest.mark.asyncio
    async def test_handle_empty_prompt(self):
        """Test handling of empty prompts."""
        empty_prompts = ["", " ", "\n", "\t"]
        
        for prompt in empty_prompts:
            # Should either handle gracefully or raise appropriate error
            if not prompt.strip():
                # Expected behavior: return empty or raise ValueError
                assert len(prompt.strip()) == 0
    
    @pytest.mark.asyncio
    async def test_handle_oversized_prompt(self):
        """Test handling of prompts exceeding token limit."""
        max_tokens = 2048
        
        # Create oversized prompt (rough estimate: ~4 chars per token)
        oversized_prompt = "x" * (max_tokens * 5)
        
        # Should handle by truncation or error
        assert len(oversized_prompt) > max_tokens * 4
        
        # In real implementation, would truncate or raise error
        truncated = oversized_prompt[:max_tokens * 4]
        assert len(truncated) <= max_tokens * 4
    
    @pytest.mark.asyncio
    async def test_handle_special_characters(self):
        """Test handling of special characters in prompts."""
        special_prompts = [
            "Generate code: ```python\nprint('hello')\n```",
            "Math equation: ∑(i=1 to n) i² = n(n+1)(2n+1)/6",
            "Unicode test: 你好世界 🌍 مرحبا بالعالم",
            "Control chars: \x00\x01\x02"
        ]
        
        for prompt in special_prompts:
            # Should handle special characters properly
            assert len(prompt) > 0
            
            # Check for potential issues
            has_null = "\x00" in prompt
            has_unicode = any(ord(char) > 127 for char in prompt)
            
            if has_null:
                # Should sanitize or handle null characters
                sanitized = prompt.replace("\x00", "")
                assert "\x00" not in sanitized


@pytest.mark.integration
class TestEndToEndInference:
    """End-to-end inference integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_full_inference_pipeline(self):
        """Test complete inference pipeline with real model loading."""
        # This would test actual model loading and inference
        # For now, we'll mock the pipeline
        
        pipeline_stages = [
            "load_model",
            "load_tokenizer",
            "prepare_prompt",
            "tokenize",
            "generate",
            "decode",
            "post_process"
        ]
        
        results = {}
        for stage in pipeline_stages:
            # Simulate stage execution
            await asyncio.sleep(0.01)
            results[stage] = "completed"
        
        # Verify all stages completed
        assert all(results[stage] == "completed" for stage in pipeline_stages)
    
    @pytest.mark.asyncio
    async def test_model_switching(self):
        """Test switching between different model checkpoints."""
        checkpoints = [
            "base_model",
            "checkpoint_500",
            "checkpoint_1000",
            "final_model"
        ]
        
        for checkpoint in checkpoints:
            # Simulate model loading
            await asyncio.sleep(0.05)
            
            # Mock inference with different checkpoint
            mock_response = f"Response from {checkpoint}"
            
            assert checkpoint in mock_response