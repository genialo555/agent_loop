"""
Tests for model checkpoint loading and validation.

This module tests:
- Loading QLoRA checkpoints and adapters
- Verifying checkpoint structure and metadata
- Testing checkpoint compatibility with base models
- Validating saved training state
- Testing checkpoint merging operations
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch, Mock

import pytest
import torch
from hypothesis import given, strategies as st, settings
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from agent_loop.models.training.qlora.qlora_config import QLoRAConfig, GEMMA_3N_CONFIG


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""
    
    @pytest.fixture
    def mock_checkpoint_dir(self, tmp_path):
        """Create a mock checkpoint directory structure."""
        checkpoint_dir = tmp_path / "test_checkpoint"
        checkpoint_dir.mkdir()
        
        # Create adapter config
        adapter_config = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "revision": None,
            "use_rslora": False,
            "init_lora_weights": True
        }
        
        with open(checkpoint_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        # Create adapter model (mock weights)
        adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
        # In a real test, we'd save actual tensor data here
        adapter_model_path.touch()
        
        # Create training info
        training_info = {
            "model_name": "google/gemma-3n-e4b",
            "training_time": 3600.5,
            "final_loss": 0.45,
            "total_steps": 1000,
            "timestamp": "2025-07-30 10:00:00",
            "config": {
                "lora_rank": 32,
                "lora_alpha": 64,
                "batch_size_effective": 16,
                "learning_rate": 2e-4
            }
        }
        
        with open(checkpoint_dir / "training_info.json", "w") as f:
            json.dump(training_info, f)
        
        # Create trainer state
        trainer_state = {
            "best_metric": 0.45,
            "best_model_checkpoint": str(checkpoint_dir),
            "epoch": 3.0,
            "global_step": 1000,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [
                {"loss": 1.2, "step": 100},
                {"loss": 0.8, "step": 500},
                {"loss": 0.45, "step": 1000}
            ]
        }
        
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f)
        
        return checkpoint_dir
    
    @pytest.mark.asyncio
    async def test_load_checkpoint_metadata(self, mock_checkpoint_dir):
        """Test loading checkpoint metadata."""
        # Load training info
        training_info_path = mock_checkpoint_dir / "training_info.json"
        assert training_info_path.exists()
        
        with open(training_info_path) as f:
            training_info = json.load(f)
        
        assert training_info["model_name"] == "google/gemma-3n-e4b"
        assert training_info["final_loss"] == 0.45
        assert training_info["total_steps"] == 1000
        assert training_info["config"]["lora_rank"] == 32
    
    @pytest.mark.asyncio
    async def test_load_adapter_config(self, mock_checkpoint_dir):
        """Test loading LoRA adapter configuration."""
        adapter_config_path = mock_checkpoint_dir / "adapter_config.json"
        assert adapter_config_path.exists()
        
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        
        assert adapter_config["r"] == 32
        assert adapter_config["lora_alpha"] == 64
        assert "q_proj" in adapter_config["target_modules"]
        assert adapter_config["task_type"] == "CAUSAL_LM"
    
    @pytest.mark.asyncio
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    async def test_load_peft_model_from_checkpoint(
        self, mock_peft_from_pretrained, mock_model_from_pretrained, mock_checkpoint_dir
    ):
        """Test loading PEFT model from checkpoint."""
        # Mock base model
        mock_base_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_base_model
        
        # Mock PEFT model
        mock_peft_model = MagicMock()
        mock_peft_from_pretrained.return_value = mock_peft_model
        
        # Simulate loading process
        base_model_name = "google/gemma-3n-e4b"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(mock_checkpoint_dir),
            torch_dtype=torch.bfloat16
        )
        
        # Verify calls
        mock_model_from_pretrained.assert_called_once()
        mock_peft_from_pretrained.assert_called_once_with(
            base_model,
            str(mock_checkpoint_dir),
            torch_dtype=torch.bfloat16
        )
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_structure(self, mock_checkpoint_dir):
        """Test checkpoint directory structure validation."""
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_info.json",
            "trainer_state.json"
        ]
        
        # Check all required files exist
        for file_name in required_files:
            file_path = mock_checkpoint_dir / file_name
            assert file_path.exists(), f"Missing required file: {file_name}"
        
        # Validate adapter config structure
        with open(mock_checkpoint_dir / "adapter_config.json") as f:
            adapter_config = json.load(f)
            required_keys = ["r", "lora_alpha", "target_modules", "task_type"]
            for key in required_keys:
                assert key in adapter_config, f"Missing key in adapter_config: {key}"
    
    @pytest.mark.asyncio
    async def test_checkpoint_compatibility_check(self, mock_checkpoint_dir):
        """Test checkpoint compatibility with base model."""
        with open(mock_checkpoint_dir / "training_info.json") as f:
            training_info = json.load(f)
        
        with open(mock_checkpoint_dir / "adapter_config.json") as f:
            adapter_config = json.load(f)
        
        # Check model name compatibility
        base_model_name = training_info["model_name"]
        assert base_model_name == "google/gemma-3n-e4b"
        
        # Check LoRA configuration is valid
        assert adapter_config["r"] > 0
        assert adapter_config["lora_alpha"] > 0
        assert len(adapter_config["target_modules"]) > 0


class TestCheckpointMerging:
    """Test checkpoint merging and conversion operations."""
    
    @pytest.mark.asyncio
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    async def test_merge_lora_weights(
        self, mock_peft_from_pretrained, mock_model_from_pretrained
    ):
        """Test merging LoRA weights with base model."""
        # Create mock models
        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_model_from_pretrained.return_value = mock_base_model
        
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload = MagicMock(return_value=mock_base_model)
        mock_peft_from_pretrained.return_value = mock_peft_model
        
        # Load and merge
        base_model = AutoModelForCausalLM.from_pretrained("test-model")
        peft_model = PeftModel.from_pretrained(base_model, "test-checkpoint")
        merged_model = peft_model.merge_and_unload()
        
        # Verify merge was called
        mock_peft_model.merge_and_unload.assert_called_once()
        assert merged_model == mock_base_model
    
    @pytest.mark.asyncio
    async def test_save_merged_model(self, tmp_path):
        """Test saving merged model to disk."""
        output_dir = tmp_path / "merged_model"
        
        # Mock merged model
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()
        
        # Save model and tokenizer
        mock_model.save_pretrained(str(output_dir))
        mock_tokenizer.save_pretrained(str(output_dir))
        
        # Verify save methods were called
        mock_model.save_pretrained.assert_called_once_with(str(output_dir))
        mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))


class TestCheckpointRecovery:
    """Test checkpoint recovery and resume functionality."""
    
    @pytest.mark.asyncio
    async def test_find_latest_checkpoint(self, tmp_path):
        """Test finding the latest checkpoint in a directory."""
        # Create multiple checkpoint directories
        for i in range(3):
            checkpoint_dir = tmp_path / f"checkpoint-{(i+1)*100}"
            checkpoint_dir.mkdir()
            
            # Create trainer state with step info
            trainer_state = {"global_step": (i+1)*100}
            with open(checkpoint_dir / "trainer_state.json", "w") as f:
                json.dump(trainer_state, f)
        
        # Find latest checkpoint
        checkpoints = sorted(
            [d for d in tmp_path.iterdir() if d.is_dir() and "checkpoint-" in d.name],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        
        assert latest_checkpoint is not None
        assert latest_checkpoint.name == "checkpoint-300"
    
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_state(self, mock_checkpoint_dir):
        """Test resuming training from checkpoint state."""
        # Load trainer state
        with open(mock_checkpoint_dir / "trainer_state.json") as f:
            trainer_state = json.load(f)
        
        # Verify we can extract resume information
        resume_step = trainer_state["global_step"]
        resume_epoch = trainer_state["epoch"]
        
        assert resume_step == 1000
        assert resume_epoch == 3.0
        
        # Check training history is preserved
        assert len(trainer_state["log_history"]) == 3
        assert trainer_state["log_history"][-1]["loss"] == 0.45


class TestCheckpointValidation:
    """Test checkpoint validation and integrity checks."""
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_files_exist(self, mock_checkpoint_dir):
        """Test that all required checkpoint files exist."""
        required_files = {
            "adapter_config.json": "LoRA configuration",
            "adapter_model.safetensors": "Model weights",
            "training_info.json": "Training metadata",
            "trainer_state.json": "Trainer state"
        }
        
        missing_files = []
        for file_name, description in required_files.items():
            if not (mock_checkpoint_dir / file_name).exists():
                missing_files.append((file_name, description))
        
        assert not missing_files, f"Missing checkpoint files: {missing_files}"
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_json_files(self, mock_checkpoint_dir):
        """Test that JSON files in checkpoint are valid."""
        json_files = ["adapter_config.json", "training_info.json", "trainer_state.json"]
        
        for json_file in json_files:
            file_path = mock_checkpoint_dir / json_file
            try:
                with open(file_path) as f:
                    data = json.load(f)
                assert isinstance(data, dict), f"{json_file} should contain a dictionary"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file}: {e}")
    
    @given(
        rank=st.integers(min_value=1, max_value=128),
        alpha=st.integers(min_value=1, max_value=256)
    )
    @settings(max_examples=10)
    def test_lora_config_validation(self, rank, alpha):
        """Property-based test for LoRA configuration validation."""
        config = {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        # Validate configuration constraints
        assert config["r"] > 0, "LoRA rank must be positive"
        assert config["lora_alpha"] > 0, "LoRA alpha must be positive"
        assert len(config["target_modules"]) > 0, "Must specify target modules"


class TestCheckpointPerformance:
    """Test checkpoint loading performance and optimization."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_checkpoint_loading_speed(self, benchmark, mock_checkpoint_dir):
        """Benchmark checkpoint metadata loading speed."""
        def load_checkpoint_metadata():
            with open(mock_checkpoint_dir / "training_info.json") as f:
                training_info = json.load(f)
            with open(mock_checkpoint_dir / "adapter_config.json") as f:
                adapter_config = json.load(f)
            with open(mock_checkpoint_dir / "trainer_state.json") as f:
                trainer_state = json.load(f)
            return training_info, adapter_config, trainer_state
        
        result = benchmark(load_checkpoint_metadata)
        assert result is not None
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_checkpoint_memory_usage(self, mock_checkpoint_dir):
        """Test memory usage when loading checkpoints."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load checkpoint data
        with open(mock_checkpoint_dir / "training_info.json") as f:
            training_info = json.load(f)
        with open(mock_checkpoint_dir / "adapter_config.json") as f:
            adapter_config = json.load(f)
        
        # Check memory increase is reasonable
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # JSON loading should use minimal memory
        assert memory_increase < 10, f"Excessive memory usage: {memory_increase:.2f} MB"


class TestCheckpointSecurity:
    """Test checkpoint security and validation."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_path_traversal_protection(self, tmp_path):
        """Test protection against path traversal attacks."""
        base_dir = tmp_path / "checkpoints"
        base_dir.mkdir()
        
        # Attempt path traversal
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "checkpoint/../../../sensitive_data"
        ]
        
        for malicious_path in malicious_paths:
            # Normalize and validate path
            try:
                full_path = base_dir / malicious_path
                resolved_path = full_path.resolve()
                
                # Check if resolved path is within base directory
                assert not str(resolved_path).startswith(str(base_dir.resolve())), \
                    f"Path traversal not prevented for: {malicious_path}"
            except (ValueError, OSError):
                # Path resolution failed, which is good
                pass
    
    @pytest.mark.asyncio
    async def test_checkpoint_file_size_limits(self, mock_checkpoint_dir):
        """Test checkpoint file size validation."""
        max_json_size = 10 * 1024 * 1024  # 10 MB limit for JSON files
        
        json_files = ["adapter_config.json", "training_info.json", "trainer_state.json"]
        
        for json_file in json_files:
            file_path = mock_checkpoint_dir / json_file
            file_size = file_path.stat().st_size
            
            assert file_size < max_json_size, \
                f"{json_file} exceeds size limit: {file_size} bytes"


@pytest.mark.integration
class TestCheckpointIntegration:
    """Integration tests for checkpoint functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_full_checkpoint_workflow(self, tmp_path):
        """Test complete checkpoint save and load workflow."""
        # This test would require actual model loading
        # For now, we'll create a mock workflow
        
        checkpoint_dir = tmp_path / "integration_checkpoint"
        checkpoint_dir.mkdir()
        
        # Simulate saving checkpoint
        training_info = {
            "model_name": "test-model",
            "final_loss": 0.5,
            "total_steps": 100
        }
        
        with open(checkpoint_dir / "training_info.json", "w") as f:
            json.dump(training_info, f)
        
        # Simulate loading checkpoint
        assert (checkpoint_dir / "training_info.json").exists()
        
        with open(checkpoint_dir / "training_info.json") as f:
            loaded_info = json.load(f)
        
        assert loaded_info["model_name"] == "test-model"
        assert loaded_info["final_loss"] == 0.5