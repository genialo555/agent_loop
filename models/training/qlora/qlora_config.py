#!/usr/bin/env python3
"""
QLoRA Configuration Module for Sprint 2
Modern 4-bit quantization setup optimized for RTX 4090 24GB GPU.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType


@dataclass
class QLoRAConfig:
    """Modern QLoRA configuration optimized for 2025 best practices."""
    
    # Model configuration
    model_name: str = "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"  # Using local Gemma 3N-E4B path
    cache_dir: Optional[str] = "/media/jerem/641C8D6C1C8D3A56/hf_cache"
    trust_remote_code: bool = False
    
    # 4-bit Quantization (NF4 + Double Quantization)
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True  # Memory optimization
    bnb_4bit_quant_type: str = "nf4"        # Normal Float 4-bit (optimal)
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16  # Better than fp16 on Ampere
    llm_int8_enable_fp32_cpu_offload: bool = True
    
    # LoRA Configuration (2025 best practices)
    lora_r: int = 32                        # Rank increased for better performance
    lora_alpha: int = 64                    # 2:1 ratio with rank
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: str = "all-linear"      # Target all linear layers (2025 practice)
    modules_to_save: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    
    # RTX 4090 Optimized Training Parameters
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4    # Effective batch size = 16
    warmup_steps: int = 100
    max_steps: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    
    # Mixed Precision (RTX 4090 optimized)
    fp16: bool = False                      # Use bf16 instead on Ampere
    bf16: bool = True                       # Better numerical stability
    tf32: bool = True                       # Enable TensorFloat-32 on Ampere
    
    # Memory Optimization
    gradient_checkpointing: bool = False  # Disabled for Gemma-3N compatibility
    dataloader_pin_memory: bool = False     # Prevents OOM on 24GB
    remove_unused_columns: bool = False
    optim: str = "adamw_8bit"              # 8-bit optimizer
    
    # Monitoring and Logging
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 3
    report_to: str = "wandb"
    
    # Reproducibility
    seed: int = 3407
    data_seed: int = 3407
    
    # Power Management (RTX 3090: 90% power for max performance)
    cuda_power_limit: Optional[int] = 315   # 90% of 350W (RTX 3090 TDP)
    
    def get_bnb_config(self) -> BitsAndBytesConfig:
        """Get BitsAndBytes configuration for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
        )
    
    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration for parameter-efficient fine-tuning."""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=self.modules_to_save,
        )
    
    def get_training_args(self, output_dir: str = "./results") -> TrainingArguments:
        """Get training arguments optimized for RTX 4090."""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            tf32=self.tf32,
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
            data_seed=self.data_seed,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_pin_memory=self.dataloader_pin_memory,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
            run_name=f"qlora-{self.model_name.split('/')[-1]}",
        )
    
    def setup_power_management(self) -> None:
        """Setup GPU power management for efficiency."""
        if self.cuda_power_limit:
            import os
            os.environ["CUDA_POWER_LIMIT"] = str(self.cuda_power_limit)
            print(f"🔋 GPU power limit set to {self.cuda_power_limit}W (90% for max performance)")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration summary."""
        return {
            "model_name": self.model_name,
            "quantization": "4-bit NF4 + Double Quantization",
            "lora_rank": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "batch_size_effective": self.per_device_train_batch_size * self.gradient_accumulation_steps,
            "precision": "bfloat16" if self.bf16 else ("float16" if self.fp16 else "float32"),
            "optimizer": self.optim,
            "power_limit": f"{self.cuda_power_limit}W" if self.cuda_power_limit else "Default",
        }


@dataclass 
class DatasetConfig:
    """Dataset configuration for fine-tuning."""
    
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    text_column: str = "text"
    max_seq_length: int = 512
    dataset_text_field: str = "text"
    packing: bool = False  # Set to True for better GPU utilization
    
    # Data preprocessing
    num_proc: int = 4  # Number of processes for data preprocessing
    streaming: bool = False
    
    def validate(self) -> None:
        """Validate dataset configuration."""
        if not self.dataset_name and not self.dataset_path:
            raise ValueError("Either dataset_name or dataset_path must be provided")


# Predefined configurations for common scenarios
GEMMA_3N_E2B_CONFIG = QLoRAConfig(
    model_name="google/gemma-3n-e2b",  # 2B effective params (5B total)
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)

GEMMA_3N_E4B_LOCAL_CONFIG = QLoRAConfig(
    model_name="/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee",  # 4B effective params (8B total) - local path
    max_steps=800,
    per_device_train_batch_size=2,  # Reduced for larger model
    gradient_accumulation_steps=8,   # Maintain effective batch size
    lora_r=16,                      # Reduce rank for memory
)

# Gemma 3N E4B configuration for Agent Loop
# Using local path as the model is experimental and not on HuggingFace Hub
GEMMA_3N_CONFIG = QLoRAConfig(
    model_name="/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee",
    max_steps=1000,
    per_device_train_batch_size=2,  # Modèle optimisé
    gradient_accumulation_steps=8,   # Effective batch size = 16
    target_modules="all-linear",     # Architecture E4B optimisée
)