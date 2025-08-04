"""
Configuration for Hierarchical Reasoning Model (HRM) with Gemma-3N.

This module defines all configuration parameters for HRM training and inference,
including hierarchical module settings, training hyperparameters, and optimization configs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class HRMModuleConfig:
    """Configuration for individual HRM modules (Low-level and High-level)."""
    
    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=list)
    
    # Module dimensions
    hidden_dim: int = 2048
    intermediate_dim: int = 4096
    
    # Convergence parameters
    max_iterations: int = 10
    convergence_threshold: float = 1e-3
    
    # Normalization
    use_layer_norm: bool = True
    norm_eps: float = 1e-6


@dataclass
class HRMConfig:
    """Main configuration for HRM-Gemma-3N model."""
    
    # Model paths
    base_model_path: str = "/media/jerem/641C8D6C1C8D3A56/MLLMODELS/gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
    output_dir: str = "./models/results/gemma-3n-hrm-2epochs"
    
    # HRM architecture parameters
    num_high_cycles: int = 4  # N in paper
    timesteps_per_cycle: int = 8  # T in paper
    
    # Module configurations
    low_level_config: HRMModuleConfig = field(default_factory=lambda: HRMModuleConfig(
        lora_rank=32,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj", "k_proj"],
        hidden_dim=2048,
        intermediate_dim=3072,
        max_iterations=16,
    ))
    
    high_level_config: HRMModuleConfig = field(default_factory=lambda: HRMModuleConfig(
        lora_rank=64,
        lora_alpha=32,
        lora_target_modules=["gate_proj", "up_proj", "down_proj"],
        hidden_dim=4096,
        intermediate_dim=8192,
        max_iterations=8,
    ))
    
    # Deep supervision parameters
    deep_supervision_segments: int = 3  # M in paper
    supervision_loss_weight: float = 1.0
    
    # Adaptive Computation Time (ACT)
    enable_act: bool = True
    act_max_segments: int = 8
    act_epsilon: float = 0.1  # Exploration rate
    q_learning_rate: float = 0.001
    q_discount_factor: float = 0.99
    
    # Training configuration - 2 EPOCHS
    batch_size: int = 1  # RTX 3090 constraint
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1  # Will be calculated from epochs
    num_train_epochs: int = 2  # 2 EPOCHS as requested
    
    # Optimization
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    
    # Gradient approximation
    use_approximate_gradient: bool = True
    gradient_checkpointing: bool = True
    
    # Memory optimization
    max_seq_length: int = 2048
    use_flash_attention_2: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 5  # Keep more checkpoints for 2 epochs
    
    # Hardware
    device_map: str = "auto"
    torch_dtype: str = "float16"
    
    # Datasets
    dataset_name: str = "gsm8k"  # Can be gsm8k, code_alpaca, etc.
    dataset_cache_dir: str = "/media/jerem/641C8D6C1C8D3A56/hf_cache"
    
    # Integration with existing QLoRA config
    use_existing_qlora: bool = True
    
    # Evaluation
    eval_batch_size: int = 4
    predict_with_generate: bool = True
    generation_max_length: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        config_dict = {
            "base_model_path": self.base_model_path,
            "output_dir": self.output_dir,
            "num_high_cycles": self.num_high_cycles,
            "timesteps_per_cycle": self.timesteps_per_cycle,
            "low_level_config": {
                "lora_rank": self.low_level_config.lora_rank,
                "lora_alpha": self.low_level_config.lora_alpha,
                "lora_dropout": self.low_level_config.lora_dropout,
                "lora_target_modules": self.low_level_config.lora_target_modules,
                "hidden_dim": self.low_level_config.hidden_dim,
                "intermediate_dim": self.low_level_config.intermediate_dim,
                "max_iterations": self.low_level_config.max_iterations,
                "convergence_threshold": self.low_level_config.convergence_threshold,
            },
            "high_level_config": {
                "lora_rank": self.high_level_config.lora_rank,
                "lora_alpha": self.high_level_config.lora_alpha,
                "lora_dropout": self.high_level_config.lora_dropout,
                "lora_target_modules": self.high_level_config.lora_target_modules,
                "hidden_dim": self.high_level_config.hidden_dim,
                "intermediate_dim": self.high_level_config.intermediate_dim,
                "max_iterations": self.high_level_config.max_iterations,
                "convergence_threshold": self.high_level_config.convergence_threshold,
            },
            "deep_supervision_segments": self.deep_supervision_segments,
            "enable_act": self.enable_act,
            "act_max_segments": self.act_max_segments,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "use_approximate_gradient": self.use_approximate_gradient,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_seq_length": self.max_seq_length,
            "use_flash_attention_2": self.use_flash_attention_2,
            "load_in_4bit": self.load_in_4bit,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HRMConfig":
        """Create config from dictionary."""
        # Extract nested configs
        low_level_dict = config_dict.pop("low_level_config", {})
        high_level_dict = config_dict.pop("high_level_config", {})
        
        # Create module configs
        low_level_config = HRMModuleConfig(**low_level_dict)
        high_level_config = HRMModuleConfig(**high_level_dict)
        
        # Create main config
        return cls(
            low_level_config=low_level_config,
            high_level_config=high_level_config,
            **config_dict
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check paths
        if not Path(self.base_model_path).exists():
            raise ValueError(f"Base model path does not exist: {self.base_model_path}")
        
        # Check HRM parameters
        if self.num_high_cycles < 1:
            raise ValueError("num_high_cycles must be >= 1")
        
        if self.timesteps_per_cycle < 1:
            raise ValueError("timesteps_per_cycle must be >= 1")
        
        # Check deep supervision
        if self.deep_supervision_segments < 1:
            raise ValueError("deep_supervision_segments must be >= 1")
        
        # Check batch size for RTX 3090
        if self.batch_size > 2:
            print(f"Warning: batch_size={self.batch_size} might cause OOM on RTX 3890")
        
        # Validate LoRA ranks
        if self.low_level_config.lora_rank > 64:
            print("Warning: Low-level LoRA rank > 64 might impact performance")
        
        if self.high_level_config.lora_rank > 128:
            print("Warning: High-level LoRA rank > 128 might impact performance")


# Preset configurations for different use cases
def get_config_gsm8k_2epochs() -> HRMConfig:
    """Configuration optimized for GSM8K mathematical reasoning - 2 epochs."""
    config = HRMConfig()
    config.dataset_name = "gsm8k"
    config.num_high_cycles = 4
    config.timesteps_per_cycle = 8
    config.generation_max_length = 512
    config.num_train_epochs = 2
    config.output_dir = "./models/results/gemma-3n-hrm-gsm8k-2epochs"
    return config


def get_config_code_generation_2epochs() -> HRMConfig:
    """Configuration optimized for code generation tasks - 2 epochs."""
    config = HRMConfig()
    config.dataset_name = "code_alpaca"
    config.num_high_cycles = 6  # More cycles for complex code
    config.timesteps_per_cycle = 12
    config.generation_max_length = 1024
    config.max_seq_length = 4096
    config.num_train_epochs = 2
    config.output_dir = "./models/results/gemma-3n-hrm-code-2epochs"
    return config


def get_config_linux_agent_2epochs() -> HRMConfig:
    """Configuration for Linux navigation agent - 2 epochs."""
    config = HRMConfig()
    config.dataset_name = "agent_instruct"
    config.num_high_cycles = 3
    config.timesteps_per_cycle = 6
    config.deep_supervision_segments = 2
    config.num_train_epochs = 2
    config.output_dir = "./models/results/gemma-3n-hrm-agent-2epochs"
    return config


def get_config_full_2epochs() -> HRMConfig:
    """Full configuration for production training - 2 epochs."""
    config = HRMConfig()
    config.num_high_cycles = 4
    config.timesteps_per_cycle = 8
    config.deep_supervision_segments = 3
    config.num_train_epochs = 2
    config.batch_size = 1
    config.gradient_accumulation_steps = 8
    config.learning_rate = 2e-4
    config.warmup_steps = 200  # More warmup for 2 epochs
    config.save_steps = 1000  # Save less frequently for long training
    config.logging_steps = 50
    return config


def get_config_debug() -> HRMConfig:
    """Lightweight configuration for debugging."""
    config = HRMConfig()
    config.num_high_cycles = 2
    config.timesteps_per_cycle = 4
    config.deep_supervision_segments = 1
    config.max_steps = 100
    config.num_train_epochs = -1  # Use max_steps for debug
    config.logging_steps = 1
    config.save_steps = 50
    return config