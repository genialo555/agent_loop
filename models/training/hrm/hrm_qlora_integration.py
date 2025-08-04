"""
Integration layer between HRM and existing QLoRA infrastructure.

This module ensures HRM works seamlessly with the existing training pipeline.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path

# Try to import existing QLoRA components
try:
    from ..qlora.qlora_config import QLoRAConfig
    from ..qlora.qlora_finetune_unsloth import UnslothTrainer
    QLORA_AVAILABLE = True
except ImportError:
    QLORA_AVAILABLE = False
    QLoRAConfig = object  # Fallback

from .hrm_config import HRMConfig
from .hrm_model import HRMGemma3N
from .hrm_modules import create_hrm_modules


class HRMQLoRAConfig(QLoRAConfig if QLORA_AVAILABLE else object):
    """
    Extends existing QLoRA configuration with HRM-specific parameters.
    """
    
    def __init__(self, hrm_config: HRMConfig):
        if QLORA_AVAILABLE:
            # Initialize parent QLoRA config
            super().__init__(
                model_name=hrm_config.base_model_path,
                dataset_name=hrm_config.dataset_name,
                max_seq_length=hrm_config.max_seq_length,
                dtype=hrm_config.torch_dtype,
                load_in_4bit=hrm_config.load_in_4bit,
                per_device_train_batch_size=hrm_config.batch_size,
                gradient_accumulation_steps=hrm_config.gradient_accumulation_steps,
                warmup_steps=hrm_config.warmup_steps,
                num_train_epochs=hrm_config.num_train_epochs,
                learning_rate=hrm_config.learning_rate,
                output_dir=hrm_config.output_dir,
            )
        
        # Add HRM-specific attributes
        self.hrm_config = hrm_config
        self.num_high_cycles = hrm_config.num_high_cycles
        self.timesteps_per_cycle = hrm_config.timesteps_per_cycle
        self.deep_supervision_segments = hrm_config.deep_supervision_segments
        self.use_approximate_gradient = hrm_config.use_approximate_gradient
        self.enable_act = hrm_config.enable_act
        
        # Update LoRA configuration to use HRM settings
        self.lora_r = hrm_config.low_level_config.lora_rank
        self.lora_alpha = hrm_config.low_level_config.lora_alpha
        self.lora_dropout = hrm_config.low_level_config.lora_dropout


class HRMUnslothIntegration:
    """
    Integrates HRM with existing Unsloth training infrastructure.
    """
    
    @staticmethod
    def create_model_with_hrm(base_model, tokenizer, hrm_config: HRMConfig):
        """
        Wraps a base Unsloth model with HRM modules.
        
        This is the key integration point where we add HRM capabilities
        to the existing model.
        """
        # Get model config
        model_config = base_model.config
        
        # Create HRM modules
        hrm_modules = create_hrm_modules(model_config, hrm_config)
        
        # Create wrapper that adds HRM to the model
        class HRMWrapper(torch.nn.Module):
            def __init__(self, base_model, hrm_modules):
                super().__init__()
                self.base_model = base_model
                self.hrm_modules = hrm_modules
                self.config = base_model.config
                
            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                # Get base embeddings
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
                
                # Initialize HRM states
                z_l, z_h = self.hrm_modules['state_init'](inputs_embeds)
                
                # Run HRM cycles (simplified for integration)
                for cycle in range(hrm_config.num_high_cycles):
                    # Low-level processing
                    for t in range(hrm_config.timesteps_per_cycle):
                        z_l_output = self.hrm_modules['low_level'](
                            z_l, z_h, inputs_embeds, iteration=t
                        )
                        z_l = z_l_output.hidden_state
                    
                    # High-level update
                    z_h_output = self.hrm_modules['high_level'](
                        z_h, z_l, cycle=cycle
                    )
                    z_h = z_h_output.hidden_state
                    
                    # Reset low-level
                    z_l = self.hrm_modules['low_level'].reset_with_context(z_h)
                
                # Use HRM output to modulate base model
                # This is a simplified integration - in practice, you'd want
                # to integrate more deeply with the transformer layers
                modulated_embeds = inputs_embeds + self.hrm_modules['high_level'].output_projector(z_h)
                
                # Run through base model with modulated embeddings
                outputs = self.base_model(
                    inputs_embeds=modulated_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )
                
                return outputs
        
        # Return wrapped model
        return HRMWrapper(base_model, hrm_modules)
    
    @staticmethod
    def get_training_callbacks(hrm_config: HRMConfig):
        """
        Returns HRM-specific training callbacks for monitoring.
        """
        from transformers.trainer_callback import TrainerCallback
        
        class HRMMonitoringCallback(TrainerCallback):
            def __init__(self):
                self.convergence_history = []
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    # Log HRM-specific metrics
                    hrm_metrics = {
                        'hrm/cycles': hrm_config.num_high_cycles,
                        'hrm/timesteps': hrm_config.timesteps_per_cycle,
                        'hrm/supervision_segments': hrm_config.deep_supervision_segments,
                    }
                    logs.update(hrm_metrics)
        
        return [HRMMonitoringCallback()]
    
    @staticmethod
    def export_to_gguf(model_path: str, output_name: str):
        """
        Export HRM model to GGUF format for Ollama.
        
        Uses existing conversion infrastructure.
        """
        # Import conversion script
        try:
            from ...scripts.merge_and_convert_lora import merge_and_convert_to_gguf
            
            # Convert
            merge_and_convert_to_gguf(
                lora_path=model_path,
                output_name=output_name,
                quantization="q4_k_m"
            )
        except ImportError:
            print("Warning: GGUF conversion not available")


def validate_integration():
    """
    Validates that HRM is properly integrated with existing infrastructure.
    """
    checks = {
        'qlora_available': QLORA_AVAILABLE,
        'model_path_exists': Path("/media/jerem/641C8D6C1C8D3A56/MLLMODELS/gemma-3n-e4b").exists(),
        'dataset_cache_exists': Path("/media/jerem/641C8D6C1C8D3A56/hf_cache").exists(),
        'unsloth_available': False,
        'monitoring_available': False,
    }
    
    # Check Unsloth
    try:
        from unsloth import FastLanguageModel
        checks['unsloth_available'] = True
    except ImportError:
        pass
    
    # Check monitoring
    monitoring_path = Path("/home/jerem/agent_loop/monitoring")
    checks['monitoring_available'] = monitoring_path.exists()
    
    return checks


if __name__ == "__main__":
    # Run integration checks
    print("HRM-QLoRA Integration Status:")
    checks = validate_integration()
    for check, status in checks.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {check}")
    
    if not all(checks.values()):
        print("\nWarning: Some integration components are missing!")
        print("HRM will work but with reduced functionality.")