"""
HRM Training script with Unsloth optimization.

This is the main training script that combines all HRM components
with Unsloth for efficient training on Gemma-3N.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Unsloth imports
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from trl import SFTTrainer

# HRM imports
from .hrm_config import (
    HRMConfig, 
    get_config_gsm8k_2epochs,
    get_config_code_generation_2epochs,
    get_config_linux_agent_2epochs,
    get_config_full_2epochs
)
from .hrm_model import HRMGemma3N
from .hrm_modules import create_hrm_modules
from .deep_supervision import (
    DeepSupervisionTrainer,
    create_deep_supervision_trainer
)
from .adaptive_compute import create_act_model
from .approximate_gradient import HRMGradientApproximation

# Import existing QLoRA infrastructure
try:
    from ..qlora.qlora_config import QLoRAConfig
    from ..qlora.qlora_utils import get_model_and_tokenizer
except ImportError:
    logger.warning("Could not import existing QLoRA infrastructure")

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HRMUnslothTrainer:
    """
    Main trainer class that integrates HRM with Unsloth.
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate config
        config.validate()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/checkpoints").mkdir(exist_ok=True)
        Path(f"{self.config.output_dir}/logs").mkdir(exist_ok=True)
        
    def load_base_model(self):
        """Load Gemma-3N base model with Unsloth."""
        logger.info("Loading Gemma-3N base model with Unsloth...")
        
        # Use full HRM-Unsloth integration
        from .hrm_unsloth_full_integration import create_hrm_unsloth_model
        
        logger.info("Creating HRM-Unsloth integrated model...")
        model = create_hrm_unsloth_model(self.config)
        tokenizer = model.tokenizer
        
        # Set tokenizer properties
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info(f"Model loaded successfully on {self.device}")
        
    def load_dataset(self) -> Dataset:
        """Load and prepare dataset for HRM training."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load from HuggingFace datasets
        if self.config.dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", cache_dir=self.config.dataset_cache_dir)
            train_dataset = dataset["train"]
            
            # Format for HRM (add hierarchical structure)
            def format_gsm8k(example):
                # Parse question and answer
                question = example["question"]
                answer = example["answer"]
                
                # Create hierarchical prompt
                prompt = f"""Solve this step by step:

Question: {question}

High-level plan:
1. Understand the problem
2. Identify key information
3. Determine approach
4. Calculate step by step
5. Verify answer

Solution:
{answer}"""
                
                return {"text": prompt}
            
            train_dataset = train_dataset.map(format_gsm8k)
            
        elif self.config.dataset_name == "code_alpaca":
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", cache_dir=self.config.dataset_cache_dir)
            train_dataset = dataset["train"]
            
            # Format for code generation
            def format_code(example):
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")
                
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                else:
                    prompt = f"Instruction: {instruction}\nOutput: {output}"
                
                return {"text": prompt}
            
            train_dataset = train_dataset.map(format_code)
            
        elif self.config.dataset_name == "agent_instruct":
            # Load from local path
            # Use correct path for agent_instruct dataset
            dataset_path = Path("/media/jerem/jeux&travail/datasets/agent_instruct")
            if dataset_path.exists():
                train_dataset = load_dataset(
                    "parquet",
                    data_files=str(dataset_path / "data/*.parquet"),
                    cache_dir=self.config.dataset_cache_dir
                )["train"]
            else:
                raise ValueError(f"Agent instruct dataset not found at {dataset_path}")
                
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
        
        logger.info(f"Dataset loaded: {len(train_dataset)} examples")
        return train_dataset
    
    def create_trainer(self, train_dataset: Dataset):
        """Create Unsloth trainer with HRM configuration."""
        logger.info("Creating HRM-enhanced Unsloth trainer...")
        
        # Training arguments
        training_args = UnslothTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.torch_dtype == "float16",
            bf16=self.config.torch_dtype == "bfloat16",
            optim=self.config.optimizer,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.gradient_clip if hasattr(self.config, 'gradient_clip') else 1.0,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to=["tensorboard"],
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
        # Create SFT trainer with Unsloth optimizations
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=True,  # Unsloth packing for efficiency
            args=training_args,
        )
        
        logger.info("Trainer created successfully")
        
    def train(self):
        """Run HRM training with deep supervision."""
        logger.info("Starting HRM training with deep supervision...")
        
        # Save config
        config_path = f"{self.config.output_dir}/hrm_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Train with Unsloth
        logger.info(f"Training for {self.config.num_train_epochs} epochs...")
        
        # Add HRM monitoring callbacks
        from .hrm_monitoring import create_hrm_monitoring
        
        hrm_callback, metrics_collector = create_hrm_monitoring(
            self.config,
            self.config.output_dir
        )
        
        self.trainer.add_callback(hrm_callback)
        
        # Start training
        self.trainer.train()
        
        logger.info("Training completed!")
        
    def save_model(self):
        """Save trained HRM model."""
        logger.info(f"Saving model to {self.config.output_dir}")
        
        # Save with Unsloth format
        self.model.save_pretrained(f"{self.config.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final_model")
        
        # Save 16bit version for easier use
        self.model.save_pretrained_merged(
            f"{self.config.output_dir}/final_model_16bit",
            self.tokenizer,
            save_method="merged_16bit",
        )
        
        # Save GGUF for Ollama (quantized)
        self.model.save_pretrained_gguf(
            f"{self.config.output_dir}/final_model_gguf",
            self.tokenizer,
            quantization_method="q4_k_m"
        )
        
        # Create conversion script for later use
        conversion_script = f"""#!/bin/bash
# Conversion script for HRM model to Ollama
python /home/jerem/agent_loop/models/scripts/lora/merge_and_convert_lora.py \\
    --lora-path {self.config.output_dir}/final_model \\
    --output-name gemma-3n-hrm-{self.config.dataset_name}
"""
        script_path = f"{self.config.output_dir}/convert_to_ollama.sh"
        with open(script_path, 'w') as f:
            f.write(conversion_script)
        os.chmod(script_path, 0o755)
        logger.info(f"Conversion script saved to: {script_path}")
        
        logger.info("Model saved in multiple formats")
        
    def run(self):
        """Run complete training pipeline."""
        try:
            # Load model
            self.load_base_model()
            
            # Load dataset
            train_dataset = self.load_dataset()
            
            # Create trainer
            self.create_trainer(train_dataset)
            
            # Train
            self.train()
            
            # Save
            self.save_model()
            
            logger.info("HRM training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train HRM with Gemma-3N using Unsloth")
    
    parser.add_argument(
        "--config",
        type=str,
        choices=["gsm8k", "code", "agent", "full", "debug"],
        default="gsm8k",
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    
    args = parser.parse_args()
    
    # Get config
    if args.config == "gsm8k":
        config = get_config_gsm8k_2epochs()
    elif args.config == "code":
        config = get_config_code_generation_2epochs()
    elif args.config == "agent":
        config = get_config_linux_agent_2epochs()
    elif args.config == "full":
        config = get_config_full_2epochs()
    else:
        config = HRMConfig()  # debug
    
    # Override config with command line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    config.num_train_epochs = args.num_epochs
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"{config.output_dir}_{timestamp}"
    
    # Create trainer and run
    trainer = HRMUnslothTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()