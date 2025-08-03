#!/usr/bin/env python3
"""
QLoRA Fine-tuning Pipeline - Sprint 2 Implementation
Modern 4-bit quantization fine-tuning with SFTTrainer and advanced optimizations.

Type-safe QLoRA fine-tuning pipeline with enhanced error handling and logging.
"""

import argparse
import json
import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from __future__ import annotations

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from .qlora_config import (
    QLoRAConfig, 
    DatasetConfig,
    GEMMA_2B_CONFIG,
    GEMMA_9B_CONFIG,
    GEMMA_3_2B_CONFIG,
    GEMMA_3N_CONFIG
)

# Setup logging with new log path structure
log_dir = Path("/home/jerem/agent_loop/models/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"qlora_training_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning Pipeline")
    
    # Model configuration
    parser.add_argument("--model-config", type=str, default="gemma-3n",
                       choices=["gemma-2b", "gemma-9b", "gemma-3-2b", "gemma-3n", "custom"],
                       help="Predefined model configuration")
    parser.add_argument("--model-name", type=str, help="Custom model name (if config=custom)")
    
    # Data configuration  
    parser.add_argument("--data", type=str, required=True,
                       help="Dataset name (HF) or local path")
    parser.add_argument("--data-split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--text-column", type=str, default="text",
                       help="Column name containing text data")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for model and logs")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size")
    
    # Resume training
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Experiment tracking
    parser.add_argument("--wandb-project", type=str, default="qlora-finetuning",
                       help="Weights & Biases project name")
    parser.add_argument("--run-name", type=str, help="Experiment run name")
    
    # Advanced options
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run without actual training")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser


def get_model_config(config_name: str, custom_model: Optional[str] = None) -> QLoRAConfig:
    """Get predefined model configuration."""
    configs = {
        "gemma-2b": GEMMA_2B_CONFIG,
        "gemma-9b": GEMMA_9B_CONFIG, 
        "gemma-3-2b": GEMMA_3_2B_CONFIG,
        "gemma-3n": GEMMA_3N_CONFIG,
    }
    
    if config_name == "custom":
        if not custom_model:
            raise ValueError("Custom model name required when using custom config")
        config = QLoRAConfig(model_name=custom_model)
    else:
        config = configs[config_name]
    
    return config


def setup_wandb(config: QLoRAConfig, project: str, run_name: Optional[str] = None) -> None:
    """Initialize Weights & Biases tracking."""
    wandb.init(
        project=project,
        name=run_name or f"qlora-{config.model_name.split('/')[-1]}",
        config=config.get_model_info(),
        tags=["qlora", "4bit", "gemma"],
    )


def load_and_prepare_dataset(
    dataset_config: DatasetConfig,
    tokenizer: AutoTokenizer,
) -> Dataset:
    """Load and prepare dataset for training."""
    logger.info(f"Loading dataset: {dataset_config.dataset_name or dataset_config.dataset_path}")
    
    if dataset_config.dataset_name:
        # Load from HuggingFace
        dataset = load_dataset(dataset_config.dataset_name, split="train")
    else:
        # Load from local path - handle different file types
        path = Path(dataset_config.dataset_path)
        
        if path.is_dir():
            # Directory - look for parquet files first, then json
            parquet_files = list(path.glob("*.parquet"))
            json_files = list(path.glob("*.json"))
            jsonl_files = list(path.glob("*.jsonl"))
            
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} parquet files")
                dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
            elif json_files:
                logger.info(f"Found {len(json_files)} json files")
                # For training, prioritize files with 'train' in name
                train_files = [f for f in json_files if 'train' in f.name.lower()]
                files_to_load = train_files if train_files else json_files[:1]
                dataset = load_dataset("json", data_files=[str(f) for f in files_to_load], split="train")
            elif jsonl_files:
                logger.info(f"Found {len(jsonl_files)} jsonl files")
                dataset = load_dataset("json", data_files=[str(f) for f in jsonl_files], split="train")
            else:
                raise ValueError(f"No supported data files found in {path}")
        else:
            # Single file
            if path.suffix == '.parquet':
                dataset = load_dataset("parquet", data_files=str(path), split="train")
            elif path.suffix in ['.json', '.jsonl']:
                dataset = load_dataset("json", data_files=str(path), split="train")
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Basic preprocessing
    def preprocess_function(examples):
        # Convert messages to text format for training
        import json
        texts = []
        
        for msg_str in examples.get(dataset_config.text_column, examples.get("messages", [])):
            try:
                # Parse JSON messages
                if isinstance(msg_str, str):
                    messages = json.loads(msg_str)
                else:
                    messages = msg_str
                
                # Convert to text format
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        text += f"### {role.title()}:\n{content}\n\n"
                
                texts.append(text.strip() if text else "")
            except:
                texts.append("")
        
        return {"text": texts}
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=dataset_config.num_proc,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    logger.info(f"Dataset features: {list(dataset.features.keys())}")
    return dataset


def setup_model_and_tokenizer(config: QLoRAConfig) -> Tuple[Any, Any]:
    """Setup quantized model and tokenizer."""
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Add special tokens if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=config.get_bnb_config(),
        device_map="auto",
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, config.get_lora_config())
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    config: QLoRAConfig,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the model using SFTTrainer."""
    logger.info("Starting QLoRA fine-tuning...")
    
    # Training arguments with SFT config
    base_training_args = config.get_training_args(output_dir)
    
    # Create SFTConfig - in TRL 0.20+, max_seq_length is now max_length
    sft_config = SFTConfig(
        **base_training_args.to_dict(),
        max_length=5000,  # Covers 95%+ of agent_instruct conversations
        packing=False,
    )
    
    # Data collator - using default for conversation data
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Formatting function for SFTTrainer - handles multiple data formats
    def formatting_prompts_func(examples):
        import json
        outputs = []
        
        # Detect which column contains the data
        data_column = None
        if 'messages' in examples:
            data_column = 'messages'
        elif 'text' in examples:
            data_column = 'text'
        elif 'conversations' in examples:
            data_column = 'conversations'
        elif 'prompt' in examples and 'completion' in examples:
            # Handle prompt/completion format
            for i in range(len(examples['prompt'])):
                prompt = examples['prompt'][i] if isinstance(examples['prompt'][i], str) else str(examples['prompt'][i])
                completion = examples['completion'][i] if isinstance(examples['completion'][i], str) else str(examples['completion'][i])
                outputs.append(f"### Instruction:\n{prompt}\n\n### Response:\n{completion}")
            return outputs
        else:
            # Try to find any column that might contain text data
            for key in examples.keys():
                if any(keyword in key.lower() for keyword in ['text', 'content', 'message', 'conv', 'dial']):
                    data_column = key
                    break
            if not data_column:
                raise ValueError(f"Could not find text data column. Available columns: {list(examples.keys())}")
        
        # Process the data based on its format
        for i in range(len(examples[data_column])):
            try:
                data = examples[data_column][i]
                
                # Case 1: Already formatted string
                if isinstance(data, str):
                    # Check if it's JSON string
                    if data.strip().startswith('[') or data.strip().startswith('{'):
                        try:
                            parsed_data = json.loads(data)
                            if isinstance(parsed_data, list):
                                # List of messages/conversations
                                formatted_text = ""
                                for msg in parsed_data:
                                    if isinstance(msg, dict):
                                        role = msg.get('role', msg.get('from', 'user'))
                                        content = msg.get('content', msg.get('value', msg.get('text', '')))
                                        if content:
                                            formatted_text += f"### {role.title()}:\n{content}\n\n"
                                    else:
                                        formatted_text += str(msg) + "\n\n"
                                outputs.append(formatted_text.strip())
                            elif isinstance(parsed_data, dict):
                                # Single message/conversation
                                if 'instruction' in parsed_data and 'response' in parsed_data:
                                    outputs.append(f"### Instruction:\n{parsed_data['instruction']}\n\n### Response:\n{parsed_data['response']}")
                                elif 'prompt' in parsed_data and 'completion' in parsed_data:
                                    outputs.append(f"### Instruction:\n{parsed_data['prompt']}\n\n### Response:\n{parsed_data['completion']}")
                                else:
                                    # Generic dict formatting
                                    formatted_text = ""
                                    for k, v in parsed_data.items():
                                        if v:
                                            formatted_text += f"### {k.title()}:\n{v}\n\n"
                                    outputs.append(formatted_text.strip())
                            else:
                                outputs.append(str(parsed_data))
                        except json.JSONDecodeError:
                            # Not JSON, use as-is
                            outputs.append(data)
                    else:
                        # Plain text string
                        outputs.append(data)
                
                # Case 2: List of messages/conversations
                elif isinstance(data, list):
                    formatted_text = ""
                    for item in data:
                        if isinstance(item, dict):
                            # Dict with role/content or from/value structure
                            role = item.get('role', item.get('from', item.get('speaker', 'user')))
                            content = item.get('content', item.get('value', item.get('text', item.get('message', ''))))
                            if content:
                                formatted_text += f"### {role.title()}:\n{content}\n\n"
                        elif isinstance(item, str):
                            # List of strings - treat as alternating user/assistant
                            if len(formatted_text) == 0 or formatted_text.count('### Assistant:') < formatted_text.count('### User:'):
                                formatted_text += f"### User:\n{item}\n\n"
                            else:
                                formatted_text += f"### Assistant:\n{item}\n\n"
                        else:
                            formatted_text += str(item) + "\n\n"
                    outputs.append(formatted_text.strip())
                
                # Case 3: Dict format
                elif isinstance(data, dict):
                    if 'instruction' in data and 'response' in data:
                        outputs.append(f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['response']}")
                    elif 'prompt' in data and 'completion' in data:
                        outputs.append(f"### Instruction:\n{data['prompt']}\n\n### Response:\n{data['completion']}")
                    elif 'input' in data and 'output' in data:
                        outputs.append(f"### Input:\n{data['input']}\n\n### Output:\n{data['output']}")
                    else:
                        # Generic dict formatting
                        formatted_text = ""
                        for k, v in data.items():
                            if v and isinstance(v, (str, int, float)):
                                formatted_text += f"### {k.title()}:\n{v}\n\n"
                        outputs.append(formatted_text.strip() if formatted_text else str(data))
                
                # Case 4: Other types
                else:
                    outputs.append(str(data))
                    
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}. Using fallback conversion.")
                outputs.append(str(examples[data_column][i]))
        
        # Ensure all outputs are non-empty strings
        outputs = [o if o else "Empty content" for o in outputs]
        
        return outputs
    
    # Setup SFTTrainer - with formatting function that returns strings
    def formatting_func_for_sft(example):
        """Format a single example for SFTTrainer - must return a string."""
        return example["text"] if example["text"] else "Empty content"
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=formatting_func_for_sft,
        processing_class=tokenizer,
    )
    
    # Train the model
    start_time = time.time()
    
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    
    training_time = time.time() - start_time
    
    # Save the final model
    trainer.save_model()
    trainer.save_state()
    
    # Get training metrics
    train_results = trainer.state.log_history
    final_loss = train_results[-1].get("train_loss", 0.0) if train_results else 0.0
    
    training_info = {
        "model_name": config.model_name,
        "training_time": round(training_time, 2),
        "final_loss": final_loss,
        "total_steps": trainer.state.global_step,
        "config": config.get_model_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save training info to models/logs directory
    log_file_path = log_dir / f"training_info_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file_path, "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Also save to output directory for backward compatibility
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training info saved to: {log_file_path}")
    
    logger.info(f"Training completed in {training_time:.2f}s")
    logger.info(f"Final loss: {final_loss:.4f}")
    
    return training_info


def convert_to_gguf(model_path: str, output_path: str) -> None:
    """Convert fine-tuned model to GGUF format (placeholder for now)."""
    logger.info("GGUF conversion is planned for future implementation")
    logger.info("For now, use tools like Unsloth or llama.cpp convert scripts")
    
    # Future implementation:
    # from unsloth import FastLanguageModel
    # model.save_pretrained_gguf(output_path, tokenizer)


def main() -> None:
    """Main training pipeline with enhanced type safety and error handling."""
    args = build_parser().parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get configuration
    config = get_model_config(args.model_config, args.model_name)
    
    # Override config with CLI arguments
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    
    # Dataset configuration
    dataset_config = DatasetConfig(
        dataset_name=args.data if not os.path.exists(args.data) else None,
        dataset_path=args.data if os.path.exists(args.data) else None,
        text_column=args.text_column,
        max_seq_length=args.max_seq_length,
    )
    
    dataset_config.validate()
    
    # Setup power management
    config.setup_power_management()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize W&B
    if not args.no_wandb and not args.dry_run:
        setup_wandb(config, args.wandb_project, args.run_name)
    
    logger.info("=" * 50)
    logger.info("QLoRA Fine-tuning Pipeline - Sprint 2")
    logger.info("=" * 50)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Configuration: {json.dumps(config.get_model_info(), indent=2)}")
    
    if args.dry_run:
        logger.info("DRY RUN - Exiting without training")
        return
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Load dataset
        dataset = load_and_prepare_dataset(dataset_config, tokenizer)
        
        # Train model
        training_info = train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume,
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Logs available in: {log_dir}")
        
        # Log to W&B and ensure proper cleanup
        if not args.no_wandb:
            wandb.log({"final_training_info": training_info})
            wandb.finish()
            logger.info("W&B logging completed")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
