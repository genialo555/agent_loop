#!/usr/bin/env python3
"""
QLoRA Fine-tuning with Unsloth for Gemma-3N-E4B
Optimized version using Unsloth framework for stable training.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Import monitoring module
from agent_loop.monitoring import create_training_monitor

# Setup logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning with Unsloth")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, 
                       default="google/gemma-3n-e4b",
                       help="Model name or path")
    
    # Data configuration  
    parser.add_argument("--data", type=str, required=True,
                       help="Dataset path")
    parser.add_argument("--text-column", type=str, default="messages",
                       help="Column name containing text data")
    parser.add_argument("--max-seq-length", type=int, default=5000,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for model and logs")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--num-epochs", type=int, default=None,
                       help="Number of epochs (overrides max-steps if set)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64,
                       help="LoRA alpha")
    
    # Other options
    parser.add_argument("--seed", type=int, default=3407,
                       help="Random seed")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--no-rich", action="store_true",
                       help="Disable rich progress display")
    parser.add_argument("--monitor-theme", type=str, default="cyberpunk",
                       choices=["default", "minimal", "cyberpunk"],
                       help="Progress monitor theme")
    
    return parser


def load_and_prepare_dataset(data_path: str, text_column: str = "messages") -> Any:
    """Load and prepare dataset for training."""
    logger.info(f"Loading dataset: {data_path}")
    
    # Check if it's a HuggingFace dataset name (no slashes or paths)
    if "/" not in data_path and "\\" not in data_path and not Path(data_path).exists():
        # Load from HuggingFace (will use cache automatically)
        logger.info(f"Loading HuggingFace dataset: {data_path}")
        dataset = load_dataset(data_path, "main", split="train")
    else:
        # Local file handling
        path = Path(data_path)
        
        if path.is_dir():
            # Directory - look for parquet files
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} parquet files")
                dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
        else:
            # Single file
            dataset = load_dataset("parquet", data_files=str(path), split="train")
    
    # Preprocessing function
    def preprocess_function(examples):
        """Convert dataset to text format."""
        texts = []
        
        # Check if it's GSM8K format (question/answer)
        if "question" in examples and "answer" in examples:
            for q, a in zip(examples["question"], examples["answer"]):
                # Format for hierarchical reasoning
                text = f"### Question:\n{q}\n\n### Answer:\n{a}"
                texts.append(text)
        else:
            # Original format for agent_instruct
            for msg_str in examples.get(text_column, []):
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
                        if role and content and not (role == "system" and not content.strip()):
                            text += f"### {role.title()}:\n{content}\n\n"
                    
                    texts.append(text.strip() if text else "")
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")
                    texts.append("")
        
        return {"text": texts}
    
    # Apply preprocessing - USE MORE CPU CORES!
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,  # Ryzen 9 power!
        remove_columns=dataset.column_names,
    )
    
    # Sort by length to minimize recompilations (Gemma-3N fix)
    logger.info("Sorting dataset by length to minimize recompilations...")
    
    def add_length(example):
        return {"length": len(example["text"])}
    
    dataset = dataset.map(add_length, num_proc=16)  # More cores!
    dataset = dataset.sort("length", reverse=True)  # Longest first
    dataset = dataset.remove_columns(["length"])
    
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    return dataset


def setup_model_and_tokenizer(model_path: str, max_seq_length: int):
    """Setup Unsloth model and tokenizer."""
    logger.info(f"Loading model with Unsloth: {model_path}")
    
    # Unsloth will auto-detect dtype
    dtype = None  # None for auto detection
    load_in_4bit = True  # Use 4bit quantization
    
    # Use the model name for Unsloth instead of local path
    # This allows Unsloth to handle the loading properly
    model_name = "google/gemma-3n-e4b"
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Setup LoRA adapters - Unsloth style
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,  # Unsloth supports dropout = 0 only
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's special checkpointing
        random_state=3407,
    )
    
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    dataset,
    args,
) -> Dict[str, Any]:
    """Train the model using SFTTrainer with Unsloth optimizations."""
    logger.info("Starting QLoRA fine-tuning with Unsloth...")
    
    # Training arguments
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": 100,
        "learning_rate": args.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "save_steps": 500,
        "save_total_limit": 2,
        "report_to": "wandb" if args.use_wandb else "none",
        "dataloader_num_workers": 8,  # Use more CPU threads!
        "dataloader_pin_memory": True,  # Faster GPU transfer
        "ddp_find_unused_parameters": False,  # Optimization
    }
    
    # Set epochs or steps
    if args.num_epochs is not None:
        training_args_dict["num_train_epochs"] = args.num_epochs
        logger.info(f"Training for {args.num_epochs} epochs")
    else:
        training_args_dict["max_steps"] = args.max_steps
        logger.info(f"Training for {args.max_steps} steps")
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Formatting function for SFTTrainer
    def formatting_func(example):
        """Format a single example for training."""
        return example["text"] if example["text"] else ""
    
    # Initialize progress monitoring
    progress_monitor = create_training_monitor(
        total_steps=args.max_steps if args.max_steps else (args.num_epochs * len(dataset) // (args.batch_size * args.gradient_accumulation_steps)),
        model_name="Gemma-3N-E4B Unsloth",
        rich_display=not args.no_rich,
        show_gpu=True,
        show_cpu=True,
        theme=args.monitor_theme
    )
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,  # Can enable for better GPU utilization
        dataset_num_proc=16,  # Max parallel processing!
        dataset_batch_size=2000,  # Bigger batches
        callbacks=[progress_monitor],
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save the model
    trainer.save_model()
    
    # Save in GGUF format for Ollama (optional)
    # Disabled for Gemma-3N due to preprocessor_config.json issues
    # if args.output_dir:
    #     logger.info("Saving model in GGUF format...")
    #     try:
    #         model.save_pretrained_gguf(
    #             args.output_dir + "_gguf",
    #             tokenizer,
    #             quantization_method="q4_k_m"
    #         )
    #         logger.info(f"GGUF model saved to: {args.output_dir}_gguf")
    #     except Exception as e:
    #         logger.warning(f"Could not save GGUF format: {e}")
    
    training_info = {
        "model_path": args.model_path,
        "training_time": round(training_time, 2),
        "total_steps": trainer.state.global_step,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save training info
    with open(Path(args.output_dir) / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training completed in {training_time:.2f}s")
    
    return training_info


def main():
    """Main training pipeline."""
    args = build_parser().parse_args()
    
    # Set environment variables for optimal performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Force dataset cache to use the SSD
    os.environ["HF_DATASETS_CACHE"] = "/media/jerem/641C8D6C1C8D3A56/hf_cache/datasets"
    
    # Fix for Gemma-3N recompilation issue
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64  # Increase from default 8
    
    # Enable logging to monitor recompilations
    os.environ['TORCH_LOGS'] = 'recompiles'
    
    logger.info("=" * 50)
    logger.info("QLoRA Fine-tuning with Unsloth")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Output: {args.output_dir}")
    
    try:
        # Load dataset
        dataset = load_and_prepare_dataset(args.data, args.text_column)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(
            args.model_path,
            args.max_seq_length
        )
        
        # Train model
        training_info = train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()