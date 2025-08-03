# Unsloth Training Examples for Gemma 3N

## Complete Training Scripts

### 1. Basic Fine-tuning Script

```python
#!/usr/bin/env python3
"""
Basic Gemma 3N fine-tuning with Unsloth
Optimized for RTX 3090 (24GB VRAM)
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_gemma3n_basic():
    # Model configuration
    max_seq_length = 2048
    dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-3n-e4b-it",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # Enable 4-bit quantization
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # 4x longer contexts
        random_state=3407,
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files="training_data.json", split="train")
    
    # Training arguments optimized for RTX 3090
    training_args = TrainingArguments(
        output_dir="./results/gemma3n-basic",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        group_by_length=True,
        report_to="none",  # Change to "wandb" for W&B logging
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Enable for short sequences
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")

if __name__ == "__main__":
    train_gemma3n_basic()
```

### 2. Advanced Multi-Modal Training

```python
#!/usr/bin/env python3
"""
Advanced Gemma 3N multimodal fine-tuning
Supports text + vision inputs
"""

from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def train_multimodal_gemma3n():
    # Model configuration
    max_seq_length = 2048
    
    # Load multimodal model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="google/gemma-3n-e4b-it",  # Multimodal version
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    
    # Selective fine-tuning configuration
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # Selective component fine-tuning
        finetune_vision_layers=True,      # Fine-tune vision
        finetune_language_layers=True,    # Fine-tune language
        finetune_attention_modules=True,  # Fine-tune attention
        finetune_mlp_modules=True,        # Fine-tune MLPs
        use_gradient_checkpointing="unsloth",
    )
    
    # Custom dataset with vision-language pairs
    def format_multimodal_prompts(examples):
        texts = []
        for i in range(len(examples["image"])):
            # Format: <image>IMAGE_TOKEN</image> Question: {question} Answer: {answer}
            text = f"<image>{examples['image'][i]}</image> "
            text += f"Question: {examples['question'][i]} "
            text += f"Answer: {examples['answer'][i]}"
            texts.append(text)
        return {"text": texts}
    
    # Load multimodal dataset
    dataset = load_dataset("your_multimodal_dataset", split="train")
    dataset = dataset.map(format_multimodal_prompts, batched=True)
    
    # Training arguments for multimodal
    training_args = TrainingArguments(
        output_dir="./results/gemma3n-multimodal",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Higher for multimodal
        num_train_epochs=2,
        learning_rate=1e-4,  # Lower LR for multimodal
        fp16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",  # Enable W&B for multimodal tracking
        run_name="gemma3n-multimodal-finetune",
    )
    
    # Create trainer with vision support
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save multimodal model
    model.save_pretrained("multimodal_model")
    tokenizer.save_pretrained("multimodal_model")

if __name__ == "__main__":
    train_multimodal_gemma3n()
```

### 3. Memory-Optimized Training for Limited VRAM

```python
#!/usr/bin/env python3
"""
Ultra memory-efficient Gemma 3N training
For GPUs with 8-16GB VRAM
"""

import torch
import gc
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

def train_memory_efficient():
    # Aggressive memory management
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reduced configuration
    max_seq_length = 512  # Reduced from 2048
    
    # Load E2B model (smaller) instead of E4B
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-3n-e2b-it",  # 2B model
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Minimal LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Reduced rank
        target_modules=["q_proj", "v_proj"],  # Fewer modules
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Load and preprocess dataset
    dataset = load_dataset("json", data_files="small_dataset.json", split="train")
    
    # Ultra-conservative training arguments
    training_args = TrainingArguments(
        output_dir="./results/gemma3n-lowmem",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # Very high accumulation
        max_steps=1000,  # Limited steps
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",  # Paged optimizer
        save_strategy="steps",
        save_steps=250,
        logging_steps=25,
        warmup_steps=50,
        max_grad_norm=0.3,  # Aggressive gradient clipping
        group_by_length=True,  # Efficient batching
        dataloader_pin_memory=False,  # Save memory
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # Create trainer with memory optimizations
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,  # Single process to save memory
        packing=True,  # Enable packing for efficiency
        args=training_args,
    )
    
    # Custom training loop with memory management
    def custom_train():
        for epoch in range(1):
            trainer.train()
            # Aggressive memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()
    
    custom_train()
    
    # Save with compression
    model.save_pretrained("lowmem_model", safe_serialization=True)
    tokenizer.save_pretrained("lowmem_model")

if __name__ == "__main__":
    train_memory_efficient()
```

### 4. Continued Pre-training Script

```python
#!/usr/bin/env python3
"""
Continued pre-training on domain-specific data
For adapting Gemma 3N to specialized domains
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

def continued_pretraining():
    # Configuration for continued pre-training
    max_seq_length = 4096  # Longer context for pre-training
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-3n-e4b",  # Base model (not instruction-tuned)
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Add LoRA for continued pre-training
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Higher rank for pre-training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Load domain-specific text data
    dataset = load_dataset("text", data_files="domain_corpus.txt", split="train")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    
    # Data collator for MLM-style training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )
    
    # Pre-training arguments
    training_args = TrainingArguments(
        output_dir="./results/gemma3n-continued-pretrain",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-5,  # Lower LR for continued pre-training
        warmup_ratio=0.1,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,  # Enable TF32 for A100/H100
        optim="adamw_torch_fused",  # Fused optimizer
        dataloader_num_workers=4,
        report_to="wandb",
        run_name="gemma3n-continued-pretrain",
    )
    
    # Create trainer for pre-training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field=None,  # Using data_collator instead
        max_seq_length=max_seq_length,
    )
    
    # Train
    trainer.train()
    
    # Save pre-trained model
    model.save_pretrained("pretrained_domain_model")
    tokenizer.save_pretrained("pretrained_domain_model")

if __name__ == "__main__":
    continued_pretraining()
```

### 5. Production-Ready Training with All Features

```python
#!/usr/bin/env python3
"""
Production-ready Gemma 3N training script
Includes: logging, checkpointing, evaluation, and model export
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
import wandb

class ProductionTrainer:
    def __init__(self, config_path="training_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.setup_directories()
        self.setup_logging()
    
    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"gemma3n_{self.config['model_size']}_{timestamp}"
        self.output_dir = Path(f"./results/{self.run_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for reproducibility
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def setup_logging(self):
        """Initialize W&B logging"""
        wandb.init(
            project="gemma3n-finetuning",
            name=self.run_name,
            config=self.config,
            tags=[self.config["model_size"], "production"],
        )
    
    def load_model(self):
        """Load and configure model"""
        model_name = f"google/gemma-3n-{self.config['model_size']}-it"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config["max_seq_length"],
            dtype=getattr(torch, self.config["dtype"]),
            load_in_4bit=self.config["quantization"]["load_in_4bit"],
        )
        
        # Configure LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config["lora"]["rank"],
            target_modules=self.config["lora"]["target_modules"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            bias=self.config["lora"]["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=self.config["seed"],
        )
    
    def prepare_dataset(self):
        """Load and prepare datasets"""
        # Load training data
        train_dataset = load_dataset(
            "json",
            data_files=self.config["data"]["train_file"],
            split="train"
        )
        
        # Load validation data
        eval_dataset = load_dataset(
            "json",
            data_files=self.config["data"]["eval_file"],
            split="train"
        )
        
        # Apply any preprocessing
        if "max_samples" in self.config["data"]:
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), self.config["data"]["max_samples"]))
            )
        
        return train_dataset, eval_dataset
    
    def get_training_args(self):
        """Configure training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            
            # Training hyperparameters
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["eval_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            
            # Optimization
            optim=self.config["training"]["optimizer"],
            weight_decay=self.config["training"]["weight_decay"],
            lr_scheduler_type=self.config["training"]["lr_scheduler"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            
            # Precision and performance
            fp16=self.config["dtype"] == "float16",
            bf16=self.config["dtype"] == "bfloat16",
            tf32=True,
            dataloader_num_workers=self.config["training"]["num_workers"],
            
            # Logging and saving
            logging_steps=self.config["logging"]["steps"],
            save_strategy="steps",
            save_steps=self.config["checkpointing"]["save_steps"],
            save_total_limit=self.config["checkpointing"]["save_total_limit"],
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.config["evaluation"]["eval_steps"],
            metric_for_best_model=self.config["evaluation"]["metric"],
            greater_is_better=False,
            load_best_model_at_end=True,
            
            # W&B integration
            report_to="wandb",
            
            # Advanced features
            group_by_length=True,
            ddp_find_unused_parameters=False,
            include_inputs_for_metrics=True,
        )
    
    def train(self):
        """Main training function"""
        # Load model
        self.load_model()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Configure training arguments
        training_args = self.get_training_args()
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=self.config["data"]["text_field"],
            max_seq_length=self.config["max_seq_length"],
            dataset_num_proc=self.config["data"]["num_proc"],
            packing=self.config["data"]["packing"],
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["training"]["early_stopping_patience"]
                )
            ],
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        
        # Save training results
        with open(self.output_dir / "train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Evaluate on test set
        eval_results = trainer.evaluate()
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        return trainer
    
    def export_model(self, trainer):
        """Export model in various formats"""
        export_dir = self.output_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Save in different formats
        # 1. Full model
        trainer.model.save_pretrained(export_dir / "full_model")
        
        # 2. Merged model (LoRA + base)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(export_dir / "merged_model")
        
        # 3. GGUF format for Ollama
        if self.config.get("export", {}).get("gguf", False):
            # This requires additional conversion tools
            print("GGUF export requires llama.cpp conversion tools")
        
        # Log completion
        wandb.log({"training_completed": True})

def main():
    # Example configuration file structure
    config = {
        "model_size": "e4b",
        "max_seq_length": 2048,
        "dtype": "float16",
        "seed": 42,
        
        "quantization": {
            "load_in_4bit": True,
        },
        
        "lora": {
            "rank": 16,
            "alpha": 16,
            "dropout": 0,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        },
        
        "data": {
            "train_file": "train.json",
            "eval_file": "eval.json",
            "text_field": "text",
            "num_proc": 4,
            "packing": False,
        },
        
        "training": {
            "num_epochs": 3,
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.05,
            "optimizer": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "max_grad_norm": 1.0,
            "num_workers": 4,
            "early_stopping_patience": 3,
        },
        
        "logging": {
            "steps": 10,
        },
        
        "checkpointing": {
            "save_steps": 500,
            "save_total_limit": 3,
        },
        
        "evaluation": {
            "eval_steps": 500,
            "metric": "loss",
        },
        
        "export": {
            "gguf": True,
        },
    }
    
    # Save example config
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run training
    trainer = ProductionTrainer("training_config.json")
    trained_model = trainer.train()
    trainer.export_model(trained_model)

if __name__ == "__main__":
    main()
```

## Dataset Preparation Examples

### Chat Format Dataset
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ]
}
```

### Instruction Format Dataset
```json
{
  "instruction": "Translate the following English text to French:",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

### Text Completion Dataset
```json
{
  "text": "The quick brown fox jumps over the lazy dog."
}
```

## Common Issues and Solutions

### 1. CUDA OOM During Training
```python
# Solution: Reduce memory usage
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Increase this
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_8bit",  # Use paged optimizer
)
```

### 2. Slow Training Speed
```python
# Solution: Enable optimizations
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # Unsloth optimizations
)

training_args = TrainingArguments(
    tf32=True,  # Enable TF32 on Ampere GPUs
    dataloader_num_workers=4,  # Parallel data loading
    group_by_length=True,  # Efficient batching
)
```

### 3. Poor Model Quality
```python
# Solution: Adjust hyperparameters
training_args = TrainingArguments(
    learning_rate=1e-4,  # Lower learning rate
    warmup_ratio=0.1,  # More warmup
    num_train_epochs=3,  # More epochs
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,  # Keep best checkpoint
)
```

## Monitoring and Debugging

### GPU Monitoring During Training
```bash
# In a separate terminal
watch -n 1 nvidia-smi

# Or use nvtop for better visualization
nvtop
```

### Weights & Biases Integration
```python
# Add to training arguments
report_to="wandb"
run_name="gemma3n-experiment-1"

# Log custom metrics
wandb.log({
    "gpu_memory": torch.cuda.memory_allocated() / 1e9,
    "learning_rate": trainer.optimizer.param_groups[0]['lr'],
})
```

### TensorBoard Monitoring
```bash
# Start TensorBoard
tensorboard --logdir ./results

# Access at http://localhost:6006
```

These examples should provide a comprehensive starting point for fine-tuning Gemma 3N with Unsloth in various scenarios.