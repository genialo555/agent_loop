#!/usr/bin/env python3
"""Test formatting for agent_instruct dataset"""

from datasets import load_dataset
import json

# Load a small sample
print("Loading dataset sample...")
dataset = load_dataset(
    "parquet", 
    data_files="/media/jerem/jeux&travail/datasets/agent_instruct/data/analytical_reasoning-00000-of-00001.parquet",
    split="train[:5]"
)

print(f"Columns: {dataset.column_names}")
print(f"Number of examples: {len(dataset)}")

# Test the preprocessing
def preprocess_function(examples):
    """Convert messages to text format for training."""
    texts = []
    
    for msg_str in examples.get("messages", []):
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
                    # Skip system messages if empty
                    if role == "system" and not content.strip():
                        continue
                    text += f"### {role.title()}:\n{content}\n\n"
            
            texts.append(text.strip() if text else "")
        except Exception as e:
            print(f"Error processing message: {e}")
            texts.append("")
    
    return {"text": texts}

# Apply preprocessing
print("\nApplying preprocessing...")
processed = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)

print(f"\nProcessed columns: {processed.column_names}")
print(f"First example type: {type(processed[0]['text'])}")
print(f"First example (truncated):\n{processed[0]['text'][:500]}...")

# Test the formatting function that would be used by SFTTrainer
def formatting_prompts_func(examples):
    """Format examples for SFTTrainer"""
    outputs = []
    
    for text in examples['text']:
        if isinstance(text, str):
            outputs.append(text)
        else:
            print(f"WARNING: Found non-string type: {type(text)}")
            outputs.append(str(text))
    
    return outputs

# Test formatting
print("\n\nTesting formatting function...")
batch = processed[:2]
formatted = formatting_prompts_func(batch)
print(f"Formatted type: {type(formatted)}")
print(f"First formatted item type: {type(formatted[0])}")
print(f"Length: {len(formatted)}")

print("\n✅ If all types are <class 'str'>, the formatting should work!")