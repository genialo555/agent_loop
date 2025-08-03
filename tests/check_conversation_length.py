#!/usr/bin/env python3
"""Check conversation lengths in agent_instruct dataset"""

from datasets import load_dataset
import json
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee",
    token="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"
)

# Load a sample of the dataset
print("Loading dataset sample...")
dataset = load_dataset(
    "parquet", 
    data_files="/media/jerem/jeux&travail/datasets/agent_instruct/data/code_-00000-of-00002.parquet",
    split="train[:100]"  # Sample 100 examples
)

# Calculate token lengths
token_lengths = []

for example in dataset:
    # Parse messages
    messages = json.loads(example['messages'])
    
    # Convert to text
    text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content and not (role == "system" and not content.strip()):
            text += f"### {role.title()}:\n{content}\n\n"
    
    # Tokenize and count
    tokens = tokenizer(text, truncation=False)
    token_lengths.append(len(tokens['input_ids']))

# Statistics
print(f"\nToken length statistics for {len(token_lengths)} conversations:")
print(f"Min: {min(token_lengths)}")
print(f"Max: {max(token_lengths)}")
print(f"Mean: {np.mean(token_lengths):.0f}")
print(f"Median: {np.median(token_lengths):.0f}")
print(f"95th percentile: {np.percentile(token_lengths, 95):.0f}")
print(f"99th percentile: {np.percentile(token_lengths, 99):.0f}")

# Distribution
print("\nDistribution:")
for threshold in [512, 1024, 2048, 4096, 8192]:
    count = sum(1 for l in token_lengths if l <= threshold)
    percentage = (count / len(token_lengths)) * 100
    print(f"<= {threshold} tokens: {percentage:.1f}%")

# Recommendation
p95 = int(np.percentile(token_lengths, 95))
recommended = min(4096, 2 ** (p95.bit_length()))  # Next power of 2, max 4096
print(f"\nRecommended max_length: {recommended}")
print(f"This covers 95% of examples without excessive padding")