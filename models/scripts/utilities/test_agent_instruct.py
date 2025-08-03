#!/usr/bin/env python3
"""
Test loading agent_instruct dataset specifically.
This dataset is critical for agent training.
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.qlora_config import DatasetConfig


def test_agent_instruct_loading():
    """Test loading agent_instruct dataset from local path."""
    print("Testing agent_instruct dataset loading...")
    print("="*80)
    
    # Path to agent_instruct data
    data_path = "/media/jerem/jeux&travail/datasets/agent_instruct/data"
    
    # List all parquet files
    parquet_files = list(Path(data_path).glob("*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in sorted(parquet_files):
        print(f"  - {f.name}")
    
    # Load the dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset(
            "parquet", 
            data_files=[str(f) for f in parquet_files],
            split="train"
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Examine first example
        if len(dataset) > 0:
            first_example = dataset[0]
            print(f"\nFirst example structure:")
            for key, value in first_example.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"  - {key}: {preview}")
                elif isinstance(value, list):
                    print(f"  - {key}: List with {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"    First item keys: {list(value[0].keys())}")
                else:
                    print(f"  - {key}: {type(value).__name__}")
        
        # Check for conversation format
        if 'conversations' in dataset.features:
            print("\n[INFO] Dataset uses 'conversations' format")
            print("This is typical for instruction-following datasets")
            
            # Show conversation structure
            conv = dataset[0]['conversations']
            if isinstance(conv, list) and len(conv) > 0:
                print(f"\nConversation structure (first turn):")
                print(f"  {conv[0]}")
        
        # Memory usage estimate
        dataset_size = dataset.data.nbytes if hasattr(dataset.data, 'nbytes') else 0
        print(f"\nEstimated memory usage: {dataset_size / 1024**3:.2f} GB")
        
        # Test different loading approaches
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR TRAINING:")
        print("="*80)
        
        print("\n1. **Direct parquet loading (RECOMMENDED):**")
        print("   python training/qlora_finetune.py \\")
        print("     --data /media/jerem/jeux&travail/datasets/agent_instruct/data \\")
        print("     --text-column conversations \\")
        print("     --model-config gemma-2b \\")
        print("     --output-dir ./results/agent_instruct")
        
        print("\n2. **HuggingFace loading (for CI/CD):**")
        print("   python training/qlora_finetune.py \\")
        print("     --data agent-instruct/agent-instruct \\")
        print("     --model-config gemma-2b \\")
        print("     --output-dir ./results/agent_instruct")
        
        print("\n3. **Memory optimization tips:**")
        print("   - Use streaming=True for very large datasets")
        print("   - Enable gradient_checkpointing in config")
        print("   - Reduce max_seq_length if possible (default: 512)")
        print("   - Use smaller batch sizes with gradient accumulation")
        
        return True
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format():
    """Test the data format and suggest preprocessing."""
    print("\n\n" + "="*80)
    print("DATA FORMAT ANALYSIS")
    print("="*80)
    
    data_path = "/media/jerem/jeux&travail/datasets/agent_instruct/data"
    
    # Load a single file to examine format
    sample_file = list(Path(data_path).glob("*.parquet"))[0]
    print(f"\nAnalyzing file: {sample_file.name}")
    
    df = pd.read_parquet(sample_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'conversations' in df.columns:
        print("\nConversation format detected!")
        first_conv = df.iloc[0]['conversations']
        print(f"Example conversation length: {len(first_conv)} turns")
        
        print("\nPreprocessing recommendation:")
        print("- Convert conversations to single text string")
        print("- Use special tokens to separate turns")
        print("- Example format: '<human>question</human><assistant>answer</assistant>'")


if __name__ == "__main__":
    print("Agent Instruct Dataset Test")
    print("="*80)
    
    # Test loading
    success = test_agent_instruct_loading()
    
    if success:
        # Analyze format
        test_data_format()
    
    print("\n[DONE] Test completed")