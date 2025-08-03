#!/usr/bin/env python3
"""
Download datasets for Hierarchical Reasoning Model (HRM) training.
These datasets teach the model to break down complex tasks into hierarchical steps.
"""

import os
from datasets import load_dataset
import json
from pathlib import Path

# Ensure we use the correct HF cache
os.environ['HF_HOME'] = '/media/jerem/641C8D6C1C8D3A56/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/media/jerem/641C8D6C1C8D3A56/hf_cache/datasets'

def download_hierarchical_datasets():
    """Download datasets that teach hierarchical thinking and task decomposition."""
    
    datasets_to_download = [
        {
            "name": "gsm8k",
            "path": "gsm8k", 
            "config": "main",
            "description": "Grade school math problems requiring step-by-step reasoning"
        },
        {
            "name": "TheoremQA",
            "path": "wenhu/TheoremQA",
            "config": None,
            "description": "Theorem proving with hierarchical reasoning steps"
        },
        {
            "name": "TaskSource",
            "path": "sileod/tasksource-instruct",
            "config": None,
            "description": "Multi-task instructions with hierarchical task decomposition"
        },
        {
            "name": "Self-Instruct",
            "path": "yizhongw/self_instruct", 
            "config": None,
            "description": "Instructions that demonstrate task planning and decomposition"
        }
    ]
    
    print("🧠 Downloading Hierarchical Reasoning Model (HRM) datasets...")
    print(f"📁 Cache location: {os.environ['HF_DATASETS_CACHE']}")
    print("-" * 80)
    
    downloaded = []
    
    for dataset_info in datasets_to_download:
        try:
            print(f"\n📥 Downloading: {dataset_info['name']}")
            print(f"   Description: {dataset_info['description']}")
            
            # Download dataset
            if dataset_info['config']:
                dataset = load_dataset(dataset_info['path'], dataset_info['config'], split='train')
            else:
                dataset = load_dataset(dataset_info['path'], split='train')
            
            # Show sample to understand structure
            print(f"   ✅ Downloaded {len(dataset)} examples")
            print(f"   📊 Features: {list(dataset.features.keys())}")
            
            # Save first example for inspection
            sample_dir = Path("models/datasets/hrm_samples")
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            with open(sample_dir / f"{dataset_info['name']}_sample.json", 'w') as f:
                json.dump(dataset[0], f, indent=2, ensure_ascii=False)
            
            downloaded.append(dataset_info['name'])
            
        except Exception as e:
            print(f"   ❌ Error downloading {dataset_info['name']}: {e}")
    
    print("\n" + "=" * 80)
    print(f"✅ Successfully downloaded {len(downloaded)} HRM datasets:")
    for name in downloaded:
        print(f"   - {name}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Inspect samples in: models/datasets/hrm_samples/")
    print(f"   2. Create unified format for hierarchical reasoning")
    print(f"   3. Mix with existing agent_instruct dataset")
    
    return downloaded

def create_hrm_training_examples():
    """Create examples that teach hierarchical task decomposition."""
    
    hrm_examples = [
        {
            "instruction": "Find all Python files modified in the last week and create a summary report",
            "hierarchical_steps": [
                "1. PLAN: Break down the task",
                "   1.1. Find Python files modified in last week",
                "   1.2. Extract relevant information from each file",
                "   1.3. Create a summary report",
                "2. EXECUTE: Find Python files",
                "   2.1. Use find command with time filter",
                "   2.2. Filter for .py extension",
                "   ```bash\n   find . -name '*.py' -mtime -7 -type f\n   ```",
                "3. EXECUTE: Process each file",
                "   3.1. Get file stats",
                "   3.2. Extract first few lines or docstring",
                "   ```bash\n   for file in $(find . -name '*.py' -mtime -7); do\n       echo \"File: $file\"\n       stat -c '%y %s' \"$file\"\n       head -n 5 \"$file\" | grep -E '(def|class|import)'\n   done\n   ```",
                "4. EXECUTE: Create report",
                "   4.1. Aggregate information",
                "   4.2. Format as markdown",
                "   ```bash\n   echo '# Python Files Modified This Week' > report.md\n   # ... aggregate data ...\n   ```"
            ]
        },
        {
            "instruction": "Set up a monitoring system for disk usage and alert when above 80%",
            "hierarchical_steps": [
                "1. ANALYZE: Understand requirements",
                "   1.1. Monitor disk usage",
                "   1.2. Set threshold at 80%",
                "   1.3. Create alert mechanism",
                "2. DESIGN: Plan implementation",
                "   2.1. Create monitoring script",
                "   2.2. Set up cron job",
                "   2.3. Configure alert method",
                "3. IMPLEMENT: Monitoring script",
                "   ```python\n   import shutil\n   import smtplib\n   \n   def check_disk_usage():\n       usage = shutil.disk_usage('/')\n       percent = (usage.used / usage.total) * 100\n       if percent > 80:\n           send_alert(percent)\n   ```",
                "4. IMPLEMENT: Schedule execution",
                "   ```bash\n   # Add to crontab\n   */5 * * * * /usr/bin/python3 /path/to/disk_monitor.py\n   ```"
            ]
        }
    ]
    
    # Save HRM examples
    output_path = Path("models/datasets/hrm_examples.json")
    with open(output_path, 'w') as f:
        json.dump(hrm_examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Created {len(hrm_examples)} HRM training examples")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    # Download HRM datasets
    downloaded = download_hierarchical_datasets()
    
    # Create custom HRM examples
    create_hrm_training_examples()
    
    print("\n🎯 HRM dataset preparation complete!")
    print("   The model will learn to think hierarchically and decompose complex tasks")