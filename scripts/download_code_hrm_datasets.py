#!/usr/bin/env python3
"""
Download code-specific HRM datasets for hierarchical reasoning in programming.
"""

import os
from datasets import load_dataset
import json
from pathlib import Path

os.environ['HF_HOME'] = '/media/jerem/641C8D6C1C8D3A56/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/media/jerem/641C8D6C1C8D3A56/hf_cache/datasets'

def download_code_hrm_datasets():
    """Download datasets focused on code generation with hierarchical thinking."""
    
    datasets_to_download = [
        {
            "name": "wizardcoder",
            "path": "WizardLM/WizardCoder-Python-V1.0",
            "config": None,
            "description": "Code generation with step-by-step explanations"
        },
        {
            "name": "code-alpaca",  
            "path": "sahil2801/CodeAlpaca-20k",
            "config": None,
            "description": "Instructional code generation dataset"
        },
        {
            "name": "python-code-instructions",
            "path": "iamtarun/python_code_instructions_18k_alpaca",
            "config": None,
            "description": "Python instructions with hierarchical solutions"
        },
        {
            "name": "shell-commands",
            "path": "b-mc2/sql-create-context",  # Alternative since pure shell dataset hard to find
            "config": None,
            "description": "Structured query decomposition (similar to shell commands)"
        }
    ]
    
    print("🧠 Downloading Code-focused HRM datasets...")
    print(f"📁 Cache location: {os.environ['HF_DATASETS_CACHE']}")
    print("-" * 80)
    
    downloaded = []
    
    for dataset_info in datasets_to_download:
        try:
            print(f"\n📥 Attempting: {dataset_info['name']}")
            print(f"   Description: {dataset_info['description']}")
            
            dataset = load_dataset(dataset_info['path'], split='train[:100]')  # Get first 100 for testing
            
            print(f"   ✅ Downloaded {len(dataset)} examples") 
            print(f"   📊 Features: {list(dataset.features.keys())}")
            
            # Save sample
            sample_dir = Path("models/datasets/hrm_samples")
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            with open(sample_dir / f"{dataset_info['name']}_sample.json", 'w') as f:
                json.dump(dataset[0], f, indent=2, ensure_ascii=False)
            
            downloaded.append(dataset_info['name'])
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return downloaded

def create_linux_agent_hrm_examples():
    """Create HRM examples specifically for Linux system administration."""
    
    examples = [
        {
            "instruction": "Clean up old log files that are taking too much space",
            "hrm_breakdown": {
                "1_analyze": "Understand the problem: disk space consumed by logs",
                "2_plan": [
                    "Find large log files",
                    "Identify old logs safe to delete", 
                    "Create cleanup strategy",
                    "Implement with safety checks"
                ],
                "3_execute": {
                    "step1": "find /var/log -type f -size +100M -mtime +30",
                    "step2": "du -sh /var/log/* | sort -hr | head -20", 
                    "step3": "find /var/log -name '*.log.[0-9]*' -mtime +30 -delete",
                    "step4": "systemctl restart rsyslog"
                },
                "4_verify": "df -h /var/log && du -sh /var/log"
            }
        },
        {
            "instruction": "Create a backup system for important configuration files",
            "hrm_breakdown": {
                "1_analyze": "Need automated config backup with versioning",
                "2_plan": [
                    "Identify critical config files",
                    "Design backup structure",
                    "Create backup script",
                    "Schedule automation"
                ],
                "3_execute": {
                    "step1": "mkdir -p /backup/configs/{etc,home,opt}",
                    "step2": """cat > /usr/local/bin/backup_configs.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/configs"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup /etc
tar -czf "$BACKUP_DIR/etc/etc_$DATE.tar.gz" /etc/

# Backup user configs
for user in $(ls /home); do
    tar -czf "$BACKUP_DIR/home/${user}_$DATE.tar.gz" /home/$user/.bashrc /home/$user/.ssh/ 2>/dev/null
done

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF""",
                    "step3": "chmod +x /usr/local/bin/backup_configs.sh",
                    "step4": "echo '0 2 * * * root /usr/local/bin/backup_configs.sh' >> /etc/crontab"
                },
                "4_verify": "ls -la /backup/configs/ && crontab -l | grep backup"
            }
        }
    ]
    
    output_path = Path("models/datasets/linux_agent_hrm.json")
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Created {len(examples)} Linux Agent HRM examples")
    print(f"   Saved to: {output_path}")
    
    return examples

if __name__ == "__main__":
    # Try to download code datasets
    print("🔍 Searching for code-focused HRM datasets...\n")
    downloaded = download_code_hrm_datasets()
    
    # Create Linux-specific examples
    linux_examples = create_linux_agent_hrm_examples()
    
    print("\n" + "="*80)
    print("📊 Summary:")
    print(f"   Downloaded {len(downloaded)} datasets")
    print(f"   Created {len(linux_examples)} Linux HRM examples")
    print("\n💡 HRM Training Strategy:")
    print("   1. Use GSM8K for basic step-by-step reasoning")
    print("   2. Mix with Linux HRM examples for system tasks") 
    print("   3. Combine with agent_instruct for comprehensive training")