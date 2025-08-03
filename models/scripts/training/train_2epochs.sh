#!/bin/bash
source .env
source venv/bin/activate
python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --num-epochs 2 \
  --output-dir ./results/gemma-3n-unsloth-2epochs