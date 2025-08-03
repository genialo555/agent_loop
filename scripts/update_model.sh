#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-gemma_agent_v0.gguf}
VM_USER=agent
VM_HOST=${VM_HOST:-vm.example}
VM_PATH=/var/agent/models

scp -q "$MODEL_PATH" "$VM_USER@$VM_HOST:$VM_PATH/"
ssh -q "$VM_USER@$VM_HOST" "sudo systemctl restart ollama"
echo "Model $MODEL_PATH deployed to $VM_HOST and Ollama restarted."
