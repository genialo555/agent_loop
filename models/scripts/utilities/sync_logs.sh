#!/usr/bin/env bash
set -euo pipefail

VM_USER=agent
VM_HOST=${VM_HOST:-vm.example}
VM_LOG_DIR=/var/agent/logs
LOCAL_DIR=logs/

rsync -avz "$VM_USER@$VM_HOST:$VM_LOG_DIR/" "$LOCAL_DIR"
