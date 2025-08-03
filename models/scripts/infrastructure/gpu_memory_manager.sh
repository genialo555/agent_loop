#!/bin/bash

# GPU Memory Manager for Ollama and ML Training
# Manages switching between Ollama inference and ML training workloads

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check GPU memory usage
check_gpu_memory() {
    print_status "Current GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
    echo ""
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
}

# Function to stop Ollama service
stop_ollama() {
    print_status "Stopping Ollama service..."
    
    # First, try to stop gracefully via systemctl if it's a service
    if systemctl is-active --quiet ollama; then
        sudo systemctl stop ollama
        print_status "Stopped Ollama systemd service"
    fi
    
    # Kill specific Ollama processes
    if pgrep -f "ollama serve" > /dev/null; then
        print_warning "Killing ollama serve processes..."
        pkill -f "ollama serve" || true
        sleep 2
    fi
    
    # Kill ollama runner processes (these hold GPU memory)
    if pgrep -f "ollama_llama_server" > /dev/null; then
        print_warning "Killing ollama runner processes..."
        pkill -f "ollama_llama_server" || true
        sleep 2
    fi
    
    # Force kill if processes still exist
    if pgrep -f "ollama" > /dev/null; then
        print_warning "Force killing remaining Ollama processes..."
        pkill -9 -f "ollama" || true
        sleep 1
    fi
    
    print_status "Ollama stopped successfully"
}

# Function to unload all Ollama models from memory
unload_ollama_models() {
    print_status "Unloading all Ollama models from memory..."
    
    # Get list of loaded models
    loaded_models=$(ollama list 2>/dev/null | grep -E "GB|MB" | awk '{print $1}' || true)
    
    if [ -n "$loaded_models" ]; then
        for model in $loaded_models; do
            print_status "Unloading model: $model"
            # Run a dummy command to unload the model
            timeout 5 ollama run $model "exit" 2>/dev/null || true
        done
    fi
}

# Function to start Ollama service
start_ollama() {
    print_status "Starting Ollama service..."
    
    # Check if Ollama should be started as a systemd service
    if systemctl list-unit-files | grep -q "ollama.service"; then
        sudo systemctl start ollama
        print_status "Started Ollama systemd service"
    else
        # Start Ollama in background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        print_status "Started Ollama serve (PID: $!)"
    fi
    
    sleep 3
    print_status "Ollama is ready"
}

# Function to optimize Ollama for low memory usage
optimize_ollama_memory() {
    print_status "Setting Ollama memory optimization..."
    
    # Set Ollama to use minimal GPU memory
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_NUM_GPU=1
    export OLLAMA_GPU_OVERHEAD=0
    
    # Optional: Force CPU-only mode for Ollama
    # export OLLAMA_NUM_GPU=0
    
    print_status "Ollama memory settings applied"
}

# Function to prepare for training
prepare_for_training() {
    print_status "Preparing GPU for ML training..."
    
    # Stop Ollama
    stop_ollama
    
    # Clear GPU memory cache
    print_status "Clearing GPU memory cache..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Wait for memory to be freed
    sleep 2
    
    # Show final GPU status
    check_gpu_memory
    
    print_status "GPU is ready for training!"
}

# Function to prepare for inference
prepare_for_inference() {
    print_status "Preparing GPU for inference..."
    
    # Clear any training-related GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Apply memory optimizations
    optimize_ollama_memory
    
    # Start Ollama
    start_ollama
    
    # Show GPU status
    check_gpu_memory
    
    print_status "GPU is ready for inference!"
}

# Main script logic
case "${1:-}" in
    "stop")
        stop_ollama
        check_gpu_memory
        ;;
    "start")
        start_ollama
        check_gpu_memory
        ;;
    "restart")
        stop_ollama
        sleep 2
        start_ollama
        check_gpu_memory
        ;;
    "status")
        check_gpu_memory
        ;;
    "training")
        prepare_for_training
        ;;
    "inference")
        prepare_for_inference
        ;;
    "unload")
        unload_ollama_models
        check_gpu_memory
        ;;
    *)
        echo "GPU Memory Manager for Ollama and ML Training"
        echo ""
        echo "Usage: $0 {stop|start|restart|status|training|inference|unload}"
        echo ""
        echo "Commands:"
        echo "  stop       - Stop Ollama service and free GPU memory"
        echo "  start      - Start Ollama service"
        echo "  restart    - Restart Ollama service"
        echo "  status     - Show current GPU memory usage"
        echo "  training   - Prepare GPU for ML training (stops Ollama)"
        echo "  inference  - Prepare GPU for inference (starts Ollama)"
        echo "  unload     - Unload all models from GPU memory"
        echo ""
        exit 1
        ;;
esac