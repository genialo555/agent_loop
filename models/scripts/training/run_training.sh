#!/bin/bash

# ============================================================================
# MLOps Fine-tuning Training Orchestrator - Sprint 2
# ============================================================================
# Production-grade training pipeline with comprehensive error handling,
# monitoring, and artifact management for autonomous fine-tuning workflows
# ============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# Configuration & Environment Setup
# ============================================================================

# Script metadata
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Environment variables with defaults
readonly ENVIRONMENT="${ENVIRONMENT:-development}"
readonly TRAINING_CONFIG="${TRAINING_CONFIG:-base}"
readonly GPU_ENABLED="${GPU_ENABLED:-false}"
readonly DISTRIBUTED_TRAINING="${DISTRIBUTED_TRAINING:-false}"
readonly MAX_RETRIES="${MAX_RETRIES:-3}"
readonly TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

# Directories
readonly MODELS_DIR="${PROJECT_ROOT}/models"
readonly FINETUNED_DIR="${MODELS_DIR}/finetuned"
readonly CHECKPOINTS_DIR="${PROJECT_ROOT}/model_checkpoints"
readonly LOGS_DIR="${PROJECT_ROOT}/logs/training"
readonly DATA_DIR="${PROJECT_ROOT}/datasets"
readonly ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/training"

# Training configurations
declare -A TRAINING_CONFIGS=(
    ["base"]="--base gemma_base.gguf --head xnet --lambda_hint 0.3"
    ["lora-optimized"]="--base gemma_base.gguf --head xnet --lambda_hint 0.5 --step-hint"
    ["production"]="--base gemma_base.gguf --head xnet --lambda_hint 0.4 --step-hint"
    ["experimental"]="--base gemma_base.gguf --head xnet --lambda_hint 0.2"
)

# ============================================================================
# Logging & Monitoring Functions
# ============================================================================

setup_logging() {
    mkdir -p "$LOGS_DIR"
    readonly LOG_FILE="${LOGS_DIR}/training_${TRAINING_CONFIG}_${TIMESTAMP}.log"
    readonly METRICS_FILE="${LOGS_DIR}/metrics_${TRAINING_CONFIG}_${TIMESTAMP}.json"
    
    # Initialize structured logging
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
    
    log_info "Training orchestrator started"
    log_info "Environment: $ENVIRONMENT"
    log_info "Configuration: $TRAINING_CONFIG"
    log_info "Log file: $LOG_FILE"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local metric_timestamp="${3:-$(date -Iseconds)}"
    
    # Append metric to JSON Lines format
    echo "{\"timestamp\":\"$metric_timestamp\",\"metric\":\"$metric_name\",\"value\":$metric_value,\"config\":\"$TRAINING_CONFIG\"}" >> "$METRICS_FILE"
}

# ============================================================================
# Environment Validation
# ============================================================================

validate_environment() {
    log_info "Validating training environment..."
    
    # Check Python version
    if ! python3.13 --version >/dev/null 2>&1; then
        log_error "Python 3.13 is required but not available"
        return 1
    fi
    
    # Check CUDA availability if GPU enabled
    if [[ "$GPU_ENABLED" == "true" ]]; then
        if ! python3.13 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_error "GPU training requested but CUDA not available"
            return 1
        fi
        log_info "CUDA detected: $(python3.13 -c "import torch; print(f'{torch.cuda.device_count()} GPUs')")"
    fi
    
    # Check disk space (minimum 10GB)
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        log_error "Insufficient disk space: $(($available_space / 1024 / 1024))GB available, 10GB required"
        return 1
    fi
    
    # Validate training configuration
    if [[ ! "${TRAINING_CONFIGS[$TRAINING_CONFIG]+exists}" ]]; then
        log_error "Invalid training configuration: $TRAINING_CONFIG"
        log_info "Available configurations: ${!TRAINING_CONFIGS[*]}"
        return 1
    fi
    
    log_info "Environment validation passed"
}

# ============================================================================
# Resource Management
# ============================================================================

setup_directories() {
    log_info "Setting up directory structure..."
    
    local dirs=(
        "$MODELS_DIR"
        "$FINETUNED_DIR"
        "$CHECKPOINTS_DIR"
        "$LOGS_DIR"
        "$ARTIFACTS_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
}

cleanup_old_artifacts() {
    log_info "Cleaning up old training artifacts..."
    
    # Remove checkpoints older than 7 days
    find "$CHECKPOINTS_DIR" -name "*.json" -mtime +7 -delete 2>/dev/null || true
    
    # Remove logs older than 30 days
    find "$LOGS_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Clean up temporary training files
    find "$PROJECT_ROOT" -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# ============================================================================
# Data Preparation & Validation
# ============================================================================

prepare_training_data() {
    log_info "Preparing training data..."
    
    local data_path="${DATA_DIR}/processed/train_splits/${TRAINING_CONFIG}"
    
    # Validate data exists
    if [[ ! -d "$data_path" ]]; then
        log_warn "Training data not found at $data_path, attempting download..."
        
        if [[ -f "${PROJECT_ROOT}/scripts/secure_dataset_downloader.py" ]]; then
            python3.13 "${PROJECT_ROOT}/scripts/secure_dataset_downloader.py" \
                --config "$TRAINING_CONFIG" \
                --output-dir "$DATA_DIR" \
                --validate-checksums
        else
            log_error "Dataset downloader not available and no training data found"
            return 1
        fi
    fi
    
    # Validate data integrity
    local sample_count
    sample_count=$(find "$data_path" -name "*.json" | wc -l)
    
    if [[ $sample_count -lt 100 ]]; then
        log_error "Insufficient training samples: $sample_count found, minimum 100 required"
        return 1
    fi
    
    log_info "Training data validated: $sample_count samples found"
    log_metric "training_samples" "$sample_count"
    
    echo "$data_path"
}

# ============================================================================
# Model Management & Versioning
# ============================================================================

generate_model_version() {
    local git_hash
    git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    echo "${TRAINING_CONFIG}_${git_hash}_${TIMESTAMP}"
}

create_model_metadata() {
    local model_version="$1"
    local data_path="$2"
    local start_time="$3"
    
    local metadata_file="${FINETUNED_DIR}/${model_version}/metadata.json"
    mkdir -p "$(dirname "$metadata_file")"
    
    cat > "$metadata_file" << EOF
{
  "model_version": "$model_version",
  "training_config": "$TRAINING_CONFIG",
  "environment": "$ENVIRONMENT",
  "start_time": "$start_time",
  "data_path": "$data_path",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "training_args": "${TRAINING_CONFIGS[$TRAINING_CONFIG]}",
  "gpu_enabled": $GPU_ENABLED,
  "distributed_training": $DISTRIBUTED_TRAINING,
  "python_version": "$(python3.13 --version)",
  "system_info": {
    "hostname": "$(hostname)",
    "os": "$(uname -s)",
    "architecture": "$(uname -m)",
    "available_memory": "$(free -h | awk '/^Mem:/ {print $2}')"
  },
  "status": "training"
}
EOF
    
    log_info "Model metadata created: $metadata_file"
}

update_model_metadata() {
    local model_version="$1"
    local status="$2"
    local end_time="$3"
    local final_loss="${4:-null}"
    
    local metadata_file="${FINETUNED_DIR}/${model_version}/metadata.json"
    
    if [[ -f "$metadata_file" ]]; then
        # Update using jq for safe JSON manipulation
        local updated_metadata
        updated_metadata=$(jq \
            --arg status "$status" \
            --arg end_time "$end_time" \
            --arg final_loss "$final_loss" \
            '.status = $status | .end_time = $end_time | .final_loss = ($final_loss | if . == "null" then null else tonumber end)' \
            "$metadata_file")
        
        echo "$updated_metadata" > "$metadata_file"
        log_info "Model metadata updated: status=$status"
    fi
}

# ============================================================================
# Training Pipeline
# ============================================================================

run_training_pipeline() {
    local data_path="$1"
    local model_version="$2"
    local start_time="$3"
    
    log_info "Starting training pipeline..."
    log_info "Model version: $model_version"
    log_info "Data path: $data_path"
    
    # Prepare training command
    local training_args="${TRAINING_CONFIGS[$TRAINING_CONFIG]}"
    local output_dir="${FINETUNED_DIR}/${model_version}"
    
    local training_cmd=(
        "python3.13" "${PROJECT_ROOT}/training/qlora_finetune.py"
        --data "$data_path"
        --output-dir "$output_dir"
        --model-version "$model_version"
        --log-file "$LOG_FILE"
        --metrics-file "$METRICS_FILE"
    )
    
    # Add configuration-specific arguments
    IFS=' ' read -ra args <<< "$training_args"
    training_cmd+=("${args[@]}")
    
    # Add GPU/distributed training flags
    if [[ "$GPU_ENABLED" == "true" ]]; then
        training_cmd+=("--gpu")
    fi
    
    if [[ "$DISTRIBUTED_TRAINING" == "true" ]]; then
        training_cmd+=("--distributed")
    fi
    
    log_info "Training command: ${training_cmd[*]}"
    log_metric "training_started" 1
    
    # Execute training with timeout
    local training_exit_code=0
    if timeout "$TIMEOUT_SECONDS" "${training_cmd[@]}"; then
        log_info "Training completed successfully"
        log_metric "training_completed" 1
        update_model_metadata "$model_version" "completed" "$(date -Iseconds)"
    else
        training_exit_code=$?
        log_error "Training failed with exit code: $training_exit_code"
        log_metric "training_failed" 1
        update_model_metadata "$model_version" "failed" "$(date -Iseconds)"
        return $training_exit_code
    fi
}

# ============================================================================
# Post-Training Validation
# ============================================================================

run_model_validation() {
    local model_version="$1"
    local model_path="${FINETUNED_DIR}/${model_version}"
    
    log_info "Running model validation..."
    
    # Check if model files exist
    if [[ ! -d "$model_path" ]]; then
        log_error "Model directory not found: $model_path"
        return 1
    fi
    
    # Run evaluation if evaluation script exists
    if [[ -f "${PROJECT_ROOT}/training/evaluate_toolbench.py" ]]; then
        local eval_output="${model_path}/evaluation.json"
        
        if python3.13 "${PROJECT_ROOT}/training/evaluate_toolbench.py" \
            --model "$model_path" \
            --output "$eval_output"; then
            log_info "Model evaluation completed: $eval_output"
            
            # Extract key metrics
            if command -v jq >/dev/null && [[ -f "$eval_output" ]]; then
                local accuracy
                accuracy=$(jq -r '.accuracy // "unknown"' "$eval_output")
                log_info "Model accuracy: $accuracy"
                log_metric "model_accuracy" "$accuracy"
            fi
        else
            log_warn "Model evaluation failed but continuing..."
        fi
    fi
    
    # Basic sanity checks
    local model_size
    model_size=$(du -sh "$model_path" | cut -f1)
    log_info "Model size: $model_size"
    
    log_info "Model validation completed"
}

# ============================================================================
# Artifact Management
# ============================================================================

package_training_artifacts() {
    local model_version="$1"
    local model_path="${FINETUNED_DIR}/${model_version}"
    
    log_info "Packaging training artifacts..."
    
    local artifacts_package="${ARTIFACTS_DIR}/training_${model_version}.tar.gz"
    mkdir -p "$(dirname "$artifacts_package")"
    
    # Create compressed archive of training artifacts
    tar -czf "$artifacts_package" \
        -C "$FINETUNED_DIR" "${model_version}" \
        -C "$LOGS_DIR" "training_${TRAINING_CONFIG}_${TIMESTAMP}.log" \
        -C "$LOGS_DIR" "metrics_${TRAINING_CONFIG}_${TIMESTAMP}.json" \
        2>/dev/null || log_warn "Some artifacts may be missing from package"
    
    log_info "Artifacts packaged: $artifacts_package"
    log_info "Package size: $(du -sh "$artifacts_package" | cut -f1)"
    
    # Generate artifact manifest
    local manifest_file="${ARTIFACTS_DIR}/manifest_${model_version}.json"
    cat > "$manifest_file" << EOF
{
  "model_version": "$model_version",
  "package_path": "$artifacts_package",
  "created_at": "$(date -Iseconds)",
  "package_size": "$(stat -c%s "$artifacts_package")",
  "package_checksum": "$(sha256sum "$artifacts_package" | cut -d' ' -f1)",
  "contents": [
    "model_files",
    "training_logs",
    "metrics",
    "metadata"
  ]
}
EOF
    
    log_info "Artifact manifest created: $manifest_file"
}

# ============================================================================
# Error Handling & Recovery
# ============================================================================

cleanup_on_failure() {
    local model_version="$1"
    
    log_warn "Cleaning up failed training artifacts..."
    
    # Remove incomplete model directory
    if [[ -d "${FINETUNED_DIR}/${model_version}" ]]; then
        rm -rf "${FINETUNED_DIR}/${model_version}"
        log_info "Removed incomplete model directory"
    fi
    
    # Mark training as failed in metrics
    log_metric "training_cleanup" 1
}

handle_interrupt() {
    log_warn "Training interrupted by signal"
    if [[ -n "${CURRENT_MODEL_VERSION:-}" ]]; then
        cleanup_on_failure "$CURRENT_MODEL_VERSION"
    fi
    exit 130
}

# ============================================================================
# Main Training Orchestration
# ============================================================================

main() {
    local start_time
    start_time=$(date -Iseconds)
    
    # Set up signal handlers
    trap handle_interrupt INT TERM
    
    # Initialize environment
    setup_logging
    validate_environment
    setup_directories
    cleanup_old_artifacts
    
    # Prepare training
    local data_path
    data_path=$(prepare_training_data)
    
    local model_version
    model_version=$(generate_model_version)
    
    export CURRENT_MODEL_VERSION="$model_version"
    
    # Create model metadata
    create_model_metadata "$model_version" "$data_path" "$start_time"
    
    # Execute training pipeline with retry logic
    local attempt=1
    local training_success=false
    
    while [[ $attempt -le $MAX_RETRIES ]] && [[ "$training_success" == false ]]; do
        log_info "Training attempt $attempt/$MAX_RETRIES"
        
        if run_training_pipeline "$data_path" "$model_version" "$start_time"; then
            training_success=true
            log_info "Training succeeded on attempt $attempt"
        else
            log_warn "Training attempt $attempt failed"
            ((attempt++))
            
            if [[ $attempt -le $MAX_RETRIES ]]; then
                local retry_delay=$((attempt * 30))
                log_info "Retrying in $retry_delay seconds..."
                sleep $retry_delay
            fi
        fi
    done
    
    if [[ "$training_success" == false ]]; then
        log_error "Training failed after $MAX_RETRIES attempts"
        cleanup_on_failure "$model_version"
        exit 1
    fi
    
    # Post-training tasks
    run_model_validation "$model_version"
    package_training_artifacts "$model_version"
    
    # Final logging
    local end_time
    end_time=$(date -Iseconds)
    local duration=$(($(date -d"$end_time" +%s) - $(date -d"$start_time" +%s)))
    
    log_info "Training orchestration completed successfully"
    log_info "Model version: $model_version"
    log_info "Duration: ${duration}s"
    log_info "Logs: $LOG_FILE"
    log_info "Artifacts: ${ARTIFACTS_DIR}/training_${model_version}.tar.gz"
    
    log_metric "training_duration" "$duration"
    log_metric "training_success" 1
    
    echo "SUCCESS: Model $model_version trained and validated"
}

# ============================================================================
# CLI Interface & Help
# ============================================================================

show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

MLOps Fine-tuning Training Orchestrator - Sprint 2

OPTIONS:
    -c, --config CONFIG     Training configuration (default: base)
                           Available: ${!TRAINING_CONFIGS[*]}
    -e, --environment ENV   Environment (development|staging|production)
    -g, --gpu              Enable GPU training
    -d, --distributed      Enable distributed training
    -r, --retries NUM      Maximum retry attempts (default: 3)
    -t, --timeout SEC      Training timeout in seconds (default: 3600)
    -h, --help             Show this help message

EXAMPLES:
    $SCRIPT_NAME --config lora-optimized --gpu
    $SCRIPT_NAME --environment production --distributed --retries 5
    
ENVIRONMENT VARIABLES:
    TRAINING_CONFIG        Training configuration override
    ENVIRONMENT           Environment override
    GPU_ENABLED           Enable GPU training (true/false)
    DISTRIBUTED_TRAINING  Enable distributed training (true/false)
    MAX_RETRIES           Maximum retry attempts
    TIMEOUT_SECONDS       Training timeout in seconds

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ENABLED="true"
            shift
            ;;
        -d|--distributed)
            DISTRIBUTED_TRAINING="true"
            shift
            ;;
        -r|--retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required tools
for tool in python3.13 jq tar; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_error "Required tool not found: $tool"
        exit 1
    fi
done

# Execute main function
main "$@"