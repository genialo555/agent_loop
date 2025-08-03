#!/bin/bash
# Setup script for Gemma Agent Loop VM deployment
# This script coordinates Terraform and Ansible to provision and configure the VM

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for required commands
    for cmd in terraform ansible aws; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed. Please install it first."
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    log_info "All prerequisites met."
}

setup_terraform() {
    log_info "Setting up infrastructure with Terraform..."
    
    cd terraform/
    
    # Initialize Terraform
    terraform init
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f terraform.tfvars ]; then
        log_warn "terraform.tfvars not found. Please create it from terraform.tfvars.example"
        exit 1
    fi
    
    # Plan and apply
    terraform plan -out=tfplan
    terraform apply tfplan
    
    # Export outputs
    AGENT_VM_IP=$(terraform output -raw public_ip)
    export AGENT_VM_IP
    
    cd ..
    log_info "Infrastructure provisioned. VM IP: $AGENT_VM_IP"
}

wait_for_vm() {
    log_info "Waiting for VM to be ready..."
    
    # Wait for SSH to be available
    while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        -i ~/.ssh/${KEY_PAIR_NAME}.pem ubuntu@${AGENT_VM_IP} "echo 'VM ready'" &> /dev/null; do
        log_info "Waiting for SSH..."
        sleep 10
    done
    
    log_info "VM is ready for configuration."
}

run_ansible() {
    log_info "Running Ansible playbooks..."
    
    cd ansible/
    
    # Update inventory with actual VM IP
    sed -i "s/{{ agent_vm_ip }}/${AGENT_VM_IP}/g" inventory.yml
    sed -i "s/{{ key_pair_name }}/${KEY_PAIR_NAME}/g" inventory.yml
    
    # Run the master playbook
    ansible-playbook site.yml
    
    cd ..
    log_info "Ansible configuration complete."
}

post_setup() {
    log_info "Running post-setup tasks..."
    
    # Create local environment file
    cat > .env << EOF
AGENT_VM_IP=${AGENT_VM_IP}
AGENT_VM_SSH_KEY=~/.ssh/${KEY_PAIR_NAME}.pem
OLLAMA_API_URL=http://${AGENT_VM_IP}:11434
FASTAPI_URL=http://${AGENT_VM_IP}:8000
GRAFANA_URL=http://${AGENT_VM_IP}:3000
EOF
    
    log_info "Environment file created: .env"
    
    # Display connection information
    echo ""
    echo "=========================================="
    echo "VM Setup Complete!"
    echo "=========================================="
    echo "SSH: ssh -i ~/.ssh/${KEY_PAIR_NAME}.pem ubuntu@${AGENT_VM_IP}"
    echo "Ollama API: http://${AGENT_VM_IP}:11434"
    echo "FastAPI: http://${AGENT_VM_IP}:8000"
    echo "Grafana: http://${AGENT_VM_IP}:3000 (admin/changeme)"
    echo "Node Exporter: http://${AGENT_VM_IP}:9100/metrics"
    echo "=========================================="
}

# Main execution
main() {
    # Get key pair name from environment or prompt
    if [ -z "${KEY_PAIR_NAME:-}" ]; then
        read -p "Enter your AWS key pair name: " KEY_PAIR_NAME
        export KEY_PAIR_NAME
    fi
    
    check_prerequisites
    setup_terraform
    wait_for_vm
    run_ansible
    post_setup
    
    log_info "Sprint 0 setup complete!"
}

# Run main function
main "$@"