#!/bin/bash
set -e

# Log all output
exec > >(tee -a /var/log/user-data.log)
exec 2>&1

echo "Starting user data script at $(date)"

# Update system
apt-get update
apt-get upgrade -y

# Set hostname
hostnamectl set-hostname ${project_name}-${environment}

# Create agent user for ML workloads
useradd -m -s /bin/bash mlops
usermod -aG sudo,docker mlops

# Basic security hardening
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Install essential packages
apt-get install -y \
    curl \
    wget \
    git \
    htop \
    iotop \
    ncdu \
    tmux \
    vim \
    python3-pip \
    python3-venv \
    build-essential \
    software-properties-common

# Install Docker for containerized ML workloads
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install CloudWatch agent for monitoring
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Install NVIDIA drivers if GPU instance (optional)
if lspci | grep -i nvidia > /dev/null; then
    echo "GPU detected, installing NVIDIA drivers..."
    apt-get install -y nvidia-driver-535 nvidia-utils-535
fi

# Prepare for Ansible
pip3 install ansible

# Create project directory structure
mkdir -p /opt/gemma-agent-loop
chown mlops:mlops /opt/gemma-agent-loop

# Set up swap for memory-intensive ML tasks
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Configure kernel parameters for ML workloads
cat >> /etc/sysctl.conf <<EOF
# Optimize for ML workloads
vm.swappiness=10
vm.overcommit_memory=1
net.core.somaxconn=65535
net.ipv4.tcp_max_syn_backlog=65535
EOF
sysctl -p

# Install monitoring tools
apt-get install -y prometheus-node-exporter
systemctl enable prometheus-node-exporter
systemctl start prometheus-node-exporter

echo "User data script completed at $(date)"
echo "Instance ready for Ansible provisioning and Gemma-3N-Agent-Loop deployment"