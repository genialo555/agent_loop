#!/bin/bash
# Script d'installation NVIDIA Container Toolkit pour Docker GPU support

set -e

echo "Installation du NVIDIA Container Toolkit..."

# Add NVIDIA GPG key and repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package list
sudo apt-get update

# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
echo "Configuration du runtime Docker..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
echo "Redémarrage de Docker..."
sudo systemctl restart docker

# Verify installation
echo "Vérification de l'installation..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo "Installation terminée avec succès!"