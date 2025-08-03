terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Backend configuration for state management
  backend "s3" {
    # Configure these values in backend.tf or via CLI
    # bucket = "your-terraform-state-bucket"
    # key    = "gemma-agent-loop/terraform.tfstate"
    # region = "us-east-1"
    # encrypt = true
    # dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Repository  = "Gemma-3N-Agent-Loop"
    }
  }
}

# Ubuntu 24.04 LTS AMI data source
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  
  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

# Security group for agent VM
resource "aws_security_group" "agent_vm" {
  name        = "${var.project_name}-agent-sg"
  description = "Security group for Gemma Agent VM"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_ips
  }

  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.api_allowed_ips
  }

  ingress {
    description = "Grafana"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = var.monitoring_allowed_ips
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.api_allowed_ips
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.api_allowed_ips
  }

  ingress {
    description = "Prometheus"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = var.monitoring_allowed_ips
  }

  ingress {
    description = "Prometheus Node Exporter"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-agent-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# EC2 instance for agent VM
resource "aws_instance" "agent_vm" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name              = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.agent_vm.id]
  subnet_id             = var.subnet_id
  availability_zone      = var.availability_zone
  
  # Enable detailed monitoring for MLOps
  monitoring = true
  
  # EBS optimization for better disk performance
  ebs_optimized = var.enable_ebs_optimization
  
  # Termination protection for production
  disable_api_termination = var.enable_termination_protection
  
  # Instance metadata options for security
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size
    encrypted   = true
    iops        = 3000
    throughput  = 125
    
    tags = {
      Name = "${var.project_name}-agent-root-volume"
    }
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    project_name = var.project_name
    environment  = var.environment
  })
  
  # Ensure instance is replaced if user data changes
  user_data_replace_on_change = true

  tags = {
    Name        = "${var.project_name}-agent-vm"
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "MLOps Agent Loop"
  }
  
  lifecycle {
    create_before_destroy = true
  }
}

# Elastic IP for stable addressing
resource "aws_eip" "agent_vm" {
  instance = aws_instance.agent_vm.id
  domain   = "vpc"

  tags = {
    Name        = "${var.project_name}-agent-eip"
    Project     = var.project_name
    Environment = var.environment
  }
}