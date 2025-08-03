variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "gemma-agent-loop"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID where resources will be created"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type (4 vCPU, 12 GB RAM requirement)"
  type        = string
  default     = "m6i.xlarge"  # 4 vCPU, 16 GB RAM - better for ML workloads
  
  validation {
    condition = can(regex("^(m6i|m6a|m5|m5a|c6i|c5)\\.(xlarge|2xlarge)", var.instance_type))
    error_message = "Instance type should be at least xlarge with 4 vCPU and 12+ GB RAM."
  }
}

variable "key_pair_name" {
  description = "AWS key pair name for SSH access"
  type        = string
}

variable "ssh_allowed_ips" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

variable "api_allowed_ips" {
  description = "CIDR blocks allowed for API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

variable "monitoring_allowed_ips" {
  description = "CIDR blocks allowed for Grafana access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 30
  
  validation {
    condition     = var.root_volume_size >= 30
    error_message = "Root volume size must be at least 30 GB."
  }
}

variable "availability_zone" {
  description = "AWS availability zone for the instance"
  type        = string
  default     = null
}

variable "enable_termination_protection" {
  description = "Enable EC2 instance termination protection"
  type        = bool
  default     = false
}

variable "enable_ebs_optimization" {
  description = "Enable EBS optimization for the instance"
  type        = bool
  default     = true
}