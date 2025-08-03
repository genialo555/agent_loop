output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.agent_vm.id
}

output "public_ip" {
  description = "Public IP address of the agent VM"
  value       = aws_eip.agent_vm.public_ip
}

output "private_ip" {
  description = "Private IP address of the agent VM"
  value       = aws_instance.agent_vm.private_ip
}

output "public_dns" {
  description = "Public DNS name of the agent VM"
  value       = aws_instance.agent_vm.public_dns
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.agent_vm.id
}

output "ssh_connection" {
  description = "SSH connection string"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_eip.agent_vm.public_ip}"
}

output "service_urls" {
  description = "URLs for accessing services"
  value = {
    ssh        = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_eip.agent_vm.public_ip}"
    api        = "http://${aws_eip.agent_vm.public_ip}:8000"
    grafana    = "http://${aws_eip.agent_vm.public_ip}:3000"
    prometheus = "http://${aws_eip.agent_vm.public_ip}:9090"
  }
}

output "instance_details" {
  description = "Detailed instance information"
  value = {
    instance_type     = aws_instance.agent_vm.instance_type
    availability_zone = aws_instance.agent_vm.availability_zone
    ami_id           = aws_instance.agent_vm.ami
    root_volume_size = var.root_volume_size
  }
}