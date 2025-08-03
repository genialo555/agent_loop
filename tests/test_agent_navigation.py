#!/usr/bin/env python3
"""
Test de navigation OS pour valider le fine-tuning Gemma-3N sur Agent Instruct
Scénarios réalistes pour un agent qui doit naviguer dans un OS
"""

test_scenarios = [
    {
        "id": "debug_logs",
        "instruction": "Find all error logs from nginx in the last 24 hours, count how many 502 errors occurred, and create a summary report with the top 5 IP addresses causing these errors.",
        "expected_actions": [
            "find /var/log/nginx -name '*.log' -mtime -1",
            "grep -E '502|Bad Gateway' *.log",
            "awk to extract IPs and count",
            "sort | uniq -c | sort -nr | head -5",
            "create report file"
        ],
        "difficulty": "medium"
    },
    {
        "id": "disk_cleanup",
        "instruction": "The server is running out of space. Find the 10 largest directories under /home, identify which ones have log files older than 30 days, and create a cleanup script that would free at least 10GB.",
        "expected_actions": [
            "du -h /home --max-depth=2 | sort -hr | head -10",
            "find large_dirs -name '*.log' -mtime +30",
            "calculate total size with du -c",
            "generate cleanup script with rm commands",
            "add safety checks"
        ],
        "difficulty": "hard"
    },
    {
        "id": "process_monitor",
        "instruction": "A Python process is consuming too much memory. Find all Python processes, identify which one is using more than 2GB RAM, check what script it's running, and create a monitoring script that alerts when any Python process exceeds 3GB.",
        "expected_actions": [
            "ps aux | grep python",
            "extract memory usage column",
            "identify process with >2GB",
            "ls -l /proc/PID/cwd",
            "create monitoring script with threshold"
        ],
        "difficulty": "medium"
    },
    {
        "id": "security_audit",
        "instruction": "Perform a security audit: find all files with SUID bit set, check for any suspicious scripts in /tmp modified in the last 7 days, and list all users who have logged in during the weekend.",
        "expected_actions": [
            "find / -perm -4000 -type f 2>/dev/null",
            "find /tmp -type f -mtime -7 -name '*.sh'",
            "last | grep 'Sat\\|Sun'",
            "compile security report"
        ],
        "difficulty": "hard"
    },
    {
        "id": "backup_automation",
        "instruction": "Set up an automated backup: find all MySQL databases, calculate their total size, create a backup script that dumps each database to /backup/mysql/ with today's date, and set up a cron job to run it daily at 2 AM.",
        "expected_actions": [
            "mysql -e 'SHOW DATABASES'",
            "calculate sizes from information_schema",
            "create mysqldump script with date variables",
            "add to crontab: 0 2 * * *",
            "test script execution"
        ],
        "difficulty": "hard"
    },
    {
        "id": "network_debug",
        "instruction": "The API server on port 8080 is not responding. Check if the port is listening, find which process should be using it, check the last 50 lines of its logs, and restart the service if needed.",
        "expected_actions": [
            "netstat -tlnp | grep 8080 or ss -tlnp | grep 8080",
            "check systemctl status api-server",
            "journalctl -u api-server -n 50",
            "systemctl restart api-server",
            "verify port is listening again"
        ],
        "difficulty": "medium"
    },
    {
        "id": "git_analysis",
        "instruction": "In the current git repository, find the 5 contributors who made the most commits in the last month, identify which files they modified most frequently, and create a contribution report in markdown format.",
        "expected_actions": [
            "git log --since='1 month ago' --format='%an'",
            "sort | uniq -c | sort -nr | head -5",
            "git log --author='name' --name-only",
            "aggregate file changes",
            "generate markdown report"
        ],
        "difficulty": "medium"
    },
    {
        "id": "docker_cleanup",
        "instruction": "Docker is using too much disk space. List all containers and images, remove all stopped containers, delete unused images, and create a maintenance script that does this cleanup weekly.",
        "expected_actions": [
            "docker ps -a",
            "docker images",
            "docker container prune -f",
            "docker image prune -a -f",
            "create cleanup script with cron"
        ],
        "difficulty": "easy"
    },
    {
        "id": "performance_analysis",
        "instruction": "The website is slow. Check CPU and memory usage, identify the top 5 processes consuming resources, analyze Apache access logs for the slowest endpoints, and suggest optimizations.",
        "expected_actions": [
            "top -b -n 1 | head -20",
            "analyze apache logs for response times",
            "grep for endpoints and calculate averages",
            "check for slow database queries",
            "create optimization report"
        ],
        "difficulty": "hard"
    },
    {
        "id": "ssl_cert_check",
        "instruction": "Check all SSL certificates on the server, identify which ones expire in the next 30 days, and create a renewal script for Let's Encrypt certificates.",
        "expected_actions": [
            "find /etc/ssl -name '*.crt'",
            "openssl x509 -enddate -noout -in cert",
            "calculate days until expiry",
            "identify Let's Encrypt certs",
            "create certbot renewal script"
        ],
        "difficulty": "medium"
    }
]

def format_test_prompt(scenario):
    """Format le test en prompt pour le modèle"""
    return f"""You are an AI assistant with access to a Linux system. You need to help with system administration tasks.

User request: {scenario['instruction']}

Please provide the exact commands you would run to accomplish this task, explaining each step.
"""

def evaluate_response(response, expected_actions):
    """Évalue si la réponse contient les bonnes actions"""
    score = 0
    for action in expected_actions:
        # Check if key concepts are present
        key_parts = action.split()
        if all(part.lower() in response.lower() for part in key_parts if len(part) > 2):
            score += 1
    return score / len(expected_actions)

# Sauvegarder les tests
if __name__ == "__main__":
    import json
    
    # Sauver en JSON pour utilisation ultérieure
    with open("agent_navigation_tests.json", "w") as f:
        json.dump(test_scenarios, f, indent=2)
    
    # Créer un fichier de prompt pour test manuel
    with open("test_prompts.txt", "w") as f:
        for scenario in test_scenarios:
            f.write(f"=== Test {scenario['id']} ({scenario['difficulty']}) ===\n")
            f.write(format_test_prompt(scenario) + "\n\n")
            f.write("Expected actions:\n")
            for action in scenario['expected_actions']:
                f.write(f"- {action}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print("✅ Tests créés !")
    print(f"📝 {len(test_scenarios)} scénarios de test")
    print("📊 Difficulté : 2 easy, 5 medium, 3 hard")
    print("\nFichiers générés:")
    print("- agent_navigation_tests.json : Tests structurés")
    print("- test_prompts.txt : Prompts formatés pour test manuel")