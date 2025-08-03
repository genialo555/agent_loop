# 📋 Procédures de Déploiement MLOps - Agent Loop

## Vue d'ensemble

Ce document décrit les procédures opérationnelles complètes pour le déploiement, la surveillance et la maintenance du système Agent Loop MLOps. Il couvre tous les scénarios de déploiement, de la mise à jour normale au rollback d'urgence.

## 🎯 Philosophie de Déploiement

**Principes Fondamentaux :**
- **Zero Downtime** : Tous les déploiements doivent maintenir la disponibilité du service
- **Observability First** : Chaque déploiement génère des métriques et logs détaillés  
- **Rollback Ready** : Capacité de rollback automatique en moins de 2 minutes
- **Progressive Rollout** : Déploiements graduels avec validation à chaque étape
- **Infrastructure as Code** : Toutes les configurations sont versionnées

## 🔄 Types de Déploiement

### 1. Déploiement Normal (Feature/Bugfix)

```bash
# Déploiement automatique via Git
git push origin main

# Déploiement manuel si nécessaire
make deploy-prod
```

**Triggers :**
- Push sur branche `main`
- Merge de Pull Request approuvée
- Tag sémantique (`v1.2.3`)

**Pipeline :**
1. Tests automatiques (unit + integration)
2. Analyse statique et sécurité
3. Build Docker multiplatform
4. Déploiement staging + tests de fumée
5. Déploiement production Blue-Green
6. Monitoring post-déploiement

### 2. Déploiement de Modèle

```bash
# Trigger entraînement + déploiement
git commit -m "feat: new model training [train-model]"
git push origin main

# Déploiement manuel d'un modèle spécifique
python scripts/deploy_model.py --version v2.1.3-abc123 --environment production
```

**Process :**
1. Entraînement automatique sur données récentes
2. Validation A/B contre modèle actuel
3. Tests de performance et drift
4. Déploiement canary (10% trafic)
5. Montée en charge progressive
6. Monitoring drift + performance

### 3. Déploiement d'Urgence (Hotfix)

```bash
# Création branche hotfix
git checkout -b hotfix/critical-security-fix main
# ... développement ...
git push origin hotfix/critical-security-fix

# Pipeline accélérée (bypass some checks)
# Déploiement direct après tests critiques
```

**Caractéristiques :**
- Tests réduits mais critiques uniquement
- Bypass de certaines validations
- Notification immédiate des équipes
- Monitoring renforcé post-déploiement

## 🚀 Procédures Détaillées

### Déploiement Production Standard

#### Phase 1: Pré-Déploiement (Automatique)

```yaml
# .github/workflows/ci.yml extracts
pre_deployment_checks:
  - Code quality (Black, Ruff, MyPy)
  - Security scan (Bandit, Trivy)
  - Unit tests (>90% coverage)
  - Integration tests
  - Performance benchmarks
  - Docker build multiplatform
```

#### Phase 2: Déploiement Staging

```bash
# Commandes exécutées automatiquement
ansible-playbook ansible/site.yml \
  -i ansible/inventory.yml \
  -e "environment=staging" \
  -e "docker_image=ghcr.io/repo/agent-loop:sha-abc123" \
  --limit staging

# Tests de fumée automatiques
python scripts/smoke_tests.py --target staging --comprehensive
python scripts/benchmark.py --endpoint https://staging.agent-loop.com --duration 300s
```

**Critères de Validation :**
- Tous les endpoints répondent (< 2s latency P95)
- Intégration Ollama fonctionnelle
- Métriques Prometheus disponibles
- Tests de charge passés (500 req/min pendant 5min)

#### Phase 3: Déploiement Production Blue-Green

```bash
# Déploiement automatique via Ansible
ansible-playbook ansible/playbooks/blue-green-deploy.yml \
  -i ansible/inventory.yml \
  -e "docker_image=ghcr.io/repo/agent-loop:sha-abc123" \
  -e "deployment_type=blue_green" \
  --limit production

# Monitoring du déploiement
watch -n 5 'curl -s http://production/health | jq'
```

**Étapes Détaillées :**

1. **Validation Pré-Déploiement**
   ```bash
   # Vérification infrastructure
   ansible all -i inventory.yml -m ping --limit production
   docker image inspect ghcr.io/repo/agent-loop:sha-abc123
   
   # Vérification ressources
   free -h && df -h && systemctl status docker nginx
   ```

2. **Déploiement Environment Green**
   ```bash
   # Démarrage nouvelle version sur port alternatif
   docker-compose -f docker-compose.green.yml up -d
   
   # Tests de santé internes
   curl -f http://localhost:8002/health
   curl -f http://localhost:8002/ready
   ```

3. **Tests de Fumée**
   ```bash
   # Test API complet
   python scripts/comprehensive_test.py --endpoint http://localhost:8002
   
   # Test intégration Ollama
   curl -X POST http://localhost:8002/run-agent \
     -H "Content-Type: application/json" \
     -d '{"instruction": "Hello test", "use_ollama": true}'
   ```

4. **Basculement Canary (10% trafic)**
   ```nginx
   # Configuration Nginx automatique
   upstream backend_current { server localhost:8001; }
   upstream backend_target  { server localhost:8002; }
   
   split_clients $remote_addr $backend {
       10% backend_target;
       *   backend_current;
   }
   ```

5. **Monitoring Canary**
   ```bash
   # Surveillance métriques pendant 5 minutes
   python scripts/canary_monitor.py \
     --duration 300 \
     --error-threshold 0.01 \
     --latency-threshold 2000ms
   ```

6. **Basculement Complet**
   ```bash
   # Si métriques OK, basculement 100%
   ansible-playbook ansible/playbooks/complete-rollout.yml
   
   # Arrêt ancienne version
   docker-compose -f docker-compose.blue.yml down
   ```

#### Phase 4: Post-Déploiement

```bash
# Health check complet
python scripts/health_check.py \
  --endpoint https://production.agent-loop.com \
  --comprehensive \
  --exit-on-failure

# Mise à jour monitoring
python scripts/setup_alerts.py \
  --deployment-version sha-abc123 \
  --enable-model-drift-detection

# Notification équipes
curl -X POST $SLACK_WEBHOOK -d '{
  "text": "✅ Production deployment successful: v1.2.3",
  "channel": "#deployments"
}'
```

## 🚨 Procédures d'Urgence

### Rollback Automatique

**Déclencheurs Automatiques :**
- Error rate > 1% pendant 2 minutes
- Latency P95 > 5 secondes pendant 1 minute  
- Health check failures > 3 consécutives
- Memory usage > 95% pendant 1 minute

**Processus Automatique :**
```bash
#!/bin/bash
# scripts/emergency_rollback.sh

echo "🚨 EMERGENCY ROLLBACK INITIATED"

# 1. Identifier version précédente
PREVIOUS_VERSION=$(kubectl rollout history deployment/fastapi-app | tail -2 | head -1)

# 2. Rollback immédiat
kubectl rollout undo deployment/fastapi-app --to-revision=$PREVIOUS_VERSION

# 3. Vérification rollback
kubectl rollout status deployment/fastapi-app --timeout=120s

# 4. Tests de santé
python scripts/health_check.py --endpoint http://production --quick

# 5. Notification
curl -X POST $SLACK_WEBHOOK -d '{
  "text": "🚨 EMERGENCY ROLLBACK COMPLETED",
  "attachments": [{
    "color": "danger",
    "fields": [
      {"title": "Previous Version", "value": "'$PREVIOUS_VERSION'", "short": true},
      {"title": "Rollback Duration", "value": "'$(date)'", "short": true}
    ]
  }]
}'

echo "✅ Emergency rollback completed in $(($SECONDS))s"
```

### Rollback Manuel

```bash
# 1. Identification du problème
kubectl get pods -o wide
kubectl logs -l app=agent-loop --tail=100

# 2. Rollback à version spécifique
ansible-playbook ansible/playbooks/rollback.yml \
  -i ansible/inventory.yml \
  -e "target_version=v1.2.2" \
  --limit production

# 3. Validation post-rollback
make health-check
python scripts/benchmark.py --quick-test

# 4. Communication incident
python scripts/create_incident.py \
  --title "Production Rollback - v1.2.3" \
  --severity "high" \
  --description "Rollback executed due to performance degradation"
```

## 📊 Monitoring et Alertes

### Dashboard de Déploiement

**URL :** `http://grafana.internal:3000/d/deployment/deployment-dashboard`

**Métriques Clés :**
- Deployment frequency (target: >10/week)
- Lead time (target: <2h)
- Mean time to recovery (target: <15min)
- Change failure rate (target: <5%)

### Alertes Critiques

```yaml
# prometheus/rules/deployment_alerts.yml
groups:
  - name: deployment_alerts
    rules:
      - alert: DeploymentFailed
        expr: increase(deployment_failures_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Deployment failed"
          description: "Deployment {{ $labels.version }} failed"
          
      - alert: HighErrorRateAfterDeployment
        expr: rate(http_requests_total{status=~"5.."}[2m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected after deployment"
          
      - alert: ModelDriftDetected
        expr: model_drift_score > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model drift detected"
          description: "Model drift score: {{ $value }}"
```

### Notifications Slack

```python
# scripts/slack_notifier.py
class SlackNotifier:
    def __init__(self, webhook_url: str, channel: str = "#deployments"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def deployment_started(self, version: str, environment: str):
        """Notification début déploiement."""
        return self.send_message({
            "text": f"🚀 Deployment started: {version} → {environment}",
            "color": "good"
        })
    
    def deployment_completed(self, version: str, environment: str, duration: int):
        """Notification déploiement réussi."""
        return self.send_message({
            "text": f"✅ Deployment completed: {version} → {environment}",
            "attachments": [{
                "color": "good",
                "fields": [
                    {"title": "Duration", "value": f"{duration}s", "short": True},
                    {"title": "Environment", "value": environment, "short": True}
                ]
            }]
        })
    
    def deployment_failed(self, version: str, environment: str, error: str):
        """Notification échec déploiement."""
        return self.send_message({
            "text": f"❌ Deployment failed: {version} → {environment}",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Error", "value": error[:200], "short": False}
                ]
            }]
        })
```

## 🔧 Maintenance et Optimisation

### Maintenance Programmée

**Fréquence :** Tous les dimanche 02:00 UTC

```bash
#!/bin/bash
# scripts/scheduled_maintenance.sh

echo "🔧 Starting scheduled maintenance"

# 1. Backup complet
python scripts/backup_system.py --full --s3-upload

# 2. Nettoyage Docker
docker system prune -f
docker volume prune -f

# 3. Rotation logs
find /var/log -name "*.log" -mtime +30 -delete
journalctl --vacuum-time=30d

# 4. Mise à jour packages système
apt update && apt upgrade -y

# 5. Tests de santé post-maintenance
python scripts/health_check.py --comprehensive

# 6. Rapport maintenance
python scripts/maintenance_report.py --email ops@company.com

echo "✅ Scheduled maintenance completed"
```

### Optimisation Performance

```bash
# scripts/performance_optimization.sh

# 1. Analyse performance actuelle
python scripts/performance_analysis.py --generate-report

# 2. Optimisation images Docker
docker buildx build --platform linux/amd64,linux/arm64 \
  --cache-from type=registry,ref=myrepo/buildcache \
  --cache-to type=registry,ref=myrepo/buildcache,mode=max \
  --push -t myrepo/agent-loop:optimized .

# 3. Tuning base de données
psql -d agent_loop -c "VACUUM ANALYZE;"
psql -d agent_loop -c "REINDEX DATABASE agent_loop;"

# 4. Optimisation Nginx
nginx -t && systemctl reload nginx

# 5. Monitoring post-optimisation
python scripts/benchmark.py --duration 600s --report optimization_results.json
```

## 📋 Checklist de Déploiement

### Pré-Déploiement ✅

- [ ] Tests passés (unit + integration + e2e)
- [ ] Analyse sécurité complète (Bandit + Trivy)
- [ ] Review de code approuvée
- [ ] Documentation mise à jour
- [ ] Versioning correct (semantic versioning)
- [ ] Backup système effectué
- [ ] Équipes notifiées (#deployments)

### Déploiement ✅

- [ ] Staging deployment successful
- [ ] Smoke tests passed
- [ ] Performance benchmarks OK
- [ ] Blue-Green deployment started
- [ ] Canary phase (10%) validated
- [ ] Full rollout completed
- [ ] Old version stopped

### Post-Déploiement ✅

- [ ] Health checks passed
- [ ] Monitoring dashboards updated
- [ ] Alerts configured
- [ ] Performance metrics normal
- [ ] Error rates < 0.1%
- [ ] Documentation deployed
- [ ] Team notification sent
- [ ] Deployment logged

### En Cas de Problème 🚨

- [ ] Incident créé et catégorisé
- [ ] Équipe On-Call notifiée
- [ ] Rollback initié si nécessaire
- [ ] Post-mortem planifié
- [ ] Actions correctives identifiées
- [ ] Processus amélioré

## 📞 Contacts d'Urgence

**On-Call Rotation :**
- Primary: ops-primary@company.com
- Secondary: ops-secondary@company.com
- Escalation: engineering-lead@company.com

**Slack Channels :**
- `#incidents` - Incidents production
- `#deployments` - Notifications déploiement  
- `#on-call` - Équipe On-Call

**Dashboards :**
- Production Health: http://grafana.internal/d/production
- Deployment Pipeline: http://jenkins.internal/job/agent-loop
- Infrastructure: http://grafana.internal/d/infrastructure

Cette documentation garantit des déploiements sûrs, reproductibles et observables pour le système Agent Loop MLOps.