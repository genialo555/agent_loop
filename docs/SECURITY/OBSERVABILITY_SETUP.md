# Configuration Stack d'Observabilité

## Vue d'ensemble

Cette documentation décrit la configuration complète de la stack d'observabilité mise en place pour l'API Agent Loop, incluant Prometheus, Node Exporter, Grafana 11.1, et l'instrumentation FastAPI.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI API   │    │   Prometheus    │    │   Grafana 11.1  │
│                 │───▶│                 │───▶│                 │
│ /metrics        │    │ :9090           │    │ :3000           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Node Exporter  │    │  Alert Manager  │    │   Dashboard     │
│                 │    │                 │    │   vm-agent      │
│ :9100           │    │ :9093           │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Composants Déployés

### 1. Prometheus (Port 9090)
- **Version**: 2.48.0
- **Configuration**: `/etc/prometheus/prometheus.yml`
- **Données**: `/var/lib/prometheus`
- **Rétention**: 30 jours / 10GB max

**Jobs configurés**:
- `prometheus`: Auto-monitoring
- `node-exporter`: Métriques système (port 9100)
- `fastapi-app`: Métriques application (port 8000)

### 2. Node Exporter (Port 9100) 
- **Version**: 1.7.0
- **Collecteurs activés**:
  - `systemd`: Services système
  - `processes`: Processus
  - `filesystem`: Système de fichiers (excluant containers)
  - `netclass`: Classes réseau (excluant Docker/veth)

### 3. Grafana (Port 3000)
- **Version**: 11.1.0
- **Accès**: admin/changeme (à changer en production!)
- **Datasource**: Prometheus automatiquement configurée
- **Dashboard**: VM Agent Dashboard (uid: vm-agent)

### 4. API FastAPI Instrumentée
- **Endpoint métriques**: `/metrics`
- **Logging structuré**: JSON avec structlog
- **Correlation IDs**: Header `X-Correlation-ID`

## Métriques Collectées

### Métriques Système (Node Exporter)
- **CPU**: Utilisation par core et globale
- **Mémoire**: Utilisation, disponible, buffers/cache
- **Disque**: Espace utilisé/libre par partition
- **Réseau**: Trafic, erreurs, paquets perdus
- **Processus**: Nombre, états, ressources

### Métriques Application (FastAPI)
- `http_requests_total`: Compteur requêtes HTTP par méthode/handler/status
- `http_request_duration_seconds`: Histogram latence des requêtes
- `inference_requests_total`: Compteur inférences par type/status
- `inference_duration_seconds`: Histogram temps d'inférence
- `active_requests`: Gauge requêtes actives
- `webhook_notifications_total`: Compteur notifications webhook
- `app_info`: Informations version/environnement

## Alertes Configurées

### Alertes Système
1. **HighCPUUsage**: CPU > 80% pendant 2min
2. **HighMemoryUsage**: Mémoire > 85% pendant 2min  
3. **DiskSpaceLow**: Disque > 90% pendant 1min
4. **NodeDown**: Node Exporter indisponible > 1min

### Alertes Application
1. **FastAPIDown**: API indisponible > 30s
2. **FastAPIHighLatency**: P95 latence > 1s pendant 2min
3. **FastAPIHighErrorRate**: Taux erreur 5xx > 5% pendant 2min

## Dashboard Grafana "vm-agent"

### Panneaux configurés:
1. **CPU Usage**: Utilisation CPU temps réel avec seuils (60%/80%)
2. **Memory Usage**: Utilisation mémoire avec seuils (70%/85%)
3. **Disk Usage**: Utilisation disque par partition avec seuils (80%/90%)
4. **API Request Rate**: Taux de requêtes par endpoint
5. **API Response Time**: Percentiles p50/p95/p99 des temps de réponse
6. **HTTP Status Codes**: Répartition des codes de statut 2xx/3xx/4xx/5xx

## Déploiement

### Via Ansible
```bash
cd ansible
ansible-playbook -i inventory.yml playbooks/monitoring-setup.yml
```

### Vérification du déploiement
```bash
# Vérifier les services
sudo systemctl status prometheus
sudo systemctl status node_exporter
sudo systemctl status grafana-server

# Vérifier les endpoints
curl http://localhost:9090/targets     # Prometheus targets
curl http://localhost:9100/metrics     # Node Exporter metrics
curl http://localhost:8000/metrics     # FastAPI metrics
curl http://localhost:3000             # Grafana UI
```

## Configuration Logging Structuré

L'API utilise `structlog` pour un logging JSON structuré:

```python
# Variables contextuelles automatiques
correlation_id, method, path, client_ip

# Logs d'exemple
{
  "event": "Request started",
  "level": "info", 
  "timestamp": "2025-07-28T10:30:00.123Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/run-agent",
  "client_ip": "192.168.1.100"
}
```

## Bonnes Pratiques de Production

### Sécurité
1. **Changer les mots de passe par défaut** (Grafana admin)
2. **Configurer HTTPS** pour tous les endpoints
3. **Restreindre l'accès réseau** aux ports monitoring
4. **Activer l'authentification** Prometheus si exposition externe

### Performance
1. **Surveiller l'utilisation disque** Prometheus (rétention 30j/10GB)
2. **Ajuster les intervalles** de scraping selon la charge
3. **Utiliser les labels** avec parcimonie pour éviter la cardinalité haute
4. **Configurer le multiprocessing** Prometheus si nécessaire

### Maintenance
1. **Sauvegarder** les configurations Grafana (`/var/lib/grafana`)
2. **Monitorer** les métriques des outils de monitoring eux-mêmes
3. **Tester les alertes** régulièrement
4. **Documenter** les seuils et leur justification

## Troubleshooting

### Problèmes fréquents

**Prometheus ne scrape pas les métriques FastAPI**:
```bash
# Vérifier que l'API expose /metrics
curl http://localhost:8000/metrics

# Vérifier la configuration Prometheus
sudo /usr/local/bin/promtool check config /etc/prometheus/prometheus.yml
```

**Grafana ne trouve pas les métriques**:
- Vérifier la datasource Prometheus dans Grafana
- Tester les requêtes manuellement dans Prometheus
- Vérifier la connectivité réseau entre Grafana et Prometheus

**Alertes ne se déclenchent pas**:
```bash
# Vérifier les règles d'alerte
sudo /usr/local/bin/promtool check rules /etc/prometheus/alert_rules.yml

# Vérifier l'état des alertes dans Prometheus UI
# http://localhost:9090/alerts
```

## Monitoring des Métriques Clés

### SLIs (Service Level Indicators) recommandés:
- **Disponibilité**: `up{job="fastapi-app"}` 
- **Latence**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Taux d'erreur**: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
- **Débit**: `rate(http_requests_total[5m])`

### SLOs (Service Level Objectives) suggérés:
- **Disponibilité**: 99.9% (moins de 43min d'indisponibilité/mois)
- **Latence P95**: < 500ms pour 95% des requêtes
- **Taux d'erreur**: < 1% d'erreurs 5xx
- **Débit**: Supporter au moins 100 req/s

Cette configuration fournit une observabilité complète de votre stack Agent Loop avec des alertes proactives et des tableaux de bord riches pour le monitoring en temps réel.