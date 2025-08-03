# FastAPI Modular Architecture - Sprint 1

## 🏗️ Architecture Overview

Cette nouvelle architecture modulaire respecte les bonnes pratiques FastAPI pour une application production-ready avec patterns async optimisés.

## 📁 Structure des dossiers

```
inference/
├── api.py                    # Application originale (conservée pour compatibilité)
├── app.py                    # Nouvelle application modulaire
├── groupthink.py            # Module groupthink (inchangé)
├── models/
│   ├── __init__.py
│   └── schemas.py           # Tous les modèles Pydantic centralisés
├── services/
│   ├── __init__.py
│   ├── dependencies.py     # Factories de dépendances
│   ├── external_api.py     # Service API externes
│   ├── health.py           # Service health checks complets
│   └── ollama.py           # Service Ollama optimisé
├── routers/
│   ├── __init__.py
│   ├── agents.py           # Endpoints agents
│   ├── health.py           # Endpoints health/ready/live
│   └── ollama.py           # Endpoints Ollama
└── middleware/
    ├── __init__.py
    └── security.py        # Middleware sécurité et logging
```

## 🚀 Améliorations apportées

### 1. **Endpoints Health robustes**
- `/health` - Basic health check (rapide)
- `/health/detailed` - Health check complet avec métriques système
- `/health/ready` - Readiness probe Kubernetes
- `/health/live` - Liveness probe Kubernetes  
- `/health/startup` - Startup probe Kubernetes

### 2. **Service Ollama optimisé**
- Health check avec cache (30s TTL)
- Métriques de performance détaillées
- Gestion d'erreurs robuste
- Support changement de modèle
- Vérification disponibilité des modèles

### 3. **Middleware de sécurité**
- Headers de sécurité complets
- Rate limiting en mémoire (60 req/min par IP)
- Logging structuré avec correlation IDs
- Gestion des erreurs améliorée

### 4. **Architecture async conforme**
- ✅ Tous les handlers sont `async def`
- ✅ Utilisation d'`httpx.AsyncClient`
- ✅ Patterns `Depends()` pour injection
- ✅ `BackgroundTasks` pour fire-and-forget
- ✅ Validation Pydantic stricte

## 🎯 Endpoints disponibles

### Health Checks
- `GET /health` - Health check basique
- `GET /health/detailed` - Health check détaillé
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /health/startup` - Startup probe

### Agents
- `POST /agents/run` - Exécution agent classique
- `POST /agents/run-agent` - Agent avec Ollama

### Ollama
- `GET /ollama/health` - Health Ollama
- `GET /ollama/health/detailed` - Health détaillé Ollama
- `GET /ollama/model-info` - Info modèle chargé
- `GET /ollama/models` - Liste modèles disponibles
- `POST /ollama/model/switch` - Changer de modèle

### Métriques
- `GET /metrics` - Métriques Prometheus

## 🔧 Configuration

### Variables d'environnement
```bash
ENVIRONMENT=development|production
ALLOWED_ORIGINS=http://localhost:3000,https://app.example.com
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

### Démarrage de l'application
```bash
# Nouvelle architecture modulaire
python -m uvicorn inference.app:app --host 0.0.0.0 --port 8000

# Ancienne architecture (compatibilité)
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

## 📊 Monitoring et observabilité

### Métriques Prometheus
- `http_requests_total` - Nombre total requêtes HTTP
- `http_request_duration_seconds` - Latence requêtes
- `inference_requests_total` - Requêtes d'inférence
- `inference_duration_seconds` - Temps d'inférence
- `active_requests` - Requêtes actives
- `webhook_notifications_total` - Notifications webhook

### Logging structuré
- Correlation IDs sur toutes les requêtes
- Métriques de performance Ollama
- Erreurs avec stack traces complètes
- Format JSON en production

## 🛡️ Sécurité

### Headers appliqués automatiquement
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- `Strict-Transport-Security: max-age=31536000`

### Rate Limiting
- 60 requêtes/minute par IP par défaut
- Exceptions pour health checks et métriques
- Headers `X-RateLimit-*` informatifs

## 🔍 Patterns respectés

### API001 ✅ : Tous les handlers sont `async def`
### API002 ✅ : Injection via `Depends()`
### API003 ✅ : `BackgroundTasks` pour notifications
### API004 ✅ : Validation Pydantic complète
### API005 ✅ : CORS restrictif + headers sécurité
### API006 ✅ : `httpx.AsyncClient` exclusivement
### API007 ✅ : Health checks Kubernetes complets
### API008 ✅ : Middleware logging + métriques

## 🚦 Migration

1. **Phase 1** : Les deux applications coexistent
2. **Phase 2** : Tests sur la nouvelle architecture
3. **Phase 3** : Migration progressive du trafic
4. **Phase 4** : Suppression de l'ancienne architecture

La nouvelle architecture est accessible via `inference.app:app` tandis que l'ancienne reste sur `inference.api:app`.