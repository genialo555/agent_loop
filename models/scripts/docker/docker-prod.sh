#!/bin/bash
# Script de déploiement pour l'environnement de production Docker

set -e

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
IMAGE_NAME="agent-loop"
DATA_DIR="/opt/agent_loop"

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérification des prérequis
check_requirements() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon n'est pas en cours d'exécution"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose n'est pas installé"
        exit 1
    fi
    
    log_success "Prérequis Docker vérifiés"
}

# Préparation de l'environnement de production
setup_production_env() {
    log_info "Préparation de l'environnement de production"
    
    # Création des répertoires de données
    sudo mkdir -p ${DATA_DIR}/{ollama_data,prometheus_data,grafana_data}
    sudo chown -R $USER:$USER ${DATA_DIR}
    
    # Création du secret Grafana si n'existe pas
    if ! docker secret ls | grep -q grafana_admin_password; then
        log_info "Création du secret Grafana"
        echo "$(openssl rand -base64 32)" | docker secret create grafana_admin_password -
        log_success "Secret Grafana créé"
    fi
    
    log_success "Environnement de production préparé"
}

# Build des images optimisées
build_production_images() {
    log_info "Build des images de production"
    
    # Build avec optimisations BuildKit
    DOCKER_BUILDKIT=1 docker build \
        --target runtime \
        --tag ${IMAGE_NAME}:latest \
        --tag ${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S) \
        .
    
    log_success "Images de production construites"
}

# Déploiement
deploy() {
    log_info "Déploiement de l'environnement de production"
    
    # Vérifications
    check_requirements
    setup_production_env
    build_production_images
    
    # Démarrage des services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} up -d
    else
        docker compose -f ${COMPOSE_FILE} up -d
    fi
    
    # Attendre la disponibilité des services
    log_info "Attente de la disponibilité des services..."
    sleep 30
    
    # Vérifications santé
    check_services_health
    
    log_success "Déploiement terminé avec succès!"
}

# Vérification de la santé des services
check_services_health() {
    log_info "Vérification de la santé des services"
    
    local failed=0
    
    # FastAPI via Nginx
    if timeout 10 curl -f http://localhost/health &> /dev/null; then
        log_success "FastAPI (via Nginx): ✓"
    else
        log_error "FastAPI (via Nginx): ✗"
        failed=1
    fi
    
    # Ollama
    if timeout 10 curl -f http://localhost:11434/api/version &> /dev/null; then
        log_success "Ollama: ✓"
    else
        log_error "Ollama: ✗"
        failed=1
    fi
    
    # Prometheus
    if timeout 10 curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus: ✓"
    else
        log_error "Prometheus: ✗"
        failed=1
    fi
    
    # Grafana
    if timeout 10 curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana: ✓"
    else
        log_error "Grafana: ✗"
        failed=1
    fi
    
    if [ $failed -eq 1 ]; then
        log_warning "Certains services ne répondent pas, vérifiez les logs"
        return 1
    fi
    
    return 0
}

# Mise à jour
update() {
    log_info "Mise à jour de l'environnement de production"
    
    # Build nouvelle image
    build_production_images
    
    # Rolling update
    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} up -d --force-recreate fastapi-app
    else
        docker compose -f ${COMPOSE_FILE} up -d --force-recreate fastapi-app
    fi
    
    # Vérification
    sleep 10
    if check_services_health; then
        log_success "Mise à jour terminée avec succès"
    else
        log_error "Problème détecté après mise à jour"
        exit 1
    fi
}

# Arrêt
stop() {
    log_info "Arrêt de l'environnement de production"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f ${COMPOSE_FILE} down
    else
        docker compose -f ${COMPOSE_FILE} down
    fi
    
    log_success "Environnement arrêté"
}

# Nettoyage
cleanup() {
    log_info "Nettoyage des ressources Docker"
    
    # Arrêt des services
    stop
    
    # Suppression des volumes (avec confirmation)
    read -p "Supprimer les volumes de données? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v docker-compose &> /dev/null; then
            docker-compose -f ${COMPOSE_FILE} down -v
        else
            docker compose -f ${COMPOSE_FILE} down -v
        fi
        sudo rm -rf ${DATA_DIR}
        log_success "Volumes supprimés"
    fi
    
    # Nettoyage des images
    docker image prune -f
    
    log_success "Nettoyage terminé"
}

# Gestion des arguments
case "${1:-}" in
    "deploy")
        deploy
        ;;
    "update")
        update
        ;;
    "stop")
        stop
        ;;
    "status")
        if command -v docker-compose &> /dev/null; then
            docker-compose -f ${COMPOSE_FILE} ps
        else
            docker compose -f ${COMPOSE_FILE} ps
        fi
        ;;
    "logs")
        if command -v docker-compose &> /dev/null; then
            docker-compose -f ${COMPOSE_FILE} logs -f
        else
            docker compose -f ${COMPOSE_FILE} logs -f
        fi
        ;;
    "health")
        check_services_health
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {deploy|update|stop|status|logs|health|cleanup}"
        echo ""
        echo "  deploy   - Déploie l'environnement de production complet"
        echo "  update   - Met à jour l'application avec rolling update"
        echo "  stop     - Arrête tous les services"
        echo "  status   - Affiche le statut des services"
        echo "  logs     - Affiche les logs en temps réel"
        echo "  health   - Vérifie la santé des services"
        echo "  cleanup  - Nettoyage complet (avec confirmation)"
        exit 1
        ;;
esac