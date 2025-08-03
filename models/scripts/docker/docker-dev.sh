#!/bin/bash
# Script de démarrage pour l'environnement de développement Docker

set -e

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon n'est pas en cours d'exécution"
        exit 1
    fi
    
    log_success "Docker est disponible"
}

check_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose n'est pas installé"
        exit 1
    fi
    
    log_success "Docker Compose est disponible"
}

# Fonction principale
main() {
    log_info "Démarrage de l'environnement de développement Agent Loop"
    
    # Vérifications
    check_docker
    check_compose
    
    # Choix du profil
    PROFILE="cpu"
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU NVIDIA détecté, utilisation du profil GPU"
        PROFILE="gpu"
    else
        log_warning "Pas de GPU détecté, utilisation du profil CPU"
    fi
    
    # Build des images si nécessaire
    log_info "Construction des images Docker..."
    if command -v docker-compose &> /dev/null; then
        docker-compose build
    else
        docker compose build
    fi
    
    # Démarrage des services
    log_info "Démarrage des services avec profil: $PROFILE"
    if command -v docker-compose &> /dev/null; then
        docker-compose --profile $PROFILE up -d
    else
        docker compose --profile $PROFILE up -d
    fi
    
    # Attendre que les services soient prêts
    log_info "Attente de la disponibilité des services..."
    sleep 10
    
    # Vérification des services
    log_info "Vérification des services..."
    
    # FastAPI
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "FastAPI: ✓ http://localhost:8000"
    else
        log_warning "FastAPI: ✗ http://localhost:8000 (peut prendre quelques minutes)"
    fi
    
    # Ollama
    if curl -f http://localhost:11434/api/version &> /dev/null; then
        log_success "Ollama: ✓ http://localhost:11434"
    else
        log_warning "Ollama: ✗ http://localhost:11434 (peut prendre quelques minutes)"
    fi
    
    # Prometheus
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus: ✓ http://localhost:9090"
    else
        log_warning "Prometheus: ✗ http://localhost:9090"
    fi
    
    # Grafana
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana: ✓ http://localhost:3000 (admin/admin123)"
    else
        log_warning "Grafana: ✗ http://localhost:3000"
    fi
    
    log_success "Environnement de développement démarré!"
    log_info "Documentation: README_DOCKER.md"
    log_info "Logs: docker-compose logs -f"
    log_info "Arrêt: docker-compose down"
}

# Gestion des arguments
case "${1:-}" in
    "stop")
        log_info "Arrêt de l'environnement de développement"
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        log_success "Environnement arrêté"
        ;;
    "logs")
        if command -v docker-compose &> /dev/null; then
            docker-compose logs -f
        else
            docker compose logs -f
        fi
        ;;
    "status")
        if command -v docker-compose &> /dev/null; then
            docker-compose ps
        else
            docker compose ps
        fi
        ;;
    "init-ollama")
        log_info "Initialisation d'Ollama avec le modèle Gemma"
        if command -v docker-compose &> /dev/null; then
            docker-compose --profile init up ollama-init
        else
            docker compose --profile init up ollama-init
        fi
        ;;
    *)
        main
        ;;
esac