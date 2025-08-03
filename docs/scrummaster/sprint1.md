Sprint 1 – Architecture initiale et mise en place du projet
docker-container-architect

    Écrire un Dockerfile de base pour containeriser l’application FastAPI (en utilisant une image Python appropriée et en installant les dépendances requises) (Livrable : Dockerfile permettant de construire l’image du projet).

    Créer un fichier docker-compose.yml pour orchestrer l’application et d’éventuels services auxiliaires (par ex. une base de données) en environnement de développement (Livrables : fichier docker-compose.yml fonctionnel pour lancer l’application).

fastapi-async-architect

    Initialiser le projet FastAPI de manière asynchrone : créer la structure de base de l’application (modules, routeurs, configuration) en suivant les bonnes pratiques du framework (Livrable : squelette d’application FastAPI exécutable).

    Implémenter une route de vérification de l’état de l’application (GET /health) qui renvoie une réponse de statut pour confirmer son bon fonctionnement (Livrable : endpoint /health opérationnel renvoyant un statut 200 OK).

llm-optimization-engineer

    Rechercher et évaluer les modèles de langage (LLM) appropriés en fonction des besoins du projet (taille, performance, coût si API externe) (Livrable : recommandation du modèle LLM à utiliser, avec justification documentée).

    Planifier l’approche d’intégration du LLM choisi dans l’architecture : décider si le modèle sera appelé via une API tierce ou hébergé en local, et définir comment gérer l’asynchronisme de ces appels (Livrable : plan d’intégration du LLM rédigé, décrivant l’interface d’appel et les besoins d’infrastructure).

mlops-pipeline-engineer

    Mettre en place une première pipeline d’intégration continue (CI) : configurer un workflow (ex. GitHub Actions, GitLab CI) exécutant les tests et l’analyse statique à chaque commit (Livrable : configuration CI valide dans le dépôt, avec exécution automatique des tests à chaque commit).

    Concevoir l’architecture initiale de la pipeline MLOps : définir les étapes clés du cycle de vie du modèle (préparation des données, entraînement, validation, déploiement) et comment ces étapes s’intègrent dans le projet (Livrable : schéma ou document d’architecture MLOps détaillant le flux du modèle de son entraînement à sa mise en production).

observability-engineer

    RAS (Pas de mise en place d’observabilité à ce stade initial du projet).

python-type-guardian

    Configurer les outils de typage statique : ajouter et paramétrer mypy (ou un équivalent) afin de vérifier les annotations de type lors de l’exécution de la CI (Livrable : fichier de configuration mypy.ini ajouté au projet).

    Revoir le code source initial (routes, modèles de données) pour ajouter les annotations de type manquantes et corriger celles incorrectes, de sorte à respecter les conventions de typage Python (Livrable : code annoté avec des types cohérents, sans avertissement de type lors de l’exécution de mypy).

system-architect

    Définir l’architecture globale du système : identifier les composants principaux (API FastAPI, service LLM, base de données, etc.), leurs interactions et les choix technologiques (langages, frameworks, hébergement) (Livrable : schéma d’architecture et document décrivant les décisions d’architecture initiales).

    Établir les exigences non-fonctionnelles majeures (performances, sécurité, scalabilité) pour guider les choix techniques dès le départ (Livrable : liste des exigences non-fonctionnelles intégrée à la documentation d’architecture).

test-automator

    Mettre en place le framework de tests automatisés : installer et configurer pytest et structurer un répertoire tests/ dans le projet, tout en l’intégrant au workflow CI (Livrable : environnement de test prêt à l’emploi, avec exécution des tests automatisée via CI).

    Écrire un premier test unitaire pour l’endpoint de santé (/health) afin de vérifier que l’application répond correctement (Livrable : test pytest valide passant sur l’endpoint /health dans la CI).

Sprint 2 – Développement du noyau fonctionnel (MVP)
docker-container-architect

    Adapter la configuration Docker aux nouveaux besoins : mettre à jour le Dockerfile ou le docker-compose.yml si de nouvelles dépendances ou services (par ex. base de données, bibliothèque LLM) sont ajoutés (Livrable : configuration Docker mise à jour assurant le bon fonctionnement de l’application avec les nouvelles composantes).

fastapi-async-architect

    Développer l’endpoint principal de l’API correspondant à la fonctionnalité de base du produit. Par exemple, implémenter une route POST (ex. /generate) recevant les données utilisateur et appelant le modèle de langage pour produire une réponse (Livrable : route FastAPI opérationnelle renvoyant une réponse générée par le LLM).

    Gérer l’appel au LLM de manière asynchrone pour ne pas bloquer le serveur : utiliser async/await ou un mécanisme de tâche de fond afin que le traitement du LLM se fasse sans dégrader la réactivité de l’API (Livrable : intégration asynchrone du LLM, confirmée par des tests de performance montrant l’absence de blocage du serveur).

llm-optimization-engineer

    Intégrer le modèle de langage choisi dans l’application : développer le module de connexion (appel API externe ou chargement du modèle local) et l’appeler depuis l’endpoint FastAPI pour obtenir des prédictions (Livrable : code d’intégration du LLM fonctionnel, permettant d’obtenir des réponses du modèle via l’API).

    Effectuer des tests initiaux de qualité et de performance du LLM : exécuter le modèle sur des entrées types pour évaluer la pertinence des réponses et mesurer le temps de réponse moyen (Livrable : rapport de tests initiaux du LLM indiquant la qualité des réponses et la latence constatée).

mlops-pipeline-engineer

    Étendre la pipeline CI/CD pour le déploiement applicatif : ajouter au workflow CI la construction de l’image Docker et son envoi vers un registre à chaque intégration réussie sur la branche principale (Livrable : pipeline CI/CD incluant des étapes de build et push de l’image Docker, validée sur un environnement de test).

    Mettre en place un stockage pour les données d’utilisation du modèle : configurer une base de données ou un autre système afin d’enregistrer les requêtes utilisateurs et les réponses du LLM en vue d’analyses ou de futur entraînement (Livrable : base de données de logs configurée et alimentée par l’application).

observability-engineer

    Implémenter la journalisation détaillée dans l’application : ajouter des logs sur les requêtes entrantes (début, fin, durée, code de réponse) et sur les appels au LLM (délai de réponse, succès/échec) en utilisant la bibliothèque de logging Python (Livrable : fichiers de log détaillés générés lors des tests, permettant de diagnostiquer le comportement de l’application).

    Exposer des métriques de base pour le monitoring : par exemple, ajouter un endpoint /metrics (compatible Prometheus) ou intégrer un collecteur pour suivre le nombre de requêtes, la latence moyenne, les taux d’erreur, etc. (Livrable : métriques de performance accessibles et vérifiées via un outil de supervision).

python-type-guardian

    Revoir le nouveau code ajouté (endpoints et intégration LLM) afin d’y ajouter toutes les annotations de type manquantes : s’assurer que les requêtes, réponses et interactions avec le LLM sont bien typées (Livrable : code mis à jour avec typage complet, sans avertissement mypy sur les modules modifiés).

    Mettre à jour ou créer les modèles de données Pydantic pour valider les entrées/sorties de la nouvelle route : définir des schémas stricts pour les requêtes utilisateur et la réponse du modèle (Livrable : schémas de validation garantissant l’intégrité des données entrantes et sortantes de l’API).

system-architect

    Ajuster la documentation d’architecture en fonction des évolutions : intégrer l’ajout d’une base de données de logs ou l’appel à une API externe LLM dans le schéma, et vérifier que ces modifications respectent les principes définis (Livrable : schéma d’architecture mis à jour incluant les nouveaux composants et flux de données).

    Trancher les décisions techniques transverses apparues pendant le sprint (par ex. format de stockage des logs, configuration de CORS pour l’API) et les consigner pour référence (Livrable : registre des décisions techniques tenu à jour dans la documentation).

test-automator

    Écrire des tests unitaires et d’intégration pour la nouvelle route principale : vérifier que l’endpoint du LLM retourne les résultats attendus pour des entrées valides et des scénarios d’erreur (en simulant l’appel au modèle avec un mock si nécessaire) (Livrable : tests automatisés couvrant les cas nominal et d’erreur de l’endpoint LLM).

    Mettre en place la mesure de couverture du code : configurer l’outil de couverture (par ex. coverage.py) pour générer un rapport à chaque exécution de tests, et définir un seuil minimal de couverture à respecter (Livrable : rapport de couverture intégré à la CI, avec par exemple ≥50% de lignes couvertes).

Sprint 3 – Extension des fonctionnalités et mise en place du pipeline MLOps
docker-container-architect

    Ajouter les nouveaux services conteneurisés requis par les évolutions : par exemple, intégrer un worker Celery (pour les tâches asynchrones en arrière-plan) ou un conteneur dédié pour un modèle ML auto-hébergé si nécessaire (Livrable : fichier docker-compose.yml mis à jour avec les nouveaux conteneurs, vérifié pour un démarrage simultané sans erreurs).

    Optimiser l’image Docker de l’application : mettre en place un build multi-étapes pour réduire la taille de l’image et cache des dépendances pour accélérer les builds (Livrable : Dockerfile optimisé produisant une image plus légère, testé en local).

fastapi-async-architect

    Implémenter les nouvelles fonctionnalités prévues (stories du sprint) via l’API FastAPI : ajouter par exemple des endpoints pour la gestion de l’historique des requêtes utilisateur ou d’autres fonctionnalités complémentaires (Livrable : endpoints supplémentaires documentés et opérationnels, testés via l’API).

    Mettre en place des tâches de fond (background jobs) pour les traitements longs : utiliser le système de BackgroundTasks de FastAPI ou intégrer Celery afin d’exécuter en arrière-plan des opérations coûteuses (ex. entraînement du modèle, pré-calcul) sans impacter les temps de réponse de l’API (Livrable : mécanisme de tâche de fond configuré, déclenchable depuis l’API, vérifié par des tests).

llm-optimization-engineer

    Affiner l’utilisation du modèle de langage : ajuster le prompt envoyé au LLM, ainsi que les hyperparamètres (température, top-k/top-p) pour améliorer la pertinence et la cohérence des réponses (Livrable : configuration optimisée des requêtes LLM, validée par des tests comparant la qualité des réponses avant/après ajustement).

    Mettre en place un pipeline d’entraînement ou de fine-tuning du modèle : utiliser les données collectées (ex. historiques de requêtes et réponses) pour réentraîner le modèle ou affiner un modèle existant, de manière reproductible (Livrable : script ou notebook d’entraînement du modèle capable de générer une nouvelle version du LLM, avec des résultats d’évaluation documentés).

mlops-pipeline-engineer

    Automatiser le pipeline d’entraînement du modèle ML : intégrer le script d’entraînement dans un workflow automatisé (par ex. ajouter un job dédié dans la CI/CD ou configurer un pipeline via Airflow/Kubeflow déclenché manuellement) afin de pouvoir lancer des ré-entraînements réguliers du modèle (Livrable : pipeline d’entraînement du modèle opérationnel, documenté avec les instructions pour l’exécuter et contrôler son succès).

    Mettre en place le versionnage du modèle ML : définir un système pour taguer et stocker chaque nouvelle version du modèle entraîné (par ex. dépôt d’artéfacts ou registre de modèles) permettant de savoir quelle version est déployée à tout moment (Livrable : mécanisme de versionnage du modèle implémenté, avec traçabilité des versions dans la documentation).

observability-engineer

    Établir un tableau de bord de supervision : configurer un outil comme Grafana pour visualiser en temps réel les métriques clés de l’application et du modèle (taux de requêtes, latence, taux d’erreurs, utilisation des ressources…) (Livrable : dashboard d’observabilité accessible, affichant les indicateurs de performance et de santé du système).

    Implémenter un traçage distribué des requêtes : intégrer une solution de tracing (ex. OpenTelemetry ou APM) afin de suivre le parcours d’une requête à travers les composants (depuis l’API FastAPI jusqu’au modèle LLM et la base de données) (Livrable : traçage activé et visible, permettant d’analyser le détail d’une requête pour identifier d’éventuels goulots d’étranglement).

python-type-guardian

    Auditer l’ensemble du code pour renforcer le typage : identifier les portions où les types sont absents ou trop génériques (Any) et refactorer le code pour introduire des types plus précis (Livrable : code refactoré éliminant les usages superflus de Any, sans avertissement de typage sur ces sections).

    Ajouter des validations supplémentaires basées sur les types : par exemple, intégrer des assertions ou utiliser Pydantic pour vérifier à l’exécution que les objets manipulés correspondent aux types attendus, afin de prévenir des erreurs runtime (Livrable : validations runtime ajoutées dans le code aux points critiques, améliorant la robustesse face à des données mal typées).

system-architect

    Planifier la montée en charge : proposer des ajustements d’architecture en prévision d’un trafic utilisateur accru (par ex. possibilité de répliquer le service FastAPI, mise en cache de certaines réponses du LLM, usage d’un CDN si pertinent) et valider que l’architecture pourra évoluer sans goulot d’étranglement majeur (Livrable : section “Scalabilité” ajoutée à la documentation d’architecture, décrivant comment le système peut évoluer en capacité).

    Intégrer les nouvelles composantes dans l’architecture : s’assurer que le pipeline d’entraînement ML, les tâches de fond et la base de données de logs sont bien pris en compte dans l’architecture cible du système et documenter leurs interactions (Livrable : schéma d’architecture mis à jour englobant le pipeline MLOps et les services en arrière-plan introduits).

test-automator

    Mettre en place des tests de bout en bout (E2E) couvrant un scénario utilisateur complet : simuler via un test l’envoi d’une requête à l’API, le traitement par le LLM et la sauvegarde éventuelle en base, afin de valider l’intégration de tous les composants (Livrable : script de test E2E automatisé validant le parcours utilisateur principal).

    Réaliser des tests de performance et de charge : mesurer le temps de réponse de l’API sous plusieurs niveaux de charge (utilisateurs concurrents) et observer le comportement du système lors d’opérations lourdes (ex. déclenchement d’un entraînement en tâche de fond) (Livrable : rapport de tests de performance identifiant la capacité du système (nombre de requêtes/s supportées) et les éventuels bottlenecks, avec des recommandations d’amélioration).

Sprint 4 – Optimisations, robustesse et scalabilité
docker-container-architect

    Préparer l’image Docker pour la production : épurer le Dockerfile en retirant les outils de debug inutiles, en verrouillant les versions de dépendances, et en configurant un serveur d’applications performant (par ex. Gunicorn couplé à Uvicorn) pour exécuter FastAPI (Livrable : image Docker optimisée pour la prod, validée par des tests locaux).

    Contribuer à la stratégie de déploiement conteneurisé : décider avec l’équipe si l’orchestration se fera via Docker Compose en production ou via Kubernetes, et préparer les fichiers/manifestes requis (ex : un docker-compose.prod.yml ou des manifests Kubernetes) (Livrable : documentation de déploiement conteneurisé prête, incluant les fichiers de configuration pour l’environnement de production).

    Effectuer un audit de sécurité des conteneurs : scanner l’image Docker à la recherche de vulnérabilités connues dans les bibliothèques embarquées et appliquer les mises à jour ou correctifs nécessaires dans le Dockerfile (Livrable : rapport de scan de sécurité et Dockerfile ajusté pour corriger les failles critiques identifiées).

fastapi-async-architect

    Optimiser les performances de l’API : identifier les éventuels points de lenteur (profiling) et mettre en œuvre des améliorations comme la mise en cache de certaines requêtes, l’activation de la compression HTTP des réponses, ou l’ajustement du nombre de workers Uvicorn pour mieux gérer la charge (Livrable : API FastAPI optimisée montrant une réduction du temps de réponse moyen sous charge).

    Renforcer la gestion des erreurs et des délais d’attente : implémenter des timeouts appropriés pour les appels au LLM afin d’éviter les blocages, et améliorer la gestion d’erreur pour renvoyer des messages clairs aux clients en cas de problème (Livrable : code de l’API modifié avec une gestion robuste des échecs du LLM, vérifié via tests d’erreurs).

llm-optimization-engineer

    Implémenter des mécanismes d’optimisation des appels au LLM : mettre en cache les réponses du modèle pour les requêtes fréquemment répétées et/ou prévoir l’utilisation d’un modèle plus léger en secours si le modèle principal est indisponible ou trop lent (Livrable : système de cache ou de fallback opérationnel, réduisant la latence moyenne des réponses du LLM).

    Évaluer la qualité du modèle en conditions quasi-réelles : analyser les données d’utilisation accumulées pour détecter d’éventuelles dérives ou biais dans les réponses du LLM, et s’assurer que les performances restent conformes aux attentes. Proposer si besoin un ajustement ou un ré-entraînement final (Livrable : rapport d’évaluation du LLM en pré-production, avec constat de la qualité des réponses et recommandations pour la version finale).

mlops-pipeline-engineer

    Finaliser la pipeline CI/CD de déploiement : intégrer l’étape de déploiement vers l’environnement de production dans le workflow, de sorte qu’une nouvelle version validée de l’application (et du modèle si applicable) soit déployable d’un seul clic ou automatiquement après validation (Livrable : pipeline de déploiement continu opérationnelle, testée sur un environnement de staging).

    Monitorer le pipeline ML : mettre en place des notifications ou alertes en cas d’échec d’un job du pipeline (par ex. entraînement du modèle) et collecter des métriques sur la durée et la fréquence des entraînements pour anticiper les besoins (Livrable : monitoring du pipeline ML en place, avec alertes configurées en cas d’échec et métriques de pipeline visibles sur le dashboard).

observability-engineer

    Mettre en place des alertes sur les métriques critiques : définir des seuils pour les indicateurs importants (par ex. latence > X ms, taux d’erreur > Y%) et configurer l’outil de monitoring pour envoyer des alertes (email, Slack) si ces seuils sont dépassés (Livrable : règles d’alerte actives et vérifiées par des tests de notification).

    Tester la résilience du système de supervision : simuler des incidents (arrêt du service LLM, montée en charge soudaine) afin de vérifier que les alertes se déclenchent et que les logs/metrics collectés suffisent à diagnostiquer le problème (Livrable : rapport de test d’observabilité décrivant chaque scénario de panne simulé, les alertes générées et les enseignements tirés).

python-type-guardian

    Effectuer une passe finale de nettoyage du code orientée typage : corriger tous les derniers avertissements de type signalés par mypy ou le linter, harmoniser les annotations sur l’ensemble du code et supprimer tout contournement temporaire introduit plus tôt (Livrable : base de code sans aucun avertissement de typage, conforme à 100% aux règles définies).

    Durcir la configuration du type-checker : activer des options plus strictes de mypy (mode strict, vérifications additionnelles) pour détecter d’éventuels problèmes subtils, et adapter le code en conséquence pour satisfaire ces nouvelles contraintes (Livrable : configuration de vérification de types renforcée et code ajusté sans nouvelles erreurs).

system-architect

    Réaliser une revue de sécurité de l’architecture : vérifier que les communications entre services sont sécurisées (chiffrement TLS si pertinent), que les données sensibles (ex : clés API, données utilisateur) sont protégées (chiffrées au repos, variables d’environnement sécurisées) et que l’architecture globale respecte les bonnes pratiques de sécurité (Livrable : rapport de revue de sécurité listant les vérifications effectuées et les corrections apportées au besoin).

    Valider la résilience et la scalabilité de l’architecture mise en œuvre : s’assurer que le système en l’état supportera une augmentation du trafic ou des volumes de données (tests de charge à l’appui) et qu’il comporte des mécanismes de tolérance aux pannes (par ex. sauvegardes, redémarrage automatique des services) (Livrable : addendum à la documentation d’architecture confirmant que les objectifs de robustesse et scalabilité sont atteints, avec les résultats de tests correspondants).

test-automator

    Exécuter une campagne complète de tests en prévision de la mise en production : lancer l’intégralité de la suite de tests (unitaires, intégration, E2E) sur la version candidate et ajouter de nouveaux tests pour couvrir tout scénario non testé précédemment, afin de garantir qu’aucune régression n’est présente (Livrable : résultats de la campagne de tests complets, démontrant que tous les tests passent ou que les anomalies détectées ont été corrigées).

    Mener des tests de charge et de stress avancés : pousser le système à ses limites en termes de nombre de requêtes simultanées et de durée (tests d’endurance) pour identifier le point de rupture et observer le comportement en situation extrême (Livrable : rapport de tests de charge intense indiquant la capacité maximale du système, et les éventuelles dégradations au-delà, accompagné de recommandations finales d’optimisation).

Sprint 5 – Finalisation et déploiement
docker-container-architect

    Assister la mise en production de l’application conteneurisée : vérifier que l’image Docker se déploie correctement dans l’environnement cible (variables d’environnement, volumes, ports, configuration réseau) et ajuster les paramètres ou fichiers de configuration en cas de problème (Livrable : application déployée en production via ses conteneurs, avec confirmation que la configuration Docker est opérationnelle).

fastapi-async-architect

    Corriger les derniers bugs et ajustements mineurs identifiés lors des tests finaux : par exemple, affiner la logique de certaines routes ou optimiser des requêtes vers la base de données si des lenteurs ont été détectées (Livrable : correctifs appliqués au code de l’API, validés par une nouvelle exécution de la suite de tests).

    Finaliser la documentation de l’API pour la livraison : s’assurer que le schéma OpenAPI généré par FastAPI est complet (ajout de descriptions, d’exemples aux endpoints si nécessaire) et vérifier que l’interface de documentation automatique (/docs) reflète correctement tous les endpoints et modèles (Livrable : documentation OpenAPI à jour et accessible, prête à être utilisée par les utilisateurs/intégrateurs).

llm-optimization-engineer

    Valider l’utilisation du modèle LLM final en production : s’assurer que la version définitive du modèle (après le dernier entraînement ou réglage) est bien déployée et utilisée par l’API, et qu’elle génère des réponses conformes aux attentes sur des requêtes réelles (Livrable : modèle LLM final en place et fonctionnel en production, vérifié via quelques tests post-déploiement sur l’API).

mlops-pipeline-engineer

    Superviser l’exécution du pipeline de déploiement final : déclencher le workflow de CI/CD de mise en production et suivre le déroulement de chaque étape (tests, build, déploiement). Intervenir en rollback si un problème critique est détecté (Livrable : déploiement final réalisé via la pipeline CI/CD, journal d’exécution conservé montrant le succès du déploiement).

    Documenter le pipeline MLOps mis en place et former l’équipe de maintenance : rédiger un guide expliquant comment lancer un nouvel entraînement de modèle, déployer une nouvelle version de l’application et surveiller les pipelines en production (Livrable : documentation détaillée du pipeline CI/CD et des procédures MLOps, transmise à l’équipe opérationnelle pour la phase de support).

observability-engineer

    Surveiller la plateforme lors du déploiement en production : monitorer en temps réel les métriques et les logs pendant et après la mise en ligne afin de détecter immédiatement tout dysfonctionnement (erreurs inattendues, usage anormal des ressources) (Livrable : rapport de suivi de mise en production confirmant l’absence d’erreurs critiques ou décrivant les incidents résolus rapidement).

    Finaliser la documentation d’observabilité : rédiger un document synthétique sur l’utilisation des tableaux de bord et des alertes (comment interpréter les métriques, quelles actions entreprendre en cas d’alerte) pour l’équipe en charge de l’exploitation continue (Livrable : guide d’utilisation du système de monitoring et d’alerting remis à l’équipe de support).

python-type-guardian

    Générer la documentation technique du code à partir des types et docstrings : utiliser un outil tel que Sphinx (avec autodoc) pour produire une documentation complète de l’API (côté code) et des modules internes, puis vérifier que cette documentation reflète fidèlement le comportement du code final (Livrable : documentation technique finale (par ex. HTML) construite à partir du code et fournie avec le livrable du projet).

system-architect

    Donner l’approbation finale de l’architecture avant la livraison : effectuer une dernière vérification que tous les composants déployés correspondent à l’architecture prévue, et qu’aucun risque majeur n’est en suspens pour le passage en production (Livrable : validation écrite de l’architecture finale, signée dans la documentation du projet).

    Clore la documentation d’architecture du projet : rassembler et mettre à jour tous les documents (schémas finaux, décisions prises, bonnes pratiques adoptées) pour qu’ils reflètent l’état réel du système livré (Livrable : pack de documentation d’architecture final archivé et partagé avec les parties prenantes et l’équipe de maintenance).

test-automator

    Effectuer une validation finale post-déploiement : exécuter une sélection de tests de régression sur l’environnement de production (ou staging final) afin de s’assurer que le déploiement s’est déroulé correctement et que l’application répond comme attendu aux fonctionnalités principales (Livrable : compte-rendu de validation finale en production, attestant que les fonctionnalités critiques ont été testées et sont opérationnelles).

    Être disponible pour tester rapidement tout correctif d’urgence post-livraison : si des bugs critiques sont remontés juste après la mise en production, écrire/adapter des tests pour reproduire ces bugs, puis valider les patchs déployés en production (Livrable : résultats de tests sur les correctifs post-production, confirmant la résolution des problèmes critiques éventuels).1