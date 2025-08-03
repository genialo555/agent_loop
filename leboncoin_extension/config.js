// Configuration de l'API
export const API_CONFIG = {
    baseUrl: 'http://localhost:8000/api', // URL de base de votre API
    endpoints: {
        notifications: '/notifications', // Endpoint pour envoyer les notifications
        search: '/search', // Endpoint pour stocker les résultats de recherche
        status: '/status' // Endpoint pour vérifier le statut de l'API
    },
    auth: {
        // Si votre API nécessite une authentification
        bearer: null, // Sera rempli avec le token JWT si nécessaire
        apiKey: null // Ou une clé API si vous préférez
    }
};
