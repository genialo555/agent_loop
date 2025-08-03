async function fetchMarketplaceData(url) {
  try {
    const response = await fetch(url, {
      method: 'GET',
      mode: 'cors', // This is important to handle CORS issues
      credentials: 'include'
    });
    const data = await response.json();
    console.log(data);
    // Traitez les données ici...
  } catch (error) {
    console.error('Error fetching data from Facebook Marketplace:', error);
  }
}

fetchMarketplaceData('https://www.facebook.com/marketplace/105683362799117/search?sortBy=creation_time_descend&query=vetement&exact=false&locale=fr_FR');
