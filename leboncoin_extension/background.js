import SearchUtils from "./js/utils/searchUtils.js";
import StorageUtils from "./js/utils/storageUtils.js";
import DateUtils from "./js/utils/dateUtils.js";
import { getDisplayableAlarm, getNextSearchDate, log, isEmpty, MessageType, NB_SEARCH_TABS } from "./js/utils/commonUtils.js";
import { API_CONFIG } from "./config.js";

const FACEBOOK_SEARCH_URL = 'https://www.facebook.com/marketplace/105683362799117/search?sortBy=creation_time_descend\u0026query=vetement\u0026exact=false\u0026locale=fr_FR';

const Background = (function () {
    const MSG_NOTIFICATION = "Nouvelle offre pour la recherche ";
    let _mockChrome;

    function getChrome() {
        return _mockChrome === undefined ? chrome : _mockChrome;
    }

    function setMockChrome(backgroundmockChrome) {
        _mockChrome = backgroundmockChrome;
    }

    async function sendToAPI(endpoint, data) {
        const url = `${API_CONFIG.baseUrl}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Ajouter l'authentification si configurée
        if (API_CONFIG.auth.bearer) {
            headers['Authorization'] = `Bearer ${API_CONFIG.auth.bearer}`;
        } else if (API_CONFIG.auth.apiKey) {
            headers['X-API-Key'] = API_CONFIG.auth.apiKey;
        }

        try {
            let response = await fetch(url, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            let result = await response.json();
            console.log('API Response:', result);
            return result;
        } catch (error) {
            console.error('Error sending data to API:', error);
            throw error;
        }
    }

    async function checkStorageUsage() {
        chrome.storage.local.getBytesInUse(null, function(bytesInUse) {
            console.log(`Current storage usage: ${bytesInUse} bytes`);
        });
    }

    async function clearOldStorageData() {
        chrome.storage.local.get(null, function(items) {
            for (let key in items) {
                // Add your condition here to determine what to remove
                chrome.storage.local.remove(key, function() {
                    console.log(`Removed item with key: ${key}`);
                });
            }
        });
    }

    async function onSearchResults(tabIndex, searchName, searchResults) {
        await log(">> bg.onSearchResults: tabIndex=" + tabIndex + ", searchName=" + searchName + ", searchResults=" + JSON.stringify(searchResults));
        let isSearchOK = searchResults.searchStatus === SearchUtils.SearchStatus.OK;
        let isNewOfferAvailable = false;

        if (isSearchOK) {
            let lastOfferUrl = await StorageUtils.getLastOfferUrl(tabIndex);
            isNewOfferAvailable = (!lastOfferUrl) || (searchResults.offerUrl !== lastOfferUrl);
            await log("bg.onSearchResults: searchResults.offerUrl=" + searchResults.offerUrl + ", lastOfferUrl=" + lastOfferUrl + " -> isNewOfferAvailable=" + isNewOfferAvailable);

            if (isNewOfferAvailable) {
                await log("Nouvelle annonce détectée : " + JSON.stringify(searchResults));

                await StorageUtils.setHasUpdate(tabIndex, true);
                await StorageUtils.setLastUpdateDate(tabIndex, DateUtils.dateToString(new Date(), true));
                await notifyUser(searchName, searchResults);
            }

            await StorageUtils.setLastOfferUrl(tabIndex, searchResults.offerUrl);
            await StorageUtils.setLastOfferTitle(tabIndex, searchResults.offerTitle);
        }

        await StorageUtils.setLastSearchDate(tabIndex, DateUtils.dateToString(new Date(), true));
        await StorageUtils.setLastSearchStatus(tabIndex, searchResults.searchStatus);

        let alarm = await getChrome().alarms.get();
        let nextSearchDate = await getNextSearchDate(alarm);
        let msg = {
            type: MessageType.ON_SEARCH_RESULTS,
            tabIndex: tabIndex,
            nextSearchDate: nextSearchDate
        };
        await sendMessage(msg);
        await log("<< bg.onSearchResults");
    }

    async function fetchFacebookMarketplaceData() {
        const proxyUrl = 'http://localhost:3000/fetch';
        const targetUrl = encodeURIComponent(FACEBOOK_SEARCH_URL);
    
        try {
            const response = await fetch(`${proxyUrl}?url=${targetUrl}`);
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Fetch error:', error);
            throw error;
        }
    }

    async function onFacebookMarketplaceResults(searchResults) {
        await log(">> bg.onFacebookMarketplaceResults: searchResults=" + JSON.stringify(searchResults));
        let isSearchOK = searchResults.searchStatus === SearchUtils.SearchStatus.OK;
        let isNewOfferAvailable = false;

        if (isSearchOK && searchResults.offers.length > 0) {
            let lastOfferUrl = await StorageUtils.getLastOfferUrl('facebook');
            isNewOfferAvailable = searchResults.offers.some(offer => offer.url !== lastOfferUrl);
            await log("bg.onFacebookMarketplaceResults: offers[0].url=" + searchResults.offers[0].url + ", lastOfferUrl=" + lastOfferUrl + " -> isNewOfferAvailable=" + isNewOfferAvailable);

            if (isNewOfferAvailable) {
                await log("Nouvelle annonce Facebook détectée : " + JSON.stringify(searchResults.offers[0]));

                await StorageUtils.setHasUpdate('facebook', true);
                await StorageUtils.setLastUpdateDate('facebook', DateUtils.dateToString(new Date(), true));
                await notifyUser('Facebook Marketplace', searchResults.offers[0]);
            }

            await StorageUtils.setLastOfferUrl('facebook', searchResults.offers[0].url);
            await StorageUtils.setLastOfferTitle('facebook', searchResults.offers[0].title);
        }

        await StorageUtils.setLastSearchDate('facebook', DateUtils.dateToString(new Date(), true));
        await StorageUtils.setLastSearchStatus('facebook', searchResults.searchStatus);
        await log("<< bg.onFacebookMarketplaceResults");
    }

    async function sendMessage(msg) {
        let isPopupVisible = await StorageUtils.getIsPopupVisible();
        await log(">> bg.sendMessage: isPopupVisible=" + isPopupVisible + ", msg=" + JSON.stringify(msg));
        if (isPopupVisible) {
            getChrome().runtime.sendMessage(msg, async function(response) {
                if (chrome.runtime.lastError) {
                    console.error("Error sending message:", chrome.runtime.lastError.message);
                } else {
                    await log("bg.sendMessage: got response=" + response);
                }
            });
        }
        await log("<< bg.sendMessage");
    }

    async function notifyUser(searchName, searchResult) {
        await log(" bg.notifyUser: searchName=" + searchName);

        const offerTitle = searchResult.title || "Titre indisponible";
        const offerUrl = searchResult.url || "Lien indisponible";
        const offerPrice = searchResult.price || "Prix indisponible";
        const offerTitleWithoutPrice = offerTitle.replace(/\s*\([^)]*\)$/, '');

        // Préparer les données pour l'API
        const notificationData = {
            searchName: searchName,
            offer: {
                title: offerTitleWithoutPrice,
                price: offerPrice,
                url: offerUrl,
                originalTitle: offerTitle
            },
            timestamp: new Date().toISOString(),
            source: searchName.includes('Facebook') ? 'facebook' : 'leboncoin'
        };

        // Envoyer la notification à l'API
        try {
            await sendToAPI(API_CONFIG.endpoints.notifications, notificationData);
        } catch (error) {
            console.error('Failed to send notification to API:', error);
        }

        getChrome().action.setBadgeText({ text: "1" });
        getChrome().action.setBadgeBackgroundColor({ color: "#FF0000" });

        let areNotificationsEnabled = await StorageUtils.getAreNotificationsEnabled();
        if (areNotificationsEnabled) {
            let useTemporaryNotifications = await StorageUtils.getUseTemporaryNotifications();
            await log("bg.notifyUser: generating notification with useTemporaryNotifications=" + useTemporaryNotifications);

            getChrome().notifications.create({
                type: "basic",
                iconUrl: "/png/icon-leboncoin-32.png",
                title: getChrome().runtime.getManifest().name,
                message: `Nouvelle offre pour la recherche "${searchName}"`,
                requireInteraction: !useTemporaryNotifications
            }, async function (notificationId) {
                if (getChrome().runtime.lastError) {
                    await log("bg.notifyUser: lastError=" + getChrome().runtime.lastError);
                }
            });
        }

        await log("<< bg.notifyUser");
    }

    async function onAlarmListener(alarm) {
        await log(">> bg.onAlarmListener: alarm=" + getDisplayableAlarm(alarm));
        await search();
        await log("<< bg.onAlarmListener");
    }

    async function search(tabIndex) {
        await log(">> bg.search: tabIndex=" + tabIndex);
        if (tabIndex !== undefined) {
            await SearchUtils.search(tabIndex, await StorageUtils.getSearchName(tabIndex), await StorageUtils.getSearchUrl(tabIndex), onSearchResults);
        } else {
            for (let nextTabIndex = 1; nextTabIndex <= NB_SEARCH_TABS; nextTabIndex++) {
                let searchName = await StorageUtils.getSearchName(nextTabIndex);
                if (searchName) {
                    await SearchUtils.search(nextTabIndex, searchName, await StorageUtils.getSearchUrl(nextTabIndex), onSearchResults);
                }
            }
            // Ajouter la recherche Facebook Marketplace
            if (FACEBOOK_SEARCH_URL) {
                await SearchUtils.searchFacebookMarketplace(FACEBOOK_SEARCH_URL, onFacebookMarketplaceResults);
            }
        }
        await log("<< bg.search");
    }

    async function testFacebookMarketplaceSearch() {
        await log(">> bg.testFacebookMarketplaceSearch");
        try {
            const searchResults = await fetchFacebookMarketplaceData();
            await onFacebookMarketplaceResults(searchResults);
        } catch (error) {
            console.error('Error during Facebook Marketplace search:', error);
        }
        await log("<< bg.testFacebookMarketplaceSearch");
    }
    

    async function initAlarm(alwaysCreateNew) {
        let searchFrequency = 1;
        let alarm = await getChrome().alarms.get();
        await log(">> bg.initAlarm: alwaysCreateNew=" + alwaysCreateNew + ", searchFrequency=" + searchFrequency + ", alarm=" + getDisplayableAlarm(alarm));
        if (searchFrequency > 0) {
            if (!alarm || (alwaysCreateNew === true)) {
                await log("bg.initAlarm: creating new alarm with searchFrequency=" + searchFrequency);
                await getChrome().alarms.create({ delayInMinutes: searchFrequency, periodInMinutes: searchFrequency });
                let newAlarm = await getChrome().alarms.get();
                let msg = {
                    type: MessageType.ON_ALARM_CREATED,
                    nextSearchDate: await getNextSearchDate(newAlarm)
                };
                await sendMessage(msg);
            }
        }
        alarm = await getChrome().alarms.get();
        await log("<< bg.initAlarm: alarm=" + getDisplayableAlarm(alarm));
    }

    function onMessageListener(msg, sender, sendResponse) {
        onMessageListenerWrapper(msg).then(() => sendResponse("done"));
        return true;
    }

    async function onMessageListenerWrapper(msg) {
        await log(">> bg.onMessageListenerWrapper: msg=" + JSON.stringify(msg));
        if (msg.type === MessageType.ON_MANUAL_SEARCH) {
            await initAlarm();
            await search(msg.tabIndex);
        } else if (msg.type === MessageType.ON_OPTIONS_SAVED) {
            await initAlarm(true);
        }
        await log("<< bg.onMessageListenerWrapper");
    }

    async function onStateChanged(currentState) {
        let previousState = await StorageUtils.getIdleState();
        previousState = isEmpty(previousState) ? "active" : previousState;
        await log(">> bg.onStateChanged: previousState=" + previousState + ", currentState=" + currentState);
        if (previousState === "locked") {
            await initAlarm(true);
            await search();
        }
        await StorageUtils.setIdleState(currentState);
        await log("<< bg.onStateChanged");
    }

    function initListeners() {
        console.debug("DEBUG >> bg.initListeners");
        getChrome().alarms.onAlarm.addListener(onAlarmListener);
        getChrome().runtime.onMessage.addListener(onMessageListener);
        getChrome().runtime.onConnect.addListener(function (port) {
            if (port.name === "popup") {
                port.onDisconnect.addListener(async function () {
                    await log("bg.initListeners.port.onDisconnect: popup was closed");
                    await StorageUtils.setIsPopupVisible(false);
                });
            }
        });
        getChrome().idle.onStateChanged.addListener(onStateChanged);

        // Appel manuel pour tester la recherche Facebook Marketplace
        testFacebookMarketplaceSearch();

        console.debug("DEBUG << bg.initListeners");
    }

    async function init() {
        initListeners();
        await log(">> bg.init");
        await initAlarm();
        await log("<< bg.init");
        checkStorageUsage();
        clearOldStorageData();
    }

    return {
        getChrome: getChrome,
        setMockChrome: setMockChrome,
        onSearchResults: onSearchResults,
        onMessageListener: onMessageListener,
        init: init,
        testFacebookMarketplaceSearch: testFacebookMarketplaceSearch
    };
})();

export default Background;

chrome.runtime.onInstalled.addListener(() => {
    chrome.declarativeNetRequest.updateDynamicRules({
        addRules: [{
            "id": 1,
            "priority": 1,
            "action": {
                "type": "modifyHeaders",
                "requestHeaders": [
                    { "header": "Origin", "operation": "set", "value": "https://www.facebook.com" },
                    { "header": "Access-Control-Allow-Origin", "operation": "set", "value": "*" }
                ]
            },
            "condition": {
                "urlFilter": "https://www.facebook.com/marketplace/*",
                "resourceTypes": ["xmlhttprequest", "sub_frame"]
            }
        }],
        removeRuleIds: [1]
    }, function() {
        console.log("CORS bypass rule set for Facebook Marketplace.");
    });
});

let isBackgroundTestContext = !chrome.permissions;
if (!isBackgroundTestContext) {
    Background.init();
}
