import { isEmpty } from "./commonUtils.js";

const CHAT_IDS_KEY = 'chatIds';
const FACEBOOK_SEARCH_URL_KEY = 'facebook_search_url';

async function addChatId(chatId) {
    let chatIds = await getChatIds() || [];
    if (!chatIds.includes(chatId)) {
        chatIds.push(chatId);
        await chrome.storage.local.set({ [CHAT_IDS_KEY]: chatIds });
    }
}

async function getChatIds() {
    let result = await chrome.storage.local.get(CHAT_IDS_KEY);
    return result[CHAT_IDS_KEY] || [];
}

async function getFacebookSearchUrl() {
    let result = await chrome.storage.local.get(FACEBOOK_SEARCH_URL_KEY);
    return result[FACEBOOK_SEARCH_URL_KEY] || '';
}

async function setFacebookSearchUrl(url) {
    await chrome.storage.local.set({ [FACEBOOK_SEARCH_URL_KEY]: url });
}

let StorageUtils = (function () {
    const _SEARCH_PREFIX = "search_";
    let StorageKey;
    (function (StorageKey) {
        StorageKey["LOGS"] = "logs";
        StorageKey["OPTIONS"] = "options";
        StorageKey["SEARCH_FREQUENCY"] = "searchFrequency";
        StorageKey["ARE_NOTIFICATIONS_ENABLED"] = "areNotificationsEnabled";
        StorageKey["USE_TEMPORARY_NOTIFICATIONS"] = "useTemporaryNotifications";
        StorageKey["SET_IDLE_STATE"] = "idleState";
        StorageKey["IS_POPUP_VISIBLE"] = "isPopupVisible";
        StorageKey["VERSION"] = "lastVersion";
        StorageKey["SEARCH_NAME"] = "searchName";
        StorageKey["SEARCH_URL"] = "searchUrl";
        StorageKey["LAST_OFFER_TITLE"] = "lastOfferTitle";
        StorageKey["LAST_OFFER_URL"] = "lastOfferUrl";
        StorageKey["LAST_SEARCH_STATUS"] = "lastSearchStatus";
        StorageKey["LAST_SEARCH_DATE"] = "lastSearchDate";
        StorageKey["LAST_UPDATE_DATE"] = "lastUpdateDate";
        StorageKey["HAS_UPDATE"] = "hasUpdate";
    })(StorageKey || (StorageKey = {}));
    async function _removeValue(key) {
        chrome.storage.local.remove(key);
    }
    async function _getValue(key1, key2) {
        let result;
        let config = await chrome.storage.local.get(key1);
        let isEmpty = Object.keys(config).length === 0;
        if (!isEmpty) {
            result = config[key1];
            if (key2 !== null)
                result = result[key2];
        }
        else {
            result = null;
        }
        return result;
    }
    async function _setValue(key1, key2, value) {
        let config = await chrome.storage.local.get(key1);
        let isEmpty = Object.keys(config).length === 0;
        if (!isEmpty) {
            config = config[key1];
            if (key2 != null)
                config[key2] = value;
            else
                config = value;
        }
        else {
            if (key2 != null)
                config[key2] = value;
            else
                config = value;
            if (key2 == null)
                config = value;
        }
        await chrome.storage.local.set({ [key1]: config });
    }
    async function getLogs() {
        return await _getValue(StorageKey.LOGS, null);
    }
    async function setLogs(value) {
        return await _setValue(StorageKey.LOGS, null, value);
    }
    async function removeLogs() {
        await _removeValue(StorageKey.LOGS);
    }
    async function getVersion() {
        return await _getValue(StorageKey.VERSION, null);
    }
    async function setVersion(version) {
        await _setValue(StorageKey.VERSION, null, version);
    }
    async function getSearchFrequency() {
        let result = await _getValue(StorageKey.OPTIONS, StorageKey.SEARCH_FREQUENCY);
        return isEmpty(result) ? 60 : result;
    }
    async function setSearchFrequency(configValue) {
        await _setValue(StorageKey.OPTIONS, StorageKey.SEARCH_FREQUENCY, configValue);
    }
    async function getAreNotificationsEnabled() {
        let result = await _getValue(StorageKey.OPTIONS, StorageKey.ARE_NOTIFICATIONS_ENABLED);
        return isEmpty(result) ? true : result;
    }
    async function setAreNotificationsEnabled(configValue) {
        await _setValue(StorageKey.OPTIONS, StorageKey.ARE_NOTIFICATIONS_ENABLED, configValue);
    }
    async function getUseTemporaryNotifications() {
        let result = await _getValue(StorageKey.OPTIONS, StorageKey.USE_TEMPORARY_NOTIFICATIONS);
        return isEmpty(result) ? true : result;
    }
    async function setUseTemporaryNotifications(configValue) {
        await _setValue(StorageKey.OPTIONS, StorageKey.USE_TEMPORARY_NOTIFICATIONS, configValue);
    }
    async function getIdleState() {
        return await _getValue(StorageKey.OPTIONS, StorageKey.SET_IDLE_STATE);
    }
    async function setIdleState(configValue) {
        await _setValue(StorageKey.OPTIONS, StorageKey.SET_IDLE_STATE, configValue);
    }
    async function getIsPopupVisible() {
        let result = await _getValue(StorageKey.OPTIONS, StorageKey.IS_POPUP_VISIBLE);
        return isEmpty(result) ? false : result;
    }
    async function setIsPopupVisible(isPopupVisible) {
        await _setValue(StorageKey.OPTIONS, StorageKey.IS_POPUP_VISIBLE, isPopupVisible);
    }
    async function removeSearch(tabIndex) {
        await _removeValue(_SEARCH_PREFIX + tabIndex);
    }
    async function getSearchName(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.SEARCH_NAME);
    }
    async function setSearchName(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.SEARCH_NAME, configValue);
    }
    async function getSearchUrl(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.SEARCH_URL);
    }
    async function setSearchUrl(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.SEARCH_URL, configValue);
    }
    async function getLastSearchDate(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_SEARCH_DATE);
    }
    async function setLastSearchDate(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_SEARCH_DATE, configValue);
    }
    async function getLastSearchStatus(tabIndex) {
        return _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_SEARCH_STATUS);
    }
    async function setLastSearchStatus(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_SEARCH_STATUS, configValue);
    }
    async function getLastOfferUrl(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_OFFER_URL);
    }
    async function setLastOfferUrl(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_OFFER_URL, configValue);
    }
    async function getLastOfferTitle(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_OFFER_TITLE);
    }
    async function setLastOfferTitle(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_OFFER_TITLE, configValue);
    }
    async function getLastUpdateDate(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_UPDATE_DATE);
    }
    async function setLastUpdateDate(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.LAST_UPDATE_DATE, configValue);
    }
    async function getHasUpdate(tabIndex) {
        return await _getValue(_SEARCH_PREFIX + tabIndex, StorageKey.HAS_UPDATE);
    }
    async function setHasUpdate(tabIndex, configValue) {
        await _setValue(_SEARCH_PREFIX + tabIndex, StorageKey.HAS_UPDATE, configValue);
    }
    return {
        getLogs: getLogs,
        setLogs: setLogs,
        removeLogs: removeLogs,
        getSearchFrequency: getSearchFrequency,
        setSearchFrequency: setSearchFrequency,
        getAreNotificationsEnabled: getAreNotificationsEnabled,
        setAreNotificationsEnabled: setAreNotificationsEnabled,
        getUseTemporaryNotifications: getUseTemporaryNotifications,
        setUseTemporaryNotifications: setUseTemporaryNotifications,
        getIdleState: getIdleState,
        setIdleState: setIdleState,
        getIsPopupVisible: getIsPopupVisible,
        setIsPopupVisible: setIsPopupVisible,
        getVersion: getVersion,
        setVersion: setVersion,
        removeSearch: removeSearch,
        getSearchName: getSearchName,
        setSearchName: setSearchName,
        getSearchUrl: getSearchUrl,
        setSearchUrl: setSearchUrl,
        getLastOfferUrl: getLastOfferUrl,
        setLastOfferUrl: setLastOfferUrl,
        getLastSearchDate: getLastSearchDate,
        setLastSearchDate: setLastSearchDate,
        getLastOfferTitle: getLastOfferTitle,
        setLastOfferTitle: setLastOfferTitle,
        getLastUpdateDate: getLastUpdateDate,
        setLastUpdateDate: setLastUpdateDate,
        getLastSearchStatus: getLastSearchStatus,
        setLastSearchStatus: setLastSearchStatus,
        getHasUpdate: getHasUpdate,
        setHasUpdate: setHasUpdate,
        getFacebookSearchUrl: getFacebookSearchUrl,
        setFacebookSearchUrl: setFacebookSearchUrl
    };
})();
export default StorageUtils;
