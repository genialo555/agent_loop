import DateUtils from "./utils/dateUtils.js";
import StorageUtils from "./utils/storageUtils.js";
import SearchUtils from "./utils/searchUtils.js";
import LogsDialog from "./logsDialog.js";
import AboutDialog from "./aboutDialog.js";
import Config from "./config.js";
import { getNextSearchDate, log, isEmpty, MessageType, NB_SEARCH_TABS, getDisplayableAlarm } from "./utils/commonUtils.js";
import DomUtils from "./utils/domUtils.js";
import logsDialog from "./logsDialog.js";
import aboutDialog from "./aboutDialog.js";
let Popup = (function () {
    const SITE_URL_PREFIX = "https://www.leboncoin.fr/";
    let ErrorMsg;
    (function (ErrorMsg) {
        ErrorMsg["NAME_EMPTY"] = "Erreur: veuillez saisir un label";
        ErrorMsg["NAME_TOO_LONG"] = "Erreur: le label ne doit pas d&eacute;passer 12 caract&egrave;res";
        ErrorMsg["URL_EMPTY"] = "Erreur: veuillez saisir une adresse";
        ErrorMsg["URL_INVALID"] = "Erreur: l'adresse n'est pas valide";
    })(ErrorMsg || (ErrorMsg = {}));
    let IdPrefix;
    (function (IdPrefix) {
        IdPrefix["TAB"] = "search_";
        IdPrefix["TAB_HEADER"] = "tabheader_";
        IdPrefix["TAB_HEADER_TITLE"] = "tabheader_title_";
        IdPrefix["DEFAULT_TAB_HEADER_LABEL"] = "Recherche ";
        IdPrefix["TAB_HEADER_BADGE"] = "tabheader_badge_";
        IdPrefix["ERROR_DIALOG"] = "errorDialog_";
        IdPrefix["ERROR_DIALOG_BUTTON"] = "errorDialogButton_";
        IdPrefix["ERROR_MESSAGE"] = "errorMsg_";
        IdPrefix["OPEN_TUTORIAL_BUTTON"] = "openTutorialButton_";
        IdPrefix["SHOW_ABOUT_BUTTON"] = "showAboutButton_";
        IdPrefix["SHOW_LOGS_BUTTON"] = "showLogsButton_";
        IdPrefix["SEARCH_NAME_INPUT"] = "searchName_";
        IdPrefix["SEARCH_URL_INPUT"] = "searchUrl_";
        IdPrefix["SEARCH_URL_OPEN_BUTTON"] = "searchUrlOpenButton_";
        IdPrefix["SEARCH_NOW_BUTTON"] = "searchNowButton_";
        IdPrefix["SEARCH_DELETE_BUTTON"] = "searchDeleteButton_";
        IdPrefix["TEST_BUTTON"] = "testButton_";
        IdPrefix["LAST_SEARCH_DATE_INPUT"] = "lastSearchDate_";
        IdPrefix["LAST_OFFER_TITLE_INPUT"] = "lastOfferTitle_";
        IdPrefix["LAST_OFFER_OPEN_BUTTON"] = "lastOfferOpenButton_";
        IdPrefix["NEXT_SEARCH_DATE_INPUT"] = "nextSearchDate_";
        IdPrefix["LAST_UPDATE_DATE_INPUT"] = "lastUpdateDate_";
    })(IdPrefix || (IdPrefix = {}));
    let g_currentTabIndex = 0;
    let _mockChrome;
    function getChrome() {
        return _mockChrome === undefined ? chrome : _mockChrome;
    }
    function setMockChrome(popupMockChrome) {
        _mockChrome = popupMockChrome;
    }
    function getTabIndex(id) {
        return parseInt(id.substr(id.length - 1));
    }
    function toggleMouseCursor(isHourglass, tabIndex) {
        let cursor = isHourglass ? "wait" : "default";
        document.body.style.cursor = cursor;
        if (tabIndex !== undefined) {
            DomUtils.getById(IdPrefix.SEARCH_NOW_BUTTON + tabIndex).style.cursor = cursor;
        }
    }
    async function onTabHeaderClick(evt) {
        let elt = evt.currentTarget;
        let clickedTabIndex = getTabIndex(elt.id);
        let tabElts = DomUtils.getByClassName("tabs");
        let tablinkElts = DomUtils.getByClassName("tablink");
        for (let i = 0; i < tabElts.length; i++) {
            DomUtils.hide(tabElts[i].id);
            tablinkElts[i].classList.remove("w3-border-color");
        }
        getChrome().action.setBadgeText({ text: "" });
        DomUtils.show(IdPrefix.TAB + clickedTabIndex);
        elt.firstElementChild.classList.add("w3-border-color");
        if (clickedTabIndex != g_currentTabIndex) {
            DomUtils.hide(IdPrefix.TAB_HEADER_BADGE + g_currentTabIndex);
        }
        g_currentTabIndex = clickedTabIndex;
        DomUtils.hide(IdPrefix.TAB_HEADER_BADGE + clickedTabIndex);
        let searchName = await StorageUtils.getSearchName(clickedTabIndex);
        if (!isEmpty(searchName)) {
            await StorageUtils.setHasUpdate(clickedTabIndex, false);
        }
    }
    async function onOpenTutorialButtonClick() {
        await log(">> Popup.onOpenTutorialButtonClick");
        window.open("https://youtu.be/T629XnAC974");
        await log("<< Popup.onOpenTutorialButtonClick");
    }
    async function onTestButtonClick() {
        await StorageUtils.setLogs("hello");
        await StorageUtils.setLogs("world");
    }
    async function showErrorDialog(tabIndex, errorMsg) {
        await log(">> Popup.showErrorDialog: tabIndex=" + tabIndex);
        DomUtils.getById(IdPrefix.ERROR_MESSAGE + tabIndex).innerHTML = errorMsg;
        DomUtils.show(IdPrefix.ERROR_DIALOG + tabIndex);
        await log("<< Popup.showErrorDialog");
    }
    async function onCloseErrorDialogButtonClick(event) {
        let tabIndex = getTabIndex(event.currentTarget.id);
        await log(">> Popup.onCloseErrorDialogButtonClick: tabIndex=" + tabIndex);
        DomUtils.hide(IdPrefix.ERROR_DIALOG + tabIndex);
        await log("<< Popup.onCloseErrorDialogButtonClick");
    }
    async function onSearchNowButtonClick(event) {
        let tabIndex = getTabIndex(event.currentTarget.id);
        await log(">> Popup.onSearchNowButtonClick: tabIndex=" + tabIndex);
        let result = await validateAndSaveSearch(tabIndex);
        if (result !== -1) {
            toggleMouseCursor(true, tabIndex);
            let msg = { type: MessageType.ON_MANUAL_SEARCH, tabIndex: tabIndex };
            await log("Popup.onSearchNowButtonClick: sending message=" + JSON.stringify(msg));
            getChrome().runtime.sendMessage(msg, async function (response) {
                await log("Popup.onSearchNowButtonClick.sendMessage: got response=" + response);
                if (chrome.runtime.lastError) {
                    await log("Popup.onSearchNowButtonClick.sendMessage: got error=" + JSON.stringify(chrome.runtime.lastError), true);
                }
            });
        }
        await log("<< Popup.onSearchNowButtonClick");
    }
    async function onSearchDeleteButtonClick(event) {
        let tabIndex = getTabIndex(event.currentTarget.id);
        await log(">> Popup.onSearchDeleteButtonClick: tabIndex=" + tabIndex);
        await StorageUtils.removeSearch(tabIndex);
        await initTabFromStorage(tabIndex);
        await log("<< Popup.onSearchDeleteButtonClick");
    }
    async function validateAndSaveSearch(tabIndex) {
        let searchName = DomUtils.getById(IdPrefix.SEARCH_NAME_INPUT + tabIndex).value;
        let searchUrl = DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex).value;
        await log(">> Popup.validateAndSaveSearch: tabIndex=" + tabIndex + ", searchName=" + searchName + ", searchUrl=" + searchUrl);
        if (searchName.length === 0) {
            DomUtils.getById(IdPrefix.SEARCH_NAME_INPUT + tabIndex).classList.add("w3-border-red");
            await showErrorDialog(tabIndex, ErrorMsg.NAME_EMPTY);
            await log("<< Popup.validateAndSaveSearch: error=" + ErrorMsg.NAME_EMPTY);
            return -1;
        }
        if (searchName.length > 12) {
            DomUtils.getById(IdPrefix.SEARCH_NAME_INPUT + tabIndex).classList.add("w3-border-red");
            await showErrorDialog(tabIndex, ErrorMsg.NAME_TOO_LONG);
            await log("<< Popup.validateAndSaveSearch: error=" + ErrorMsg.NAME_TOO_LONG);
            return -1;
        }
        if (searchUrl.length === 0) {
            DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex).classList.add("w3-border-red");
            await showErrorDialog(tabIndex, ErrorMsg.URL_EMPTY);
            await log("<< Popup.validateAndSaveSearch: error=" + ErrorMsg.URL_EMPTY);
            return -1;
        }
        let isUrlValid = searchUrl.startsWith(SITE_URL_PREFIX);
        if (!isUrlValid) {
            DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex).classList.add("w3-border-red");
            await showErrorDialog(tabIndex, ErrorMsg.URL_INVALID);
            await log("<< Popup.validateAndSaveSearch: error=" + ErrorMsg.URL_INVALID);
            return -1;
        }
        DomUtils.getById(IdPrefix.SEARCH_NAME_INPUT + tabIndex).classList.remove("w3-border-red");
        DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex).classList.remove("w3-border-red");
        DomUtils.getById(IdPrefix.ERROR_DIALOG_BUTTON + tabIndex).click();
        DomUtils.getById(IdPrefix.TAB_HEADER_TITLE + tabIndex).innerHTML = searchName;
        await StorageUtils.setSearchName(tabIndex, searchName);
        await StorageUtils.setSearchUrl(tabIndex, searchUrl);
        await log("<< Popup.validateAndSaveSearch: OK");
        return 0;
    }
    function onMessageListener(msg, sender, sendResponse) {
        onMessageListenerWrapper(msg).then(() => sendResponse("done"));
        return true;
    }
    async function onMessageListenerWrapper(msg) {
        let alarm = await getChrome().alarms.get();
        await log(">> Popup.onMessageListener: msg=" + JSON.stringify(msg) + ", alarm=" + getDisplayableAlarm(alarm));
        if (msg.type === MessageType.ON_SEARCH_RESULTS) {
            toggleMouseCursor(false, msg.tabIndex);
            await initTabFromStorage(msg.tabIndex);
            if (alarm) {
                DomUtils.getById(IdPrefix.NEXT_SEARCH_DATE_INPUT + msg.tabIndex).value = await getNextSearchDate(alarm);
            }
        }
        else if (msg.type === MessageType.ON_ALARM_CREATED) {
            for (let tabIndex = 1; tabIndex <= NB_SEARCH_TABS; tabIndex++) {
                let searchName = await StorageUtils.getSearchName(tabIndex);
                if (alarm && !isEmpty(searchName)) {
                    DomUtils.getById(IdPrefix.NEXT_SEARCH_DATE_INPUT + tabIndex).value = await getNextSearchDate(alarm);
                }
            }
        }
        else if (msg.type === MessageType.ON_BACKGROUND_ERROR) {
            toggleMouseCursor(false);
        }
        await log("<< Popup.onMessageListener");
    }
    async function initTabFromStorage(tabIndex) {
        await log(">> Popup.initTabFromStorage: tabIndex=" + tabIndex);
        let lastSearchStatus = await StorageUtils.getLastSearchStatus(tabIndex);
        let isLastSearchOk = lastSearchStatus === SearchUtils.SearchStatus.OK;
        let tabHeaderId = IdPrefix.TAB_HEADER_BADGE + tabIndex;
        let hasUpdate = await StorageUtils.getHasUpdate(tabIndex);
        if (hasUpdate)
            DomUtils.show(tabHeaderId, "inline");
        else
            DomUtils.hide(tabHeaderId);
        let searchName = await StorageUtils.getSearchName(tabIndex);
        let tabHeader = isEmpty(searchName) ? IdPrefix.DEFAULT_TAB_HEADER_LABEL + tabIndex : searchName;
        DomUtils.getById(IdPrefix.TAB_HEADER_TITLE + tabIndex).innerHTML = tabHeader;
        DomUtils.getById(IdPrefix.SEARCH_NAME_INPUT + tabIndex).value = isEmpty(searchName) ? "" : searchName;
        let searchUrl = await StorageUtils.getSearchUrl(tabIndex);
        DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex).value = isEmpty(searchUrl) ? "" : searchUrl;
        let lastSearchDate = await StorageUtils.getLastSearchDate(tabIndex);
        lastSearchDate = isEmpty(lastSearchDate) ? "" : lastSearchDate;
        let displLastSearchDate = lastSearchDate.length > 0 ? DateUtils.absoluteDateToRelativeDate(lastSearchDate) : lastSearchDate;
        DomUtils.getById(IdPrefix.LAST_SEARCH_DATE_INPUT + tabIndex).value = displLastSearchDate;
        let lastOfferTitle = isLastSearchOk ? await StorageUtils.getLastOfferTitle(tabIndex) : lastSearchStatus;
        DomUtils.getById(IdPrefix.LAST_OFFER_TITLE_INPUT + tabIndex).value = isEmpty(lastOfferTitle) ? "" : lastOfferTitle;
        let lastUpdateDate = isLastSearchOk ? await StorageUtils.getLastUpdateDate(tabIndex) : "";
        lastUpdateDate = isEmpty(lastUpdateDate) ? "" : lastUpdateDate;
        let displLastUpdateDate = lastUpdateDate.length > 0 ? DateUtils.absoluteDateToRelativeDate(lastUpdateDate) : lastUpdateDate;
        DomUtils.getById(IdPrefix.LAST_UPDATE_DATE_INPUT + tabIndex).value = displLastUpdateDate;
        DomUtils.getById(IdPrefix.NEXT_SEARCH_DATE_INPUT + tabIndex).value = "";
        await log("<< Popup.initTabFromStorage");
    }
    async function initListeners() {
        await log(">> Popup.initListeners");
        getChrome().runtime.onMessage.addListener(onMessageListener);
        getChrome().runtime.connect({ name: "popup" });
        await StorageUtils.setIsPopupVisible(true);
        DomUtils.getById(IdPrefix.TAB_HEADER + 0).addEventListener("click", onTabHeaderClick);
        DomUtils.getById(IdPrefix.OPEN_TUTORIAL_BUTTON + 0).addEventListener("click", onOpenTutorialButtonClick);
        DomUtils.getById(IdPrefix.SHOW_ABOUT_BUTTON + 0).addEventListener("click", aboutDialog.onShowDialogButtonClick);
        DomUtils.getById(IdPrefix.SHOW_LOGS_BUTTON + 0).addEventListener("click", logsDialog.onShowDialogButtonClick);
        DomUtils.getById(IdPrefix.TAB_HEADER + 0).click();
        for (let tabIndex = 0; tabIndex <= NB_SEARCH_TABS; tabIndex++) {
            DomUtils.getById(IdPrefix.ERROR_DIALOG_BUTTON + tabIndex).addEventListener("click", onCloseErrorDialogButtonClick);
            DomUtils.getById(IdPrefix.TEST_BUTTON + tabIndex).addEventListener("click", onTestButtonClick);
            if (tabIndex > 0) {
                DomUtils.getById(IdPrefix.TAB_HEADER + tabIndex).addEventListener("click", onTabHeaderClick);
                DomUtils.getById(IdPrefix.SEARCH_URL_OPEN_BUTTON + tabIndex).addEventListener("click", function () {
                    let elt = DomUtils.getById(IdPrefix.SEARCH_URL_INPUT + tabIndex);
                    window.open(elt.value);
                });
                DomUtils.getById(IdPrefix.SEARCH_NOW_BUTTON + tabIndex).addEventListener("click", onSearchNowButtonClick);
                DomUtils.getById(IdPrefix.SEARCH_DELETE_BUTTON + tabIndex).addEventListener("click", onSearchDeleteButtonClick);
                DomUtils.getById(IdPrefix.LAST_OFFER_OPEN_BUTTON + tabIndex).addEventListener("click", async function () {
                    let url = await StorageUtils.getLastOfferUrl(tabIndex);
                    if (!isEmpty(url)) {
                        window.open(url);
                    }
                });
            }
        }
        await log("<< Popup.initListeners");
    }
    async function initHTML() {
        await log(">> Popup.initHTML");
        let manifest = getChrome().runtime.getManifest();
        DomUtils.getById("extension_name_0").innerHTML = manifest.name;
        DomUtils.getById("aboutDialogTitle").innerHTML = `A propos de la version ${manifest.version}`;
        let tab1Html = DomUtils.getById(IdPrefix.TAB + "1").innerHTML;
        for (let i = 2; i <= NB_SEARCH_TABS; i++) {
            DomUtils.getById(IdPrefix.TAB + i).innerHTML = tab1Html.replace(/_1/g, "_" + i);
        }
        let alarm = await getChrome().alarms.get();
        await log("Popup.initHTML: alarm=" + getDisplayableAlarm(alarm));
        for (let tabIndex = 1; tabIndex <= NB_SEARCH_TABS; tabIndex++) {
            await initTabFromStorage(tabIndex);
            let searchName = await StorageUtils.getSearchName(tabIndex);
            if (alarm && !isEmpty(searchName)) {
                DomUtils.getById(IdPrefix.NEXT_SEARCH_DATE_INPUT + tabIndex).value = await getNextSearchDate(alarm);
            }
        }
        await log("<< Popup.initHTML");
    }
    async function init() {
        await log(">> Popup.init");
        window.onerror = function (msg, url, line) {
            log("Popup.onerror: msg=[" + msg + "], url=]" + url + "], line=[" + line + "]", true);
            toggleMouseCursor(false);
        };
        await initHTML();
        await initListeners();
        await Config.init();
        await LogsDialog.init();
        await AboutDialog.init();
        await log("<< Popup.init");
    }
    return {
        IdPrefix: IdPrefix,
        getChrome: getChrome,
        setMockChrome: setMockChrome,
        showErrorDialog: showErrorDialog,
        onMessageListener: onMessageListener,
        init: init
    };
})();
export default Popup;
document.addEventListener("DOMContentLoaded", async function () {
    let isPopupTestContext = !Popup.getChrome().permissions;
    if (!isPopupTestContext) {
        await Popup.init();
    }
});
