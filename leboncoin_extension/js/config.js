import StorageUtils from "./utils/storageUtils.js";
import { log, MessageType } from "./utils/commonUtils.js";
import Popup from "./popup.js";
import DomUtils from "./utils/domUtils.js";
let Config = (function () {
    let Id;
    (function (Id) {
        Id["SEARCH_FREQUENCY"] = "searchFrequency_0";
        Id["SHOW_NOTIFICATIONS"] = "showNotifications_0";
        Id["HIDE_NOTIFICATIONS"] = "hideNotifications_0";
        Id["TEMPORARY_NOTIFICATIONS"] = "temporaryNotifications_0";
        Id["PERMANENT_NOTIFICATIONS"] = "permanentNotifications_0";
        Id["SAVE_BUTTON"] = "saveButton_0";
        Id["STATUS"] = "status_0";
    })(Id || (Id = {}));
    let ErrorMsg;
    (function (ErrorMsg) {
        ErrorMsg["ERROR_SEARCH_FREQUENCY"] = "Erreur: veuillez saisir une valeur sup&eacute;rieure &agrave; 1";
        ErrorMsg["OPTIONS_SAVED"] = "Options sauvegard&eacute;es !";
    })(ErrorMsg || (ErrorMsg = {}));
    let _mockChrome;
    function getChrome() {
        return _mockChrome === undefined ? chrome : _mockChrome;
    }
    function setMockChrome(optionsMockChrome) {
        _mockChrome = optionsMockChrome;
    }
    async function onSaveOptionsButtonClick() {
        let searchFrequencyAsString = DomUtils.getById(Id.SEARCH_FREQUENCY).value;
        let searchFrequency = searchFrequencyAsString.length !== 0 ? parseInt(searchFrequencyAsString) : 0;
        let showNotifications = DomUtils.getById(Id.SHOW_NOTIFICATIONS).checked;
        let useTemporaryNotifications = DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).checked;
        await log(">> Config.onSaveOptionsButtonClick: searchFrequency=" + searchFrequency + ", showNotifications=" + showNotifications + ", tempNotifications=" + useTemporaryNotifications);
        if (searchFrequency < 1) {
            DomUtils.getById(Id.SEARCH_FREQUENCY).classList.add("w3-border-red");
            await Popup.showErrorDialog(0, ErrorMsg.ERROR_SEARCH_FREQUENCY);
            await log("<< Config.onSaveOptionsButtonClick: error=" + ErrorMsg.ERROR_SEARCH_FREQUENCY);
            return;
        }
        DomUtils.getById(Id.SEARCH_FREQUENCY).classList.remove("w3-border-red");
        DomUtils.getById(Popup.IdPrefix.ERROR_DIALOG_BUTTON + 0).click();
        await StorageUtils.setSearchFrequency(searchFrequency);
        await StorageUtils.setAreNotificationsEnabled(showNotifications);
        await StorageUtils.setUseTemporaryNotifications(useTemporaryNotifications);
        let msg = { type: MessageType.ON_OPTIONS_SAVED };
        getChrome().runtime.sendMessage(msg);
        let status = DomUtils.getById(Id.STATUS);
        status.innerHTML = ErrorMsg.OPTIONS_SAVED;
        setTimeout(function () {
            status.innerHTML = '';
        }, 1000);
        await log("<< Config.onSaveOptionsButtonClick");
    }
    async function initListeners() {
        await log(">> Config.initListeners");
        DomUtils.getById(Id.SAVE_BUTTON).addEventListener('click', onSaveOptionsButtonClick);
        DomUtils.getById(Id.SHOW_NOTIFICATIONS).addEventListener('click', function () {
            DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).disabled = false;
            DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).checked = true;
            DomUtils.getById(Id.PERMANENT_NOTIFICATIONS).disabled = false;
        });
        DomUtils.getById(Id.HIDE_NOTIFICATIONS).addEventListener('click', function () {
            DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).disabled = true;
            DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).checked = false;
            DomUtils.getById(Id.PERMANENT_NOTIFICATIONS).disabled = true;
            DomUtils.getById(Id.PERMANENT_NOTIFICATIONS).checked = false;
        });
        await log("<< Config.initListeners");
    }
    async function initHTML() {
        await log(">> Config.initHTML");
        let hasNotifications = await StorageUtils.getAreNotificationsEnabled();
        if (hasNotifications) {
            DomUtils.getById(Id.SHOW_NOTIFICATIONS).checked = true;
            let useTempNotifications = await StorageUtils.getUseTemporaryNotifications();
            if (useTempNotifications) {
                DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).checked = true;
            }
            else {
                DomUtils.getById(Id.PERMANENT_NOTIFICATIONS).checked = true;
            }
        }
        else {
            DomUtils.getById(Id.HIDE_NOTIFICATIONS).checked = true;
            DomUtils.getById(Id.TEMPORARY_NOTIFICATIONS).disabled = true;
            DomUtils.getById(Id.PERMANENT_NOTIFICATIONS).disabled = true;
        }
        DomUtils.getById(Id.SEARCH_FREQUENCY).value = (await StorageUtils.getSearchFrequency()).toString();
        DomUtils.getById(Id.SAVE_BUTTON).addEventListener('click', onSaveOptionsButtonClick);
        await log("<< Config.initHTML");
    }
    async function init() {
        await log(">> Config.init");
        await initHTML();
        await initListeners();
        await log("<< Config.init");
    }
    return {
        Id: Id,
        getChrome: getChrome,
        setMockChrome: setMockChrome,
        init: init
    };
})();
export default Config;
document.addEventListener("DOMContentLoaded", function () {
    let isOptionsTestContext = !Config.getChrome().permissions;
    if (!isOptionsTestContext) {
        Config.init();
    }
});
