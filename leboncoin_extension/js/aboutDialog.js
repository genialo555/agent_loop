import DomUtils from "./utils/domUtils.js";
import Popup from "./popup.js";
import { log } from "./utils/commonUtils.js";
import StorageUtils from "./utils/storageUtils.js";
let AboutDialog = (function () {
    let Id;
    (function (Id) {
        Id["ABOUT_DIALOG_DIV"] = "aboutDialog";
        Id["CLOSE_ABOUT_BUTTON1"] = "closeAboutButton1_0";
        Id["CLOSE_ABOUT_BUTTON2"] = "closeAboutButton2_0";
    })(Id || (Id = {}));
    async function onShowDialogButtonClick() {
        await log(">> AboutDialog.onShowDialogButtonClick");
        DomUtils.getById(Id.ABOUT_DIALOG_DIV).style.display = 'block';
        await log("<< AboutDialog.onShowDialogButtonClick");
    }
    async function onCloseDialogButtonClick() {
        await log(">> AboutDialog.onCloseDialogButtonClick");
        DomUtils.getById(Id.ABOUT_DIALOG_DIV).style.display = 'none';
        let currentVersion = chrome.runtime.getManifest().version;
        await StorageUtils.setVersion(currentVersion);
        await log("<< AboutDialog.onCloseDialogButtonClick");
    }
    async function initListeners() {
        await log(">> AboutDialog.initListeners");
        DomUtils.getById(Id.CLOSE_ABOUT_BUTTON1).addEventListener("click", onCloseDialogButtonClick);
        DomUtils.getById(Id.CLOSE_ABOUT_BUTTON2).addEventListener("click", onCloseDialogButtonClick);
        await log("<< AboutDialog.initListeners");
    }
    async function init() {
        await log(">> AboutDialog.init");
        await initListeners();
        await log("<< AboutDialog.init");
    }
    return {
        Id: Id,
        init: init,
        onShowDialogButtonClick: onShowDialogButtonClick
    };
})();
export default AboutDialog;
document.addEventListener("DOMContentLoaded", function () {
    let isTestContext = !Popup.getChrome().permissions;
    if (!isTestContext) {
        AboutDialog.init();
    }
});
