import DateUtils from "./utils/dateUtils.js";
import DomUtils from "./utils/domUtils.js";
import StorageUtils from "./utils/storageUtils.js";
import Popup from "./popup.js";
import { log } from "./utils/commonUtils.js";
let LogsDialog = (function () {
    let Id;
    (function (Id) {
        Id["LOGS_DIALOG_DIV"] = "logsDialog";
        Id["LOGS_DIV"] = "logs";
        Id["CLOSE_LOGS_BUTTON"] = "closeLogsButton_0";
        Id["EXPORT_LOGS_BUTTON"] = "exportLogsButton_0";
        Id["DELETE_LOGS_BUTTON"] = "deleteLogsButton_0";
    })(Id || (Id = {}));
    async function onShowDialogButtonClick() {
        await log(">> LogsDialog.onShowDialogButtonClick");
        let logs = await StorageUtils.getLogs();
        logs = logs.replace(/\n/g, "<br>");
        DomUtils.getById(Id.LOGS_DIV).innerHTML = logs;
        DomUtils.getById(Id.LOGS_DIALOG_DIV).style.display = 'block';
        await log("<< LogsDialog.onShowDialogButtonClick");
    }
    async function onCloseDialogButtonClick() {
        await log(">> LogsDialog.onCloseDialogButtonClick");
        DomUtils.getById(Id.LOGS_DIALOG_DIV).style.display = 'none';
        await log("<< LogsDialog.onCloseDialogButtonClick");
    }
    async function onDeleteLogsButtonClick() {
        await log(">> LogsDialog.onDeleteLogsButtonClick");
        await StorageUtils.removeLogs();
        DomUtils.getById(Id.LOGS_DIV).innerHTML = "";
        await log("<< LogsDialog.onDeleteLogsButtonClick");
    }
    async function onExportLogsButtonClick() {
        await log(">> LogsDialog.onExportLogsButtonClick");
        let logs = await StorageUtils.getLogs();
        let elt = document.createElement('a');
        elt.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(logs));
        elt.setAttribute('download', "logs" + DateUtils.dateToString(new Date()) + ".txt");
        elt.style.display = 'none';
        document.body.appendChild(elt);
        elt.click();
        document.body.removeChild(elt);
        await log("<< LogsDialog.onExportLogsButtonClick");
    }
    async function initListeners() {
        await log(">> LogsDialog.initListeners");
        DomUtils.getById(Id.CLOSE_LOGS_BUTTON).addEventListener("click", onCloseDialogButtonClick);
        DomUtils.getById(Id.DELETE_LOGS_BUTTON).addEventListener("click", onDeleteLogsButtonClick);
        DomUtils.getById(Id.EXPORT_LOGS_BUTTON).addEventListener("click", onExportLogsButtonClick);
        await log("<< LogsDialog.initListeners");
    }
    async function init() {
        await log(">> LogsDialog.init");
        await initListeners();
        await log("<< LogsDialog.init");
    }
    return {
        Id: Id,
        init: init,
        onShowDialogButtonClick: onShowDialogButtonClick
    };
})();
export default LogsDialog;
document.addEventListener("DOMContentLoaded", function () {
    let isLogsModalTestContext = !Popup.getChrome().permissions;
    if (!isLogsModalTestContext) {
        LogsDialog.init();
    }
});
