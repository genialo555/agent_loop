import StorageUtils from "./storageUtils.js";
import DateUtils from "./dateUtils.js";
export const NB_SEARCH_TABS = 5;
const MAX_LOG_LENGTH = 25000;
export var MessageType;
(function (MessageType) {
    MessageType["ON_MANUAL_SEARCH"] = "ON_MANUAL_SEARCH";
    MessageType["ON_OPTIONS_SAVED"] = "ON_OPTIONS_SAVED";
    MessageType["ON_SEARCH_RESULTS"] = "ON_SEARCH_RESULTS";
    MessageType["ON_ALARM_CREATED"] = "ON_ALARM_CREATED";
    MessageType["ON_SHOW_BADGE"] = "ON_SHOW_BADGE";
    MessageType["ON_BACKGROUND_ERROR"] = "ON_BACKGROUND_ERROR";
})(MessageType || (MessageType = {}));
export function isEmpty(variable) {
    return variable === undefined || variable === null || ("" === variable);
}
export async function log(log, isError = false) {
    log = DateUtils.getDisplayableHoursMinutesSeconds(new Date()) + " " + (isError ? "ERROR:" : "") + log;
    if (isError) {
        console.error(log);
    }
    else {
        console.log(log);
    }
    let logs = await StorageUtils.getLogs();
    if (!logs)
        logs = "";
    if (logs.length > MAX_LOG_LENGTH) {
        await StorageUtils.removeLogs();
        logs = "";
    }
    logs = (log + "\n") + logs;
    await StorageUtils.setLogs(logs);
}
export function getDisplayableAlarm(alarm) {
    let result = "alarm {";
    if (alarm) {
        let scheduledDate = new Date();
        scheduledDate.setTime(alarm.scheduledTime);
        result += "name:" + alarm.name + ", ";
        result += "scheduledDate:" + DateUtils.dateToString(scheduledDate) + ", ";
        result += "periodInMinutes:" + alarm.periodInMinutes;
    }
    result += "}";
    return result;
}
export async function getNextSearchDate(alarm) {
    await log(">> getNextSearchDate: alarm=" + getDisplayableAlarm(alarm));
    let nextSearchDate = null;
    if (alarm) {
        let now = new Date();
        now.setTime(alarm.scheduledTime);
        nextSearchDate = DateUtils.dateToString(now, false);
    }
    await log("<< getNextSearchDate: nextSearchDate=" + nextSearchDate);
    return nextSearchDate;
}
