import { isEmpty } from "./commonUtils.js";
let DateUtils = (function () {
    let Separator;
    (function (Separator) {
        Separator["DMY"] = "/";
        Separator["HMS"] = ":";
        Separator["DATE"] = " - ";
    })(Separator || (Separator = {}));
    const ONE_DAY = 24 * 60 * 60 * 1000;
    function dateToString(date, isReturnedAsAbsolute = true) {
        let result = "";
        if (!isEmpty(date)) {
            result += getDisplayableDayMonthYear(date, isReturnedAsAbsolute);
            result += Separator.DATE;
            result += getDisplayableHoursMinutesSeconds(date);
        }
        return result;
    }
    function stringToDate(displAbsoluteDate) {
        let splitted = displAbsoluteDate.split(Separator.DATE);
        let dmy = splitted[0];
        let hms = splitted[1];
        splitted = dmy.split(Separator.DMY);
        let displDay = splitted[0];
        let displMonth = splitted[1];
        let displYear = splitted[2];
        let day = parseFloat(displDay);
        let month = parseFloat(displMonth) - 1;
        let year = parseFloat(displYear);
        splitted = hms.split(Separator.HMS);
        let displHours = splitted[0];
        let displMinutes = splitted[1];
        let displSeconds = splitted[2];
        let hours = parseFloat(displHours);
        let minutes = parseFloat(displMinutes);
        let seconds = parseFloat(displSeconds);
        let date = new Date();
        date.setDate(day);
        date.setMonth(month);
        date.setFullYear(year);
        date.setHours(hours);
        date.setMinutes(minutes);
        date.setSeconds(seconds);
        return date;
    }
    function absoluteDateToRelativeDate(displAbsoluteDate) {
        let result = "";
        if (!isEmpty(displAbsoluteDate)) {
            let date = stringToDate(displAbsoluteDate);
            result = dateToString(date, false);
        }
        return result;
    }
    function getDisplayableDayMonthYear(date, isReturnedAsAbsolute = true) {
        let result;
        if (isReturnedAsAbsolute) {
            let displDay = date.getDate() < 10 ? "0" + date.getDate() : date.getDate();
            let displMonth = (date.getMonth() + 1) < 10 ? "0" + (date.getMonth() + 1) : (date.getMonth() + 1);
            let displYear = date.getFullYear();
            result = displDay + Separator.DMY + displMonth + Separator.DMY + displYear;
        }
        else {
            result = getRelativeDayMonthYear(date);
        }
        return result;
    }
    function getRelativeDayMonthYear(date) {
        let result;
        let now = new Date();
        let yesterday = new Date(now.getTime() - ONE_DAY);
        let today = now;
        let tomorrow = new Date(now.getTime() + ONE_DAY);
        let isYesterday = date.getDate() === yesterday.getDate() && date.getMonth() === yesterday.getMonth() && date.getFullYear() === yesterday.getFullYear();
        let isToday = date.getDate() === today.getDate() && date.getMonth() === today.getMonth() && date.getFullYear() === today.getFullYear();
        let isTomorrow = date.getDate() === tomorrow.getDate() && date.getMonth() === tomorrow.getMonth() && date.getFullYear() === tomorrow.getFullYear();
        if (isYesterday) {
            result = "Hier";
        }
        else if (isToday) {
            result = "Aujourd'hui";
        }
        else if (isTomorrow) {
            result = "Demain";
        }
        else {
            result = getDisplayableDayMonthYear(date);
        }
        return result;
    }
    function getDisplayableHoursMinutesSeconds(date) {
        let result = "";
        if (!isEmpty(date)) {
            let displHours = date.getHours() < 10 ? "0" + date.getHours() : date.getHours();
            let displMinutes = date.getMinutes() < 10 ? "0" + date.getMinutes() : date.getMinutes();
            let displSeconds = date.getSeconds() < 10 ? "0" + date.getSeconds() : date.getSeconds();
            result = displHours + Separator.HMS + displMinutes + Separator.HMS + displSeconds;
        }
        return result;
    }
    return {
        dateToString: dateToString,
        stringToDate: stringToDate,
        absoluteDateToRelativeDate: absoluteDateToRelativeDate,
        getDisplayableHoursMinutesSeconds: getDisplayableHoursMinutesSeconds
    };
})();
export default DateUtils;
