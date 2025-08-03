import { log } from "./commonUtils.js";
let DomUtils = (function () {
    function getByClassName(className) {
        return document.getElementsByClassName(className);
    }
    function getById(id) {
        let elt = document.getElementById(id);
        if (elt === null) {
            log("getById: unknown id=[" + id + "]", true);
        }
        return elt;
    }
    function show(id, display = "block") {
        getById(id).style.display = display;
    }
    function hide(id) {
        getById(id).style.display = "none";
    }
    return {
        getByClassName: getByClassName,
        getById: getById,
        show: show,
        hide: hide,
    };
})();
export default DomUtils;
