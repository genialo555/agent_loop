import { log } from "./commonUtils.js";
let SearchUtils = (function () {
    let SearchStatus;
    (function (SearchStatus) {
        SearchStatus["OK"] = "OK";
        SearchStatus["NO_RESULTS"] = "(Aucun r\u00E9sultat)";
        SearchStatus["ERROR_GENERIC"] = "(Une erreur s'est produite)";
        SearchStatus["ERROR_NO_CNX"] = "(Impossible d'acc\u00E9der au site leboncoin.fr)";
    })(SearchStatus || (SearchStatus = {}));
    async function search(tabIndex, searchName, searchUrl, callback) {
        await log(">> SearchUtils.search: tabIndex=" + tabIndex + ", searchName=" + searchName + ", searchUrl=" + searchUrl);
        let httpStatus, data;
        try {
            let response = await fetch(searchUrl);
            httpStatus = response.status;
            data = await response.text();
        }
        catch (e) {
            httpStatus = 500;
            await log("SearchUtils.search: got error=" + e);
        }
        let searchResults = await _processResponse(httpStatus, data);
        await log("<< SearchUtils.search: invoking callback");
        callback(tabIndex, searchName, searchResults);
    }
    async function _processResponse(httpStatus, response) {
        await log(">> SearchUtils.processResponse: httpStatus=" + httpStatus);
        let offerTitleAndUrl = null;
        let searchStatus = null;
        if (httpStatus === 200) {
            offerTitleAndUrl = await _extractFirstOffer(response);
            if (offerTitleAndUrl) {
                searchStatus = SearchUtils.SearchStatus.OK;
            }
            else {
                let hasNoResults = response.indexOf('data-test-id="noResultMessages"') !== -1;
                await log("LbcSearchUtils.processResponse: hasNoResults=" + hasNoResults);
                searchStatus = hasNoResults ? SearchUtils.SearchStatus.NO_RESULTS : SearchUtils.SearchStatus.ERROR_GENERIC;
            }
        }
        else {
            searchStatus = SearchUtils.SearchStatus.ERROR_NO_CNX;
        }
        let searchResponse = {
            searchStatus: searchStatus,
            offerTitle: offerTitleAndUrl === null || offerTitleAndUrl === void 0 ? void 0 : offerTitleAndUrl.offerTitle,
            offerUrl: offerTitleAndUrl === null || offerTitleAndUrl === void 0 ? void 0 : offerTitleAndUrl.offerUrl
        };
        await log("<< SearchUtils.processResponse: " + JSON.stringify(searchResponse));
        return searchResponse;
    }
    async function _extractFirstOffer(input) {
        await log(">> SearchUtils._extractFirstOffer");
        let match = input.match(/<a.*?data-qa-id="aditem_container" data-test-id="ad".*?href="(.*?)"/);
        if (!match) {
            await log("<< SearchUtils._extractFirstOffer: couldn't find index of first offer");
            return null;
        }
        let offerUrl = match[1];
        match = input.match(/<p data-qa-id="aditem_title".*?title="(.*?)"/);
        if (!match) {
            await log("<< SearchUtils._extractFirstOffer: couldn't find offer title");
            return null;
        }
        let offerTitle = match[1];
        match = input.match(/<p data-test-id="price".*?aria-label="(.*?)"/);
        if (!match) {
            await log("<< SearchUtils._extractFirstOffer: couldn't find offer price");
            return null;
        }
        let offerPrice = match[1];
        offerTitle = offerTitle + ` (${offerPrice})`;
        let result = {
            offerTitle: offerTitle,
            offerUrl: offerUrl
        };
        await log("<< SearchUtils._extractFirstOffer: result=[" + JSON.stringify(result) + "]");
        return result;
    }
    return {
        SearchStatus: SearchStatus,
        search: search,
    };
})();
export default SearchUtils;
