import { log } from "./commonUtils.js";

let SearchUtils = (function () {
    let SearchStatus;
    (function (SearchStatus) {
        SearchStatus["OK"] = "OK";
        SearchStatus["NO_RESULTS"] = "(Aucun r\u00E9sultat)";
        SearchStatus["ERROR_GENERIC"] = "(Une erreur s'est produite)";
        SearchStatus["ERROR_NO_CNX"] = "(Impossible d'acc\u00E9der au site)";
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
        let searchResponse = {
            searchStatus: null,
            offerTitle: null,
            offerUrl: null
        };
        if (httpStatus === 200) {
            searchResponse = await _extractFirstOffer(response);
        }
        else {
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_NO_CNX;
        }
        await log("<< SearchUtils.processResponse: " + JSON.stringify(searchResponse));
        return searchResponse;
    }

    async function _extractFirstOffer(input) {
        await log(">> SearchUtils._extractFirstOffer");
        let searchResponse = {
            searchStatus: null,
            offerTitle: null,
            offerUrl: null
        };
        let match = input.match(/<script id="__NEXT_DATA__" type="application\/json">(.*?)<\/script>/);
        if (!match) {
            await log("<< SearchUtils._extractFirstOffer: couldn't retrieve json");
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let jsonString = match[1];
        let json;
        try {
            json = JSON.parse(jsonString);
        }
        catch (error) {
            await log("<< SearchUtils._extractFirstOffer: couldn't parse json: ", error);
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let props = json.props;
        if (props === undefined) {
            await log("<< SearchUtils._extractFirstOffer: couldn't retrieve props");
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let pageProps = props.pageProps;
        if (pageProps === undefined) {
            await log("<< SearchUtils._extractFirstOffer: couldn't retrieve pageProps");
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let searchData = pageProps.searchData;
        if (searchData === undefined) {
            await log("<< SearchUtils._extractFirstOffer: couldn't retrieve search data");
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let hasNoResults = searchData.total === 0;
        if (hasNoResults) {
            await log("LbcSearchUtils._extractFirstOffer: hasNoResults=" + hasNoResults);
            searchResponse.searchStatus = SearchUtils.SearchStatus.NO_RESULTS;
            return searchResponse;
        }
        let offer = searchData.ads[0];
        searchResponse.searchStatus = SearchUtils.SearchStatus.OK;
        searchResponse.offerTitle = offer.subject + ` (${offer.price_cents / 100} €)`;
        searchResponse.offerUrl = offer.url;
        await log("<< SearchUtils._extractFirstOffer: result=[" + JSON.stringify(searchResponse) + "]");
        return searchResponse;
    }

    async function searchFacebookMarketplace(searchUrl, callback) {
        await log(">> SearchUtils.searchFacebookMarketplace: searchUrl=" + searchUrl);
        let httpStatus, data;
        try {
            let response = await fetch(searchUrl);
            httpStatus = response.status;
            data = await response.text();
        }
        catch (e) {
            httpStatus = 500;
            await log("SearchUtils.searchFacebookMarketplace: got error=" + e);
            await writeResponseToStorage(`Error: ${e}\n`);
        }
        await writeResponseToStorage(data); // Log response to storage
        let searchResults = await _processFacebookResponse(httpStatus, data);
        await log("<< SearchUtils.searchFacebookMarketplace: invoking callback");
        callback(searchResults);
    }

    async function _processFacebookResponse(httpStatus, response) {
        await log(">> SearchUtils.processFacebookResponse: httpStatus=" + httpStatus);
        let searchResponse = {
            searchStatus: null,
            offers: []
        };
        if (httpStatus === 200) {
            searchResponse = await _extractFacebookOffers(response);
        }
        else {
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_NO_CNX;
        }
        await log("<< SearchUtils.processFacebookResponse: " + JSON.stringify(searchResponse));
        return searchResponse;
    }

    async function _extractFacebookOffers(input) {
        await log(">> SearchUtils._extractFacebookOffers");
        let searchResponse = {
            searchStatus: null,
            offers: []
        };
        let match = input.match(/<script type="application\/json" data-content-len="\d+">(.*?)<\/script>/);
        if (!match) {
            await log("<< SearchUtils._extractFacebookOffers: couldn't retrieve json");
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }
        let jsonString = match[1];
        let json;
        try {
            json = JSON.parse(jsonString);
        }
        catch (error) {
            await log("<< SearchUtils._extractFacebookOffers: couldn't parse json: ", error);
            searchResponse.searchStatus = SearchUtils.SearchStatus.ERROR_GENERIC;
            return searchResponse;
        }

        let edges = json.require[0][1][2][0].__bbox.result.data.marketplace_search.feed_units.edges;
        if (!edges) {
            await log("<< SearchUtils._extractFacebookOffers: no edges found");
            searchResponse.searchStatus = SearchUtils.SearchStatus.NO_RESULTS;
            return searchResponse;
        }

        for (let edge of edges) {
            let node = edge.node;
            if (node && node.listing) {
                let listing = node.listing;
                searchResponse.offers.push({
                    title: listing.marketplace_listing_title,
                    price: listing.listing_price.formatted_amount,
                    url: `https://www.facebook.com/marketplace/item/${listing.id}`,
                    image: listing.primary_listing_photo.image.uri,
                    location: listing.location.reverse_geocode.city_page.display_name,
                    seller: listing.marketplace_listing_seller.name
                });
            }
        }

        searchResponse.searchStatus = SearchUtils.SearchStatus.OK;
        await log("<< SearchUtils._extractFacebookOffers: result=[" + JSON.stringify(searchResponse) + "]");
        return searchResponse;
    }

    async function writeResponseToStorage(response) {
        let responses = await chrome.storage.local.get('responses');
        responses = responses.responses || [];
        responses.push(response);
        await chrome.storage.local.set({ 'responses': responses });
    }

    async function downloadResponses() {
        let responses = await chrome.storage.local.get('responses');
        responses = responses.responses || [];
        let blob = new Blob(responses, { type: 'text/plain' });
        let url = URL.createObjectURL(blob);
        chrome.downloads.download({
            url: url,
            filename: 'responses.txt',
            saveAs: true
        });
    }

    return {
        SearchStatus: SearchStatus,
        search: search,
        searchFacebookMarketplace: searchFacebookMarketplace,
        downloadResponses: downloadResponses
    };
})();

export default SearchUtils;
