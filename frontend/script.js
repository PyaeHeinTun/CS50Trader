// Call the router check when the page loads
import {checkAuthAndRoute} from "./script/router.js"

document.addEventListener('DOMContentLoaded', () => {
    checkAuthAndRoute();
});