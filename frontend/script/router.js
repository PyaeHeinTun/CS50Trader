import { getBaseUrl, getFolderPathForUI } from "./context.js";
import { listenForLogOut } from "./login/auth.js";

// Helper function to get a cookie value by name
export function getCookie(name) {
    const cookies = document.cookie.split('; ');
    for (const cookie of cookies) {
        const [key, value] = cookie.split('=');
        if (key === name) return value;
    }
    return null;
}

// Helper function to set a cookie value by name
export function setCookie(name,value,days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}

export function removeCookie(name) {
    document.cookie = name + "=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/";
}

// Function to validate JWT (basic validation for expiration)
function isTokenValid(token) {
    var formData = new FormData()
    formData.append("token",token)
    return fetch(`${getBaseUrl()}/auth/checktoken`,{
            "method": "POST",
            "body" : formData,
        }).then(response=>{
            if (!response.ok){
                return false
            }
            return true
        })
}

// Router check
export function checkAuthAndRoute() {
    const token = getCookie('token');

    if (!token || !isTokenValid(token)) {
        window.location.href = `${getFolderPathForUI()}/login.html`;
    } else {
        listenForLogOut()
    }
}