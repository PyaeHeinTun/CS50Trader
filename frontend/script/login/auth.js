import { getBaseUrl, getFolderPathForUI } from "../context.js"
import { removeCookie, setCookie } from "../router.js"
import { makeToast } from "../toast.js"

export function listenForSignInForm() {
    var form = document.getElementById("signinform")
    form.addEventListener("submit",(e)=>{
        e.preventDefault()
        var formData = new FormData(form)
        fetch(`${getBaseUrl()}/auth/login`,{
            "method": "POST",
            "body" : formData,
        }).then(response=>{
            if (!response.ok){
                response.json().then(json=>{
                    makeToast(json['title'],json['message'])
                })
                return;
            }
            return response.json()
        }).then(json=>{
            setCookie("token",json['access_token'])
            setCookie("username",json['username'])
            window.location.href = `${getFolderPathForUI()}/index.html`
        })
    })
}

export function listenForSignUpForm() {
    var form = document.getElementById("signupform")
    form.addEventListener("submit",(e)=>{
        e.preventDefault()
        var formData = new FormData(form)
        fetch(`${getBaseUrl()}/auth/signup`,{
            "method": "POST",
            "body" : formData,
        }).then(response=>{
            if (!response.ok){
                response.json().then(json=>{
                    makeToast(json['title'],json['message'])
                })
                return;
            }
            return response.json()
        }).then(json=>{
            setCookie("token",json['access_token'])
            window.location.href = `${getFolderPathForUI()}/index.html`
        })
    })
}

export function listenForLogOut() {
    var elementList = document.getElementsByClassName("logout_btn")
    var length = elementList.length
    
    
    for (let index = 0; index < length; index++) {
        const element = elementList.item(index);
        element.removeEventListener("click",logoutFunction)
        element.addEventListener("click",logoutFunction)
    }
    
}

function logoutFunction() {
    removeCookie("token")
    removeCookie("username")
    window.location.href = `${getFolderPathForUI()}/login.html`;
}