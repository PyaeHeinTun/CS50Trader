import { listenForLogOut, listenForSignInForm, listenForSignUpForm } from "./auth.js"

document.addEventListener("DOMContentLoaded",()=>{
    listenForSignInForm()
    listenForSignUpForm()
})