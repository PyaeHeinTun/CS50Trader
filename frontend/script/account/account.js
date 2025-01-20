import { getBaseUrl } from "../context.js"
import { getCookie } from "../router.js"
import {makeToast} from "../toast.js"

document.addEventListener("DOMContentLoaded",() => {
    document.getElementById("change_password_form").addEventListener("submit",(e)=>{
        e.preventDefault()
        var form = new FormData(document.getElementById("change_password_form"))
        form.append("token",getCookie("token"))
        fetch(`${getBaseUrl()}/account/change_password`,{
            "method" : "POST",
            "body" : form,
        }).then(response=>{
            return response.json()
        }).then(data=>{
            makeToast(data['title'],data['message'])
        })
    })

    document.getElementById("add_balance").addEventListener("submit",(e)=>{
        e.preventDefault()
        var form = new FormData(document.getElementById("add_balance"))
        form.append("token",getCookie("token"))
        fetch(`${getBaseUrl()}/account/add_balance`,{
            "method" : "POST",
            "body" : form,
        }).then(response=>{
            return response.json()
        }).then(data=>{
            makeToast(data['title'],data['message'])
        })
    })
})