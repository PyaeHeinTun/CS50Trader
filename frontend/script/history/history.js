import { getBaseUrl } from "../context.js";
import { getCookie } from "../router.js";
import { makeToast } from "../toast.js";
import { renderTableHistory } from "./render_table.js";

// Select the filter form elements
const tradingPairSelect = document.getElementById('trading-pair');
const selectDateInput = document.getElementById('select-date');

// Add event listener for the "Apply Filters" button
document.addEventListener("DOMContentLoaded",()=>{
    // List of trading pairs
    fetch(`${getBaseUrl()}/history/pair_list`,{
        "method" : "GET",
    }).then(response=>{
        if (response.ok){
            return response.json()
        }
    }).then(data=>{
        const tradingPairs = data["pair_list"];
        var selectDateTimestamp = Math.floor(new Date().getTime() / 1000)
        // Get the <select> element by ID
        const tradingPairSelect = document.getElementById('trading-pair');

        // Loop through the array and create <option> elements
        tradingPairs.forEach(pair => {
            const option = document.createElement('option'); // Create an <option> element
            option.value = pair; // Set the value of the option
            option.textContent = pair; // Set the display text of the option
            tradingPairSelect.appendChild(option); // Add the option to the <select>
        });

        // Apply Filter BTN
        document.querySelector('button.bg-green-500').addEventListener('click', () => {
            // Convert dates to Unix timestamps
            selectDateTimestamp = selectDateInput.value ? Math.floor(new Date(selectDateInput.value).getTime() / 1000) : null;
            fetchAndRenderTable(selectDateTimestamp,1)
        });

        document.getElementById("previous_btn").addEventListener("click",()=>{
            var pageText = document.getElementById("page_number").innerText
            const currentPage = parseInt(pageText.match(/\d+/)[0], 10);
            fetchAndRenderTable(selectDateTimestamp,currentPage-1)
        })

        document.getElementById("next_btn").addEventListener("click",()=>{
            var pageText = document.getElementById("page_number").innerText
            const currentPage = parseInt(pageText.match(/\d+/)[0], 10);
            fetchAndRenderTable(selectDateTimestamp,currentPage+1)
        })

        fetchAndRenderTable(selectDateTimestamp,1)
    })


    function fetchAndRenderTable(selectDateTimestamp,page) {
        var form = new FormData()
        form.append("token",getCookie("token"))
        form.append("tradingPair",tradingPairSelect.value)
        form.append("selectDate",selectDateTimestamp)
        form.append("page",page)
        
        fetch(`${getBaseUrl()}/history/trade_history`,{
            "method" : "POST",
            "body" : form,
        }).then(response=>{
            return response.json()
        }).then(data=>{
            makeToast(data['title'],data['message'])
            renderTableHistory(data['history'])
            var pagination =  data['pagination']
            var previousBtn = document.getElementById("previous_btn")
            var nextBtn = document.getElementById("next_btn")
            var page_number = document.getElementById("page_number")

            // Update Previous Next And Page Number
            if (pagination['hasNextPage']){
                nextBtn.removeAttribute("disabled")
            }else{
                nextBtn.setAttribute("disabled",true)
            }
            if (pagination['hasPreviousPage']){
                previousBtn.removeAttribute("disabled")
            }else{
                previousBtn.setAttribute("disabled",true)
            }

            // Update Middle Text
            page_number.innerText = `Page ${pagination['currentPage']} of ${pagination['totalPage']}`
        })
    }
})