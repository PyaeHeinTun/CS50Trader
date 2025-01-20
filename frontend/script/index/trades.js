import { getBaseUrl } from "../context.js";
import { getCookie } from "../router.js";

export function renderTable() {
    // Data to populate the table
    var trades = [];

    // Get Data From Server
    var form = new FormData()
    form.append("token",getCookie("token"))
    fetch(`${getBaseUrl()}/dashboard/open_trades`,{
        "method" : "POST",
        "body" : form,
    }).then(response=>{
        if (response.ok){
            return response.json()
        }
        return
    }).then(json=>{
        trades = json['open_trades']
        // Render the rows
        renderTableRows(trades);
    })

    // Function to render table rows
    function renderTableRows(data) {
        const tbody = document.getElementById("trade-tbody");

        data.forEach((trade) => {
            const row = document.createElement("tr");
            row.className = "bg-white hover:bg-gray-100";

            row.innerHTML = `
                <td class="border border-gray-300 px-4 py-2">${trade.pair}</td>
                <td class="border border-gray-300 px-4 py-2 ${trade.actionClass}">${trade.action}</td>
                <td class="border border-gray-300 px-4 py-2">${trade.entryPrice}</td>
                <td class="border border-gray-300 px-4 py-2">${trade.exitPrice}</td>
                <td class="border border-gray-300 px-4 py-2 ${trade.profitLossClass}">${trade.profitLoss}</td>
                <td class="border border-gray-300 px-4 py-2">${trade.timestamp}</td>
            `;

            tbody.appendChild(row);
        });
    }
}