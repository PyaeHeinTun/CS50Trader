export function renderTableHistory(trades) {
    // Function to render table rows
    function renderTableRows(data) {
        const tbody = document.getElementById("trade-tbody");
        tbody.innerHTML = ""

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
    renderTableRows(trades);
}