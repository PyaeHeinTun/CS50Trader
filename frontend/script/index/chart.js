import { getBaseUrl } from "../context.js";
import { getCookie } from "../router.js";
import { makeToast } from "../toast.js";
import { renderMetrics } from "./metrics.js";

let chartData = {};
let marketChart;
export function renderChart() {
    // Chart.js Example
    const ctx = document.getElementById('marketChart').getContext('2d');
    var form = new FormData()
    form.append("token",getCookie("token"))
    fetch(`${getBaseUrl()}/dashboard/chart`,{
        "method" : "POST",
        "body" : form,
    }).then(response=>{
        if (response.ok){
            return response.json()
        }
        makeToast("Could not fetch data.","Chart Data Cannot be loaded from server.")
    }).then(jsonData=>{
        chartData = jsonData['chart']
        var metrics = chartData['7days'].metrics
        // Initialize Chart.js
        renderMetrics(metrics)
        marketChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData['7days'].labels,
                datasets: [{
                    label: 'Profit',
                    data: chartData['7days'].data,
                    borderColor: '#34D399',
                    borderWidth: 2,
                    fill: false,
                }],
            },
            options: {
                responsive: true,
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true },
                },
            },
        });
    })
    // Update chart on dropdown change
    document.getElementById('timeFrame').addEventListener('change', (event) => {
        const timeFrame = event.target.value;

        // Update chart labels and data
        marketChart.data.labels = chartData[timeFrame].labels;
        marketChart.data.datasets[0].data = chartData[timeFrame].data;
        marketChart.update();

        var metrics = chartData[timeFrame].metrics
        renderMetrics(metrics)
    });
}