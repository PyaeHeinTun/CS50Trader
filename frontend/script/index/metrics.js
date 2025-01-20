export function renderMetrics(metrics) {
    // Function to format numbers as currency
    const formatCurrency = (value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
        }).format(value);
    };

    // Function to update performance metrics
    const updatePerformanceMetrics = () => {
        document.getElementById('profit-loss').textContent = formatCurrency(metrics.profitLoss);
        document.getElementById('win-rate').textContent = `${metrics.winRate.toFixed(1)}%`;
        document.getElementById('trades').textContent = metrics.totalTrades;
        document.getElementById('holdings').textContent = formatCurrency(metrics.holdings);
    };

    // Update metrics on page load
    updatePerformanceMetrics()
}