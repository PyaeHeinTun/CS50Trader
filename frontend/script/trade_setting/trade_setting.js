const jsonData = {
    "TimeFrame": "1m",
    "TradeInfo": {
        "leverage": 20,
        "stakeAmmount": 2
    },
};

const container = document.getElementById('json-container');

function renderJSON(data, container) {
    for (const key in data) {
        if (typeof data[key] === 'object' && !Array.isArray(data[key])) {
            // Create a collapsible section for nested objects
            const section = document.createElement('div');
            section.className = 'border border-gray-300 rounded-lg p-4';
            section.innerHTML = `
                <h2 class="text-lg font-semibold text-gray-700 mb-4">${key}</h2>
                <div class="space-y-2" id="${key}-container"></div>
            `;
            container.appendChild(section);

            // Recursively render nested object
            renderJSON(data[key], section.querySelector(`#${key}-container`));
        } else if (Array.isArray(data[key])) {
            // Display array values
            const arraySection = document.createElement('div');
            arraySection.className = 'mb-4';
            arraySection.innerHTML = `
                <label class="block text-gray-600 font-medium mb-2">${key}</label>
                <ul class="list-disc list-inside pl-4 space-y-1">
                    ${data[key].map(value => `<li>${value}</li>`).join('')}
                </ul>
            `;
            container.appendChild(arraySection);
        } else {
            // Display single key-value pair
            const inputField = document.createElement('div');
            inputField.className = 'mb-4';
            inputField.innerHTML = `
                <label class="block text-gray-600 font-medium mb-2">${key}</label>
                <input type="text" value="${data[key]}" readonly class="w-full border border-gray-300 rounded px-3 py-2 bg-gray-50 text-gray-700">
            `;
            container.appendChild(inputField);
        }
    }
}

// Render JSON data
renderJSON(jsonData, container);