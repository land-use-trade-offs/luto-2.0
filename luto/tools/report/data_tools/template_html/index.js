document.addEventListener("DOMContentLoaded", function () {
    
    // Select the container to insert the table
    const container = document.getElementById('settingsTable');
    const data = document.getElementById('settingsTxt').innerText;


    data.split('\n').forEach(line => {
        let label, value;

        if (line.includes(':')) {
            // Split the line into label and value if a colon is present
            [label, ...value] = line.split(':');
            value = value.join(':'); // Join the value array into a string and split it into an array
        } else {
            // Use the entire line as the label if no colon is present
            label = String(line).trim();
            value = '';
        }

        const rowDiv = document.createElement('div');
        rowDiv.classList.add('row');

        const labelSpan = document.createElement('span');
        labelSpan.classList.add('label');
        labelSpan.textContent = label;

        const valueSpan = document.createElement('span');
        valueSpan.classList.add('value');
        valueSpan.textContent = value;

        if (value.length >= 40) {
            valueSpan.style.fontSize = '12px';
        } else if (value.length >= 20) {
            valueSpan.style.fontSize = '14px';
        }

        rowDiv.appendChild(labelSpan);
        rowDiv.appendChild(valueSpan);

        container.appendChild(rowDiv);
    });
});



