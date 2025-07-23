document.addEventListener("DOMContentLoaded", function () {
    
    // Select the container to insert the table
    const container = document.getElementById('settingsTable');
    const data = JSON.parse(document.getElementById('Supporting_info').innerText).model_run_settings;


   for (let i = 0; i < data.length; i++) {
        const line = data[i];

        label = line.parameter;
        value = line.val;

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
    }
});



