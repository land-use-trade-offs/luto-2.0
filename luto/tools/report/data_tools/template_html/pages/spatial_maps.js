
document.addEventListener('DOMContentLoaded', function () {

    // Load the default values
    let lucc = '';
    let map_name = '';
    let year = '';
    let names = [];
    let file_name = '';
    let map_idx = '0';

    // Load the selected data to report HTML
    load_data(get_dataDir() + '/data/Map_data/lumap_2010.html');
   
    // Get the names (original) and renames (rename for reporting)
    let lucc_names = document.getElementById('SPATIAL_MAP_DICT').innerText;
    lucc_names = lucc_names.replace(/'/g, '"');       // JSON.parse() does not accept single quotes
    lucc_names = JSON.parse(lucc_names);

    let lucc_rename = document.getElementById('RENAME_AM_NON_AG').innerText;
    lucc_rename = lucc_rename.replace(/'/g, '"');       // JSON.parse() does not accept single quotes
    lucc_rename = JSON.parse(lucc_rename);

    // Initialize the select_2 dropdown
    let int_map_names = {
        'lumap':'All Land-use',
        'non_ag':'Non Agricultural', 
        'ammap':'Agricultural Management', 
        'lmmap':'Water Management',
    };
    init_select_2();



    // Listen for changes in the lucc dropdown
    document.getElementById("select_1").addEventListener("change", function () {
        lucc = this.value;

        console.log(lucc)

        // Get the select_2 element
        let select_2 = document.getElementById("select_2");

        // Clear any existing options in select_2
        select_2.innerHTML = "";

        // Get the array of names for the selected lucc
        names = lucc_names[lucc];

        // Add each name as an option in select_2
        for (let i = 0; i < names.length; i++) {
            let option = document.createElement("option");
            option.text = names[i] in int_map_names ? int_map_names[names[i]] : names[i];
            option.text = names[i] in lucc_rename ? lucc_rename[names[i]] : option.text;
            option.value = names[i];
            select_2.appendChild(option);
        }

        // Load the selected data to report HTML
        load_data(update_fname());

    });



    // Listen for changes in the select_2 dropdown
    document.getElementById("select_2").addEventListener("change", function () {
        // Load the selected data to report HTML
        load_data(update_fname());
    });



    // Load the the selected year
    document.getElementById("year").addEventListener("change", function () {
        // Load the selected data to report HTML
        load_data(update_fname());
    });



    // Increment year
    document.getElementById('increment').addEventListener('click', function () {
        let yearInput = document.getElementById('year');

        if (yearInput.value < yearInput.max) {
            yearInput.value = parseInt(yearInput.value) + parseInt(yearInput.step);
            document.getElementById('yearOutput').value = yearInput.value;
        }
        // Load the selected data to report HTML
        load_data(update_fname());
    });


    // Decrement year
    document.getElementById('decrement').addEventListener('click', function () {
        let yearInput = document.getElementById('year');
        if (yearInput.value > yearInput.min) {
            yearInput.value = parseInt(yearInput.value) - parseInt(yearInput.step);
            document.getElementById('yearOutput').value = yearInput.value;
        }
        // Load the selected data to report HTML
        load_data(update_fname());
    });


    // Function to load the data to the report HTML
    function load_data(path) {
        document.getElementById("map").innerHTML = `<object type="text/html" data=${path} ></object>`;
    }


    function init_select_2() {
        // Get the select_2 element
        let select_2 = document.getElementById("select_2");

        // Clear any existing options in select_2
        select_2.innerHTML = "";

        // Add each name as an option in select_2
        for (let key in int_map_names) {
            let option = document.createElement("option");
            option.value = key;
            option.text = int_map_names[key];
            select_2.appendChild(option);
        }
    }


    // Function to update the file name
    function update_fname() {

        // Get the selected values
        lucc = document.getElementById("select_1").value;
        map_name = document.getElementById("select_2").value;
        year = document.getElementById("year").value;
        names = lucc_names[lucc];


        // The index for Ag_Mgt is always 00
        map_idx = lucc == 'Ag_Mgt' ? '00' : String(names.indexOf(map_name)).padStart(2, '0');

        // The file name for Int_Map is different
        file_name = lucc == 'Int_Map' ? map_name + '_' + year + '.html' : lucc + '_' + map_idx + '_' + map_name + '_' + year + '.html';

        // Get the full path to the file
        file_name = get_dataDir() + '/data/Map_data/' + file_name;

        // Replace spaces with %20, so the file can be found by the browser
        file_name = file_name.replace(/ /g, '%20');

        console.log(file_name);

        return file_name;
    }


    // Function to get the data directory
    function get_dataDir() {
        // Get the data path, replace spaces with %20 so the browser can find the file
        let url = new URL(window.location.href.replace(/ /g, '%20'));
        let path = url.pathname.split('/');
        path.splice(-3, 3);
        url.pathname = path.join('/');
        let dataDir = url.href;
        return dataDir;
    }

    // Update the year range selection input and output values
    window.onload = function () {
        let yearInput = document.getElementById('year');
        let yearOutput = document.getElementById('yearOutput');
        let modelYears = eval(document.getElementById('model_years').innerText);

        // Sort the modelYears array in ascending order
        modelYears.sort(function (a, b) { return a - b; });

        yearInput.min = modelYears[0];
        yearInput.max = modelYears[modelYears.length - 1];
        yearInput.step = modelYears[1] - modelYears[0];   // The step is the difference between the first two elements
        yearInput.value = modelYears[0];
        yearOutput.value = modelYears[0];
    }


});




