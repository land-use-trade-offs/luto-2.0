
document.addEventListener('DOMContentLoaded', function () {

    // Load the default values
    var lucc = '';
    var map_name = '';
    var year = '';
    var names = [];
    var file_name = '';

    // Load the selected data to report HTML
    load_data( get_dataDir() + '/data/Map_data/lumap_2010.html');


    var lucc_names = {
        "Ag_LU": ['Apples', 'Beef - modified land', 'Beef - natural land', 'Citrus', 'Cotton', 'Dairy - modified land', 'Dairy - natural land',
            'Grapes', 'Hay', 'Nuts', 'Other non-cereal crops', 'Pears', 'Plantation fruit', 'Rice', 'Sheep - modified land',
            'Sheep - natural land', 'Stone fruit', 'Sugar', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit',
            'Unallocated - modified land', 'Unallocated - natural land', 'Vegetables', 'Winter cereals', 'Winter legumes',
            'Winter oilseeds', 'Environmental Plantings', 'Riparian Plantings', 'Agroforestry'],
        "Ag_Mgt": ['Asparagopsis taxiformis', 'Ecological Grazing', 'Precision Agriculture'],
        "Land_Mgt": ['dry', 'irr'],
        'Non-Ag_LU': ['Environmental Plantings', 'Riparian Plantings', 'Agroforestry'],
        "lumap": ['Land-use All']
    }


    // Listen for changes in the lucc dropdown
    document.getElementById("select_1").addEventListener("change", function () {
        lucc = this.value;

        // Get the select_2 element
        var select_2 = document.getElementById("select_2");

        // Clear any existing options in select_2
        select_2.innerHTML = "";

        // Get the array of names for the selected lucc
        names = lucc_names[lucc];

        // Add each name as an option in select_2
        for (var i = 0; i < names.length; i++) {
            var option = document.createElement("option");
            option.value = names[i];
            option.text = names[i].charAt(0).toUpperCase() + names[i].slice(1);
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







    function load_data(path) {
        document.getElementById("map").innerHTML = `<object type="text/html" data=${path} ></object>`;
    }


    function update_fname() {
        lucc = document.getElementById("select_1").value;
        map_name = document.getElementById("select_2").value;
        year = document.getElementById("year").value;
        names = lucc_names[lucc];

        file_name = lucc == 'lumap' ? 'lumap_' + year + '.html' : lucc + '_' + String(names.indexOf(map_name)).padStart(2, '0') + '_' + map_name + '_' + year + '.html';
        file_name = get_dataDir() + '/data/Map_data/' + file_name;
        // Replace spaces with %20
        file_name = file_name.replace(/ /g, '%20');

        console.log(file_name);

        return file_name;
    }

    function get_dataDir() {
        // Get the data path
        var url = new URL(window.location.href);
        var path = url.pathname.split('/');
        path.splice(-3, 3);
        url.pathname = path.join('/');
        var dataDir = url.href;
        return dataDir;
    }


});




