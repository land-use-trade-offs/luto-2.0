
document.addEventListener('DOMContentLoaded', function () {


    load_data('lumap_2030.html');


    function load_data(path) {
        document.getElementById("map").innerHTML = `<object type="text/html" data=${path} ></object>`;
    }


});




