
document.addEventListener('DOMContentLoaded', function () {

    const map = L.map('map').setView([-28, 493], 4);

    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    var popup = L.popup();

    function onMapClick(e) {
        popup
            .setLatLng(e.latlng)
            .setContent("You clicked the map at " + e.latlng.toString())
            .openOn(map);
    }

    map.on('click', onMapClick);

    $.ajax({
        url: "./",
        success: function(data){
           $(data).find("a:contains(.jpg)").each(function(){
      
              console.log(data);
      
           });
        }
      });
    

});




