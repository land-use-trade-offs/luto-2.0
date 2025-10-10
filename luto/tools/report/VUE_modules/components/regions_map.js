window.RegionsMap = {

  props: {
    mapData: {
      type: String,
      required: true
    },
  },

  setup(props) {
    const { ref, inject, onMounted, computed } = Vue;
    const globalMapViewpoint = inject('globalMapViewpoint');
    const selectedRegion = inject('globalSelectedRegion');

    const map = ref(null);
    const boundingBox = ref(null);
    const loadScript = window.loadScript;
    const selectedBaseMap = ref('OpenStreetMap');
    const tileLayers = ref({});
    const baseMapOptions = ref(['OpenStreetMap', 'Satellite', 'None']);



    const initMap = () => {
      // Initialize the map with saved viewpoint
      map.value = L.map('map', {
        zoomControl: false
      }).setView(globalMapViewpoint.value.center, globalMapViewpoint.value.zoom);

      // Save current view on map events
      map.value.on('moveend zoomend', () => {
        globalMapViewpoint.value.center = [map.value.getCenter().lat, map.value.getCenter().lng];
        globalMapViewpoint.value.zoom = map.value.getZoom();
      });

      // Create tile layers but don't add them yet
      tileLayers.value = {
        OSM: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors',
          maxZoom: 18
        }),
        Satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
          maxZoom: 18
        })
      };

      // Add initial base map
      const initialMapType = selectedBaseMap.value === 'OpenStreetMap' ? 'OSM' : selectedBaseMap.value;
      if (initialMapType !== 'None') {
        tileLayers.value[initialMapType].addTo(map.value);
      }
    };

    // Update map when region changes - only animate if region actually changed
    const updateMap = (forceAnimation = false) => {
      // Check if this is a real region change or just a page navigation
      const regionChanged = globalMapViewpoint.value.lastSelectedRegion !== selectedRegion.value;

      if (!regionChanged && !forceAnimation) {
        // Just update the region layer without animation
        updateRegionLayerOnly();
        return;
      }

      // Update the last selected region
      globalMapViewpoint.value.lastSelectedRegion = selectedRegion.value;

      // Fade out existing elements first
      fadeOutExistingElements().then(() => {
        // Remove existing rectangles after fade out
        if (boundingBox.value) {
          map.value.removeLayer(boundingBox.value);
        }

        // Calculate bounds for smooth transition
        const bbox = window.NRM_AUS_centroid_bbox[selectedRegion.value].bounding_box;
        const bounds = [
          [bbox[1], bbox[0]], // Southwest corner
          [bbox[3], bbox[2]]  // Northeast corner
        ];

        // Smooth pan and zoom to the new region
        map.value.flyToBounds(bounds, {
          padding: [20, 20],
          duration: 1.5,
          easeLinearity: 0.25
        });

        // Add new elements with a delay to allow map transition
        setTimeout(() => {
          addRegionLayer();
        }, 500);
      });
    };

    // Update only the region layer without map animation
    const updateRegionLayerOnly = () => {
      // Remove existing rectangles
      if (boundingBox.value) {
        map.value.removeLayer(boundingBox.value);
      }

      // Add new region layer immediately
      addRegionLayer();
    };

    // Fade out existing map elements
    const fadeOutExistingElements = () => {
      return new Promise((resolve) => {
        if (boundingBox.value) {
          animateRectangleOpacity(boundingBox.value, 0.2, 0, 300);
        }
        setTimeout(resolve, 300);
      });
    };

    // Add new elements to the map
    const addRegionLayer = () => {
      // Skip adding region overlay for AUSTRALIA
      if (selectedRegion.value === 'AUSTRALIA') {
        return;
      }

      // Find the actual region feature from NRM_AUS data
      const regionLayer = window.NRM_AUS.features.find(feature =>
        feature.properties.NRM_REGION === selectedRegion.value
      );

      // Add the actual region polygon with initial opacity 0
      boundingBox.value = L.geoJSON(regionLayer, {
        style: {
          color: '#3b82f6',
          weight: 2,
          fillColor: '#3b82f6',
          fillOpacity: 0,
          opacity: 0
        }
      }).addTo(map.value);

      // Fade in new elements
      setTimeout(() => {
        animateRectangleOpacity(boundingBox.value, 0, 0.2, 500);
      }, 200);
    };

    // Animation functions
    const animateRectangleOpacity = (rectangle, startFillOpacity, endFillOpacity, duration) => {
      const startTime = Date.now();
      const startStrokeOpacity = startFillOpacity > 0 ? 1 : 0;
      const endStrokeOpacity = endFillOpacity > 0 ? 1 : 0;

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easeInOut(progress);

        const currentFillOpacity = startFillOpacity + (endFillOpacity - startFillOpacity) * easedProgress;
        const currentStrokeOpacity = startStrokeOpacity + (endStrokeOpacity - startStrokeOpacity) * easedProgress;

        rectangle.setStyle({
          fillOpacity: currentFillOpacity,
          opacity: currentStrokeOpacity
        });

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      requestAnimationFrame(animate);
    };

    const easeInOut = (t) => {
      return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    };

    // Load region data and initialize map on component mount
    onMounted(async () => {
      try {
        // Load region data
        await loadScript("services/MapService.js", 'MapService');
        await loadScript('data/geo/NRM_AUS_centroid_bbox.js', 'NRM_AUS_centroid_bbox');
        await loadScript('data/geo/NRM_AUS.js', 'NRM_AUS');

        // Initialize map first
        initMap();

        // Skip initial map data load - will be loaded by the watcher when props are populated
        // The watch handler will take care of loading map data when props are ready

        // Update map if a region is already selected
        if (selectedRegion.value) {
          updateMap();
        }
      } catch (error) {
        console.error('Failed to initialize RegionsMap:', error);
      }
    });

    const loadMapData = async () => {

      // Always remove existing overlays first
      map.value.eachLayer((layer) => {
        if (layer instanceof L.ImageOverlay) {
          map.value.removeLayer(layer);
        }
      });

      const data = props.mapData;

      if (!data.img_str || !data.bounds) {
        console.warn('Map data is missing required properties (img_str or bounds):', data);
        // No overlay will be added - map shows base layer only
        return;
      }

      // Add new image overlay only if data is valid
      const imageOverlay = L.imageOverlay(
        data.img_str,
        data.bounds,
        {
          className: 'crisp-image'
        }
      ).addTo(map.value);

      // Apply CSS to disable image interpolation
      setTimeout(() => {
        const imgElement = imageOverlay.getElement();
        if (imgElement) {
          imgElement.style.imageRendering = 'pixelated';
          imgElement.style.imageRendering = '-moz-crisp-edges';
          imgElement.style.imageRendering = 'crisp-edges';
        }
      }, 100);
    };

    Vue.watch(() => props.mapData, (newVal) => {
      loadMapData();
    });

    Vue.watch(selectedRegion, (newValue, oldValue) => {
      if (newValue) {
        // Only trigger animation if this is a real region change (not a page navigation)
        const forceAnimation = oldValue !== undefined && oldValue !== newValue;
        updateMap(forceAnimation);
      }
    });

    const handleBaseMapChange = (mapType) => {
      selectedBaseMap.value = mapType;
      // Map display names to internal values
      const mapTypeMap = {
        'OpenStreetMap': 'OSM',
        'Satellite': 'Satellite',
        'None': 'None'
      };
      changeBaseMap(mapTypeMap[mapType]);
    };

    const changeBaseMap = (mapType) => {
      // Remove all existing tile layers
      Object.values(tileLayers.value).forEach(layer => {
        if (map.value.hasLayer(layer)) {
          map.value.removeLayer(layer);
        }
      });

      // Add new tile layer if not 'None'
      if (mapType !== 'None' && tileLayers.value[mapType]) {
        tileLayers.value[mapType].addTo(map.value);
      }
    };

    return {
      selectedRegion,
      updateMap,
      selectedBaseMap,
      changeBaseMap,
      baseMapOptions,
      handleBaseMapChange
    };
  },
  template: `
    <div class="bg-white h-screen flex flex-col">
    
      <!-- Map Container with Controls Overlay - Base map selector and map element -->
      <div class="bg-white shadow-lg flex-1 relative">

        <!-- Base Map Selector - Dropdown to switch between map types -->
        <div class="absolute top-[40px] left-[20px] z-50 bg-white/70 rounded-lg shadow-lg z-[9999]">
          <div style="min-width: 150px;">
            <filterable-dropdown
              :use-search="false"
              :items="baseMapOptions"
              :selected-value="selectedBaseMap"
              placeholder="Select base map"
              @change="handleBaseMapChange"
            />
          </div>
        </div>

        <!-- Map Container - Leaflet map will be initialized here -->
        <div id="map" class="w-full h-full relative z-10"></div>
      </div>
    </div>
  `
};