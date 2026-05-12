// Linear interpolation between two [R,G,B] colours.
function _lerpRGB(c1, c2, t) {
    return [
        Math.round(c1[0] + (c2[0] - c1[0]) * t),
        Math.round(c1[1] + (c2[1] - c1[1]) * t),
        Math.round(c1[2] + (c2[2] - c1[2]) * t),
        255
    ];
}

// Returns a function(pixelValue) → [R,G,B,A] (0-255) or null (transparent).
// Float layers use the same ramp as the Python renderer and the colorbar:
//   positive: #FFF6B0 (pale yellow) → #FD933E (orange) → #800026 (dark maroon)
//   negative: #5D94DC (light blue)  → #104991 (mid blue) → #000E2B (dark navy)
//   zero:     #E1E1E1 (grey, on-land cell with no value for this layer)
//   nodata:   transparent
function getColorFn(meta) {
    const NODATA = -9999;

    if (meta.intOrFloat === 'int') {
        const codes = window.legend_registry?.[meta.legendKey]?.code_colors ?? {};
        return (value) => {
            if (value === null || value === NODATA) return null;
            return codes[String(Math.round(value))] ?? null;
        };
    } else {
        // raw_min_max: actual GeoTIFF value range for normalisation
        // min_max:     display-scale range (colorbar labels only — divided by RESFACTOR²×121)
        const [minVal, maxVal] = meta.raw_min_max ?? meta.min_max ?? [0, 1];
        // Anchor colours matching the colorbar gradient in regions_map colorbarInfo
        const POS = [[255,246,176], [253,147,62], [128,0,38]];   // yellow→orange→maroon
        const NEG = [[93,148,220],  [16,73,145],  [0,14,43]];    // light→mid→dark blue

        return (value) => {
            if (value === null || value === NODATA) return null;
            if (value === 0) return [225, 225, 225, 255];   // #E1E1E1

            if (value > 0 && maxVal > 0) {
                const t = Math.min(value / maxVal, 1);
                // 3-stop ramp: t∈[0,0.5] → stops[0→1], t∈[0.5,1] → stops[1→2]
                return t <= 0.5
                    ? _lerpRGB(POS[0], POS[1], t * 2)
                    : _lerpRGB(POS[1], POS[2], (t - 0.5) * 2);
            }
            if (value < 0 && minVal < 0) {
                const t = Math.min(Math.abs(value) / Math.abs(minVal), 1);
                // t=0 (near zero) → light blue, t=1 (most negative) → dark navy
                return t <= 0.5
                    ? _lerpRGB(NEG[0], NEG[1], t * 2)
                    : _lerpRGB(NEG[1], NEG[2], (t - 0.5) * 2);
            }
            return null;
        };
    }
}

// Patch L.ImageOverlay to guard every method that touches this._map against the
// remove-during-animation race:
//   _animateZoom  — fires via 'zoomanim' RAF, reads this._map.getZoomScale
//   _reset        — fires via 'zoomend' setTimeout(250ms), reads this._map.latLngToLayerPoint
// After removeLayer() nulls _map, any in-flight RAF or timer still fires these
// callbacks.  The null-guard makes removed overlays silently skip them.
;(function() {
    ['_animateZoom', '_reset'].forEach(function(method) {
        const orig = L.ImageOverlay.prototype[method];
        if (orig) {
            L.ImageOverlay.prototype[method] = function() {
                if (this._map) orig.apply(this, arguments);
            };
        }
    });
})();

window.RegionsMap = {
  name: 'RegionsMap',

  // v-draggable: makes any absolutely-positioned element freely moveable via mouse drag.
  // On first mousedown it snapshots the element's rendered position (handling bottom/transform
  // initial placement) and switches to explicit top/left so subsequent moves are predictable.
  directives: {
    draggable: {
      mounted(el) {
        el.style.cursor = 'grab';
        el.style.userSelect = 'none';
        let startX, startY, startLeft, startTop;

        const onMouseMove = (e) => {
          el.style.left = (startLeft + e.clientX - startX) + 'px';
          el.style.top = (startTop + e.clientY - startY) + 'px';
          el.style.cursor = 'grabbing';
        };

        const onMouseUp = () => {
          document.removeEventListener('mousemove', onMouseMove);
          document.removeEventListener('mouseup', onMouseUp);
          el.style.cursor = 'grab';
        };

        el.addEventListener('mousedown', (e) => {
          // Convert current rendered position to explicit top/left, clearing any
          // bottom/right/transform that were set by the initial CSS classes.
          const rect = el.getBoundingClientRect();
          const parentRect = el.offsetParent ? el.offsetParent.getBoundingClientRect() : { left: 0, top: 0 };
          el.style.transform = 'none';
          el.style.bottom = 'auto';
          el.style.right = 'auto';
          startLeft = rect.left - parentRect.left;
          startTop = rect.top - parentRect.top;
          el.style.left = startLeft + 'px';
          el.style.top = startTop + 'px';
          startX = e.clientX;
          startY = e.clientY;
          document.addEventListener('mousemove', onMouseMove);
          document.addEventListener('mouseup', onMouseUp);
          e.preventDefault();
        });
      }
    }
  },

  props: {
    mapData: {
      type: Object,
      default: () => ({})
    },
    overlayGeoJSON: {
      type: Object,
      default: null
    },
    regionType: {
      type: String,
      default: 'NRM'  // 'NRM' or 'STATE'
    },
    showLegend: {
      type: Boolean,
      default: true
    }
  },

  setup(props) {
    const { ref, inject, onMounted, computed } = Vue;
    const globalMapViewpoint = inject('globalMapViewpoint');
    const selectedRegion = inject('globalSelectedRegion');

    const map = ref(null);
    const boundingBox = ref(null);
    const gbf2Layer = ref(null);
    const loadScript = window.loadScript;
    const selectedBaseMap = ref('CartoDB');
    const tileLayers = ref({});
    const baseMapOptions = ref(['CartoDB', 'Satellite', 'None']);



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
        OSM: L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
          subdomains: 'abcd',
          maxZoom: 20
        }),
        Satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
          maxZoom: 18
        })
      };

      // Add initial base map
      const initialMapType = selectedBaseMap.value === 'CartoDB' ? 'OSM' : selectedBaseMap.value;
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
        const centroidBbox = props.regionType === 'STATE' ? window.AUS_STATE_centroid_bbox : window.NRM_AUS_centroid_bbox;
        const bbox = centroidBbox?.[selectedRegion.value]?.bounding_box;
        if (!bbox) return;
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

      // Find the actual region feature from the appropriate GeoJSON data
      const geoJsonData = props.regionType === 'STATE' ? window.AUS_STATE : window.NRM_AUS;
      const regionProp = props.regionType === 'STATE' ? 'STATE_NAME' : 'NRM_REGION';
      const regionLayer = geoJsonData?.features.find(feature =>
        feature.properties[regionProp] === selectedRegion.value
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
        await loadScript('data/map_layers/legend_registry.js', 'legend_registry');
        if (props.regionType === 'STATE') {
          await loadScript('data/geo/AUS_STATE_centroid_bbox.js', 'AUS_STATE_centroid_bbox');
          await loadScript('data/geo/AUS_STATE.js', 'AUS_STATE');
        } else {
          await loadScript('data/geo/NRM_AUS_centroid_bbox.js', 'NRM_AUS_centroid_bbox');
          await loadScript('data/geo/NRM_AUS.js', 'NRM_AUS');
        }

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

    let _currentRasterOverlay = null;

    const loadMapData = async () => {
      if (_currentRasterOverlay) {
        map.value.removeLayer(_currentRasterOverlay);
        _currentRasterOverlay = null;
      }

      const data = props.mapData;
      if (!data?.tif_b64 || !data?.intOrFloat) return;

      // Decode base64 → parse GeoTIFF (already EPSG:3857, no reprojection needed)
      const bytes     = Uint8Array.from(atob(data.tif_b64), c => c.charCodeAt(0));
      const georaster = await parseGeoraster(bytes.buffer);
      const { width, height, values, xmin, xmax, ymin, ymax } = georaster;

      // Pre-render all pixels to a canvas once — zoom/pan is then instant (static image)
      const canvas  = document.createElement('canvas');
      canvas.width  = width;
      canvas.height = height;
      const ctx       = canvas.getContext('2d');
      const imageData = ctx.createImageData(width, height);
      const px        = imageData.data;
      const rawBand   = values[0];
      const colorFn   = getColorFn(data);

      // georaster returns rows as Int16Array/Float32Array (TypedArray, not plain Array)
      // Array.isArray() misses TypedArrays — use typeof instead
      const is2D = typeof rawBand[0] !== 'number';
      if (is2D) {
        for (let row = 0; row < height; row++) {
          const rowData = rawBand[row];
          for (let col = 0; col < width; col++) {
            const rgba = colorFn(rowData[col]);
            if (rgba) {
              const j = (row * width + col) << 2;
              px[j] = rgba[0]; px[j+1] = rgba[1]; px[j+2] = rgba[2]; px[j+3] = rgba[3];
            }
          }
        }
      } else {
        for (let i = 0; i < rawBand.length; i++) {
          const rgba = colorFn(rawBand[i]);
          if (rgba) {
            const j = i << 2;
            px[j] = rgba[0]; px[j+1] = rgba[1]; px[j+2] = rgba[2]; px[j+3] = rgba[3];
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);

      // Convert EPSG:3857 extent to LatLng for Leaflet imageOverlay
      const sw = L.CRS.EPSG3857.unproject(L.point(xmin, ymin));
      const ne = L.CRS.EPSG3857.unproject(L.point(xmax, ymax));

      _currentRasterOverlay = L.imageOverlay(canvas.toDataURL(), [sw, ne])
        .addTo(map.value);
      const imgEl = _currentRasterOverlay.getElement();
      if (imgEl) {
        imgEl.style.imageRendering = 'pixelated';
        imgEl.style.imageRendering = '-moz-crisp-edges';
        imgEl.style.imageRendering = 'crisp-edges';
      }
    };

    Vue.watch(() => props.mapData, async (newVal) => {
      await loadMapData();
    });

    Vue.watch(() => props.overlayGeoJSON, (geojson) => {
      if (!map.value) return;
      if (gbf2Layer.value) {
        map.value.removeLayer(gbf2Layer.value);
        gbf2Layer.value = null;
      }
      if (geojson) {
        gbf2Layer.value = L.geoJSON(geojson, {
          style: { color: '#555', weight: 1.5, fillColor: '#666', fillOpacity: 0.35, opacity: 0.7 }
        }).addTo(map.value);
      }
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
        'CartoDB': 'OSM',
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

    // Colorbar for float map layers — derived from min_max embedded in mapData
    const colorbarInfo = computed(() => {
      const data = props.mapData;
      if (!data || data.intOrFloat !== 'float') return null;
      const [minVal, maxVal] = data.min_max || [0, 1];

      // Format a number compactly; use exponential notation for very small values to avoid
      // long strings like "-0.000300" that clutter the colorbar.
      const fmt = (v) => {
        if (v === 0) return '0';
        const abs = Math.abs(v);
        if (abs >= 1e6) return (v / 1e6).toPrecision(3) + 'M';
        if (abs >= 1e3) return (v / 1e3).toPrecision(3) + 'k';
        if (abs >= 1) return v.toPrecision(4);
        if (abs >= 0.01) return v.toPrecision(3);
        return v.toExponential(1);  // e.g. -3.0e-4 for very small values
      };

      // Build a CSS gradient that mirrors the COLORS_FLOAT ramp used during rendering:
      //   codes 1–50  → dark navy (#000E2B) to light blue (#5D94DC)  [negative range]
      //   codes 51–100 → pale yellow (#FFF6B0) to dark red (#800026) [positive range]
      let gradient;
      // displayZeroFrac: fraction of the bar at the zero crossing (null = one-sided).
      // Clamped to MIN_BLUE_FRAC so the blue section is always wide enough for the min label.
      let displayZeroFrac = null;
      const MIN_BLUE_FRAC = 0.15;

      if (minVal >= 0) {
        // Positive-only: yellow → orange → dark red
        gradient = 'linear-gradient(to right, #FFF6B0, #FD933E, #800026)';
      } else if (maxVal <= 0) {
        // Negative-only: dark blue → lighter blue
        gradient = 'linear-gradient(to right, #000E2B, #104991, #5D94DC)';
      } else {
        // Bipolar: clamp the blue section to a minimum visual width for readability
        const rawFrac = Math.abs(minVal) / (Math.abs(minVal) + Math.abs(maxVal));
        const clamped = rawFrac < MIN_BLUE_FRAC;
        displayZeroFrac = clamped ? MIN_BLUE_FRAC : rawFrac;
        const pct = (displayZeroFrac * 100).toFixed(1);
        gradient = `linear-gradient(to right, #000E2B 0%, #5D94DC ${pct}%, #FFF6B0 ${pct}%, #800026 100%)`;
        // Hide the '0' tick when clamped: its position is artificial and would overlap min label
        if (clamped) displayZeroFrac = null;
      }

      return { minVal, maxVal, gradient, displayZeroFrac, fmtMin: fmt(minVal), fmtMax: fmt(maxVal) };
    });

    // Categorical legend for integer map layers — look up by legendKey in the shared registry
    const intLegend = computed(() => {
      const data = props.mapData;
      if (!data || data.intOrFloat !== 'int') return null;
      const legendKey = data.legendKey;
      if (!legendKey) return null;
      const reg = window.legend_registry?.[legendKey];
      if (!reg) return null;
      const legend = reg.legend ?? reg;   // .legend sub-key (new format); fall back to flat (old)
      if (!legend || Object.keys(legend).length === 0) return null;
      return legend;
    });

    const isExporting = ref(false);

    async function exportLayer() {
      if (!props.mapData?.tif_b64) return;
      isExporting.value = true;
      await new Promise(resolve => setTimeout(resolve, 50)); // let UI update before heavy atob
      const bytes = Uint8Array.from(atob(props.mapData.tif_b64), c => c.charCodeAt(0));
      const blob  = new Blob([bytes], { type: 'image/tiff' });
      const url   = URL.createObjectURL(blob);
      const a     = Object.assign(document.createElement('a'), {
        href: url, download: `luto_layer_${Date.now()}.tif`
      });
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      isExporting.value = false;
    }

    return {
      selectedRegion,
      updateMap,
      selectedBaseMap,
      changeBaseMap,
      baseMapOptions,
      handleBaseMapChange,
      colorbarInfo,
      intLegend,
      isExporting,
      exportLayer,
      canExport: computed(() => !!props.mapData?.tif_b64),
    };
  },
  template: `
    <div class="bg-white h-screen flex flex-col">
    
      <!-- Map Container with Controls Overlay - Base map selector and map element -->
      <div class="bg-white shadow-lg flex-1 relative">

        <!-- Base Map Selector + Export button row -->
        <div class="absolute top-[40px] left-[20px] z-[9999] flex items-center gap-2">
          <div class="bg-white/70 rounded-lg shadow-lg" style="min-width: 150px;">
            <filterable-dropdown
              :use-search="false"
              :items="baseMapOptions"
              :selected-value="selectedBaseMap"
              placeholder="Select base map"
              @change="handleBaseMapChange"
            />
          </div>

          <!-- Export GeoTIFF button -->
          <button
            @click="exportLayer"
            :disabled="!canExport || isExporting"
            class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg shadow-lg text-[0.72rem] font-medium transition-all select-none"
            :class="canExport && !isExporting
              ? 'bg-sky-500 text-white hover:bg-sky-600 cursor-pointer'
              : 'bg-white/70 text-gray-400 cursor-not-allowed'">
            <span v-if="isExporting" class="flex items-center gap-1.5">
              <svg class="animate-spin h-3.5 w-3.5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
              Downloading…
            </span>
            <span v-else class="flex items-center gap-1">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m0 0l-4-4m4 4l4-4"/>
              </svg>
              Export GeoTIFF
            </span>
          </button>
        </div>

        <!-- Map Container - Leaflet map will be initialized here -->
        <div id="map" class="w-full h-full relative z-10"></div>

        <!-- Categorical (integer) legend — shown for DVAR / land-use map layers -->
        <div v-if="intLegend && showLegend"
             v-draggable
             class="absolute right-[20px] z-[1001] bg-white/80 p-2 rounded-lg shadow"
             style="top: 50%; transform: translateY(-50%); max-width: 220px; max-height: 60vh; overflow-y: auto;">
          <div class="flex flex-col space-y-0.5">
            <div v-for="(color, label) in intLegend" :key="label" class="flex items-center gap-1.5">
              <span class="inline-block flex-shrink-0 w-3 h-3 rounded-sm" :style="{ backgroundColor: color }"></span>
              <span class="text-[0.6rem] text-gray-700 leading-tight">{{ label }}</span>
            </div>
          </div>
        </div>

        <!-- Float colorbar legend — shown only for continuous (non-categorical) map layers -->
        <div v-if="colorbarInfo"
             v-draggable
             class="absolute bottom-[28px] left-1/2 z-[1001] bg-white/80 px-3 py-2 rounded-lg shadow"
             style="transform: translateX(-50%); min-width: 260px; max-width: 380px;">
          <!-- Header row: unit label -->
          <div class="text-[0.6rem] text-gray-400 text-right mb-0.5 italic">per ha</div>
          <!-- Gradient bar -->
          <div class="h-3 w-full rounded" :style="{ background: colorbarInfo.gradient }"></div>
          <!-- Tick labels -->
          <div class="relative text-[0.65rem] text-gray-600 mt-1" style="height: 1.1em;">
            <!-- Min label — always at left -->
            <span class="absolute left-0">{{ colorbarInfo.fmtMin }}</span>
            <!-- Max label — always at right -->
            <span class="absolute right-0">{{ colorbarInfo.fmtMax }}</span>
          </div>
        </div>

      </div>
    </div>
  `
};