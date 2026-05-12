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

// Patch Leaflet overlay/grid classes to guard _animateZoom and _reset against
// the remove-during-animation race.  After removeLayer() nulls this._map,
// in-flight RAFs and setTimeout callbacks still fire — the null-guard makes
// them no-ops.  Covers: ImageOverlay, DivOverlay (Popup/Tooltip), GridLayer.
;(function() {
    // Patch every class whose prototype defines one of these methods, including
    // subclasses that override the parent's implementation (e.g. Tooltip / Popup
    // override DivOverlay._animateZoom; Marker overrides Layer; SVG / Canvas
    // override Renderer). Patching a parent does NOT cover overridden methods
    // on subclasses, so we walk the class list explicitly.
    const classes = [
        L.ImageOverlay, L.DivOverlay, L.Popup, L.Tooltip,
        L.GridLayer, L.TileLayer,
        L.Marker, L.Path, L.Polyline, L.Polygon, L.Circle, L.CircleMarker,
        L.Renderer, L.SVG, L.Canvas,
    ].filter(Boolean);
    const methods = ['_animateZoom', '_reset', '_updateTransform', '_update', '_updatePath'];
    classes.forEach(function(cls) {
        methods.forEach(function(method) {
            if (!Object.prototype.hasOwnProperty.call(cls.prototype, method)) return;
            const orig = cls.prototype[method];
            cls.prototype[method] = function() { if (this._map) return orig.apply(this, arguments); };
        });
    });
})();

// Write a minimal Classic TIFF (Little-Endian) with correct raw-byte handling
// for multi-byte sample types (Int16 / Float32). The bundled geotiff.js
// writeArrayBuffer is broken for non-uint8 data — it does element-wise
// new Uint8Array(typedArray) conversion (1 byte per element) instead of reading
// the raw buffer, producing 4× horizontal tiling for Float32 and 2× for Int16.
function buildGeoTiff({ width, height, bitsPerSample, sampleFormat,
                        westLng, northLat, lngPerPx, latPerPx, rawData, nodata }) {
    const LE = true;
    // Layout: header(8) | image(W×H×BPS/8) | mpScale(24) | tiepoint(48) | geoKeys(32) | nodataStr | IFD
    const imageOffset    = 8;
    const imageByteCount = rawData.byteLength;
    const mpScaleOffset  = imageOffset    + imageByteCount;
    const tieOffset      = mpScaleOffset  + 24;   // 3 × float64
    const geoKeyOffset   = tieOffset      + 48;   // 6 × float64
    const nodataStr      = `${nodata}\0`;
    const nodataOffset   = geoKeyOffset   + 32;   // 16 × uint16
    const ifdOffset      = nodataOffset   + nodataStr.length;

    // IFD entries sorted by tag number (TIFF spec requirement)
    // Inline rule: count × typeSize ≤ 4 → value stored inline, else offset.
    // Types: 2=ASCII(1B), 3=SHORT(2B), 4=LONG(4B), 12=DOUBLE(8B)
    const tags = [
        { tag: 256,   type: 4,  count: 1,                  inline: width },
        { tag: 257,   type: 4,  count: 1,                  inline: height },
        { tag: 258,   type: 3,  count: 1,                  inline: bitsPerSample },
        { tag: 259,   type: 3,  count: 1,                  inline: 1 },              // no compression
        { tag: 262,   type: 3,  count: 1,                  inline: 1 },              // BlackIsZero
        { tag: 273,   type: 4,  count: 1,                  inline: imageOffset },    // StripOffsets
        { tag: 277,   type: 3,  count: 1,                  inline: 1 },              // SamplesPerPixel
        { tag: 278,   type: 4,  count: 1,                  inline: height },         // RowsPerStrip
        { tag: 279,   type: 4,  count: 1,                  inline: imageByteCount }, // StripByteCounts
        { tag: 284,   type: 3,  count: 1,                  inline: 1 },              // PlanarConfiguration
        { tag: 339,   type: 3,  count: 1,                  inline: sampleFormat },   // SampleFormat
        { tag: 33550, type: 12, count: 3,                  offset: mpScaleOffset },  // ModelPixelScale
        { tag: 33922, type: 12, count: 6,                  offset: tieOffset },      // ModelTiepoint
        { tag: 34735, type: 3,  count: 16,                 offset: geoKeyOffset },   // GeoKeyDirectory
        { tag: 42113, type: 2,  count: nodataStr.length,   offset: nodataOffset },   // GDAL_NODATA
    ];

    const buf  = new ArrayBuffer(ifdOffset + 2 + tags.length * 12 + 4);
    const view = new DataView(buf);

    // TIFF header: 'II' + magic 42 + IFD offset
    view.setUint8(0, 0x49); view.setUint8(1, 0x49);
    view.setUint16(2, 42, LE);
    view.setUint32(4, ifdOffset, LE);

    // Image data — raw bytes preserve full precision for Int16 / Float32
    new Uint8Array(buf, imageOffset, imageByteCount).set(rawData);

    // ModelPixelScale: [lngPerPx, latPerPx, 0]
    view.setFloat64(mpScaleOffset,      lngPerPx, LE);
    view.setFloat64(mpScaleOffset +  8, latPerPx, LE);
    view.setFloat64(mpScaleOffset + 16, 0,        LE);

    // ModelTiepoint: [i=0, j=0, k=0, x=westLng, y=northLat, z=0]
    view.setFloat64(tieOffset,      0,        LE);
    view.setFloat64(tieOffset +  8, 0,        LE);
    view.setFloat64(tieOffset + 16, 0,        LE);
    view.setFloat64(tieOffset + 24, westLng,  LE);
    view.setFloat64(tieOffset + 32, northLat, LE);
    view.setFloat64(tieOffset + 40, 0,        LE);

    // GeoKeyDirectory: GTModelType=2 (Geographic), GTRasterType=1 (PixelIsArea), GeographicType=4283 (GDA94)
    new Uint16Array(buf, geoKeyOffset, 16).set([
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 1,
        2048, 0, 1, 4283,
    ]);

    // GDAL_NODATA ASCII string
    const ndArr = new Uint8Array(buf, nodataOffset, nodataStr.length);
    for (let i = 0; i < nodataStr.length; i++) ndArr[i] = nodataStr.charCodeAt(i);

    // IFD
    let p = ifdOffset;
    view.setUint16(p, tags.length, LE); p += 2;
    for (const { tag, type, count, inline, offset } of tags) {
        view.setUint16(p, tag,   LE); p += 2;
        view.setUint16(p, type,  LE); p += 2;
        view.setUint32(p, count, LE); p += 4;
        if (inline !== undefined) {
            if (type === 3) { view.setUint16(p, inline, LE); view.setUint16(p + 2, 0, LE); }
            else            { view.setUint32(p, inline, LE); }
        } else {
            view.setUint32(p, offset, LE);
        }
        p += 4;
    }
    view.setUint32(p, 0, LE); // next IFD = none
    return buf;
}

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
    },
    fileName: {
      type: String,
      default: ''
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

      // Close any open popup before zoom animation starts — prevents the
      // null-_map race where Popup._animateZoom fires after removeLayer().
      map.value.on('zoomstart', () => map.value.closePopup());

      // Zoom state machine: track animation in-flight and flush deferred layer swaps
      // once the animation completes.  Raster layer remove/add during zoom causes
      // _animateZoom to fire on a null-_map layer — deferring eliminates the race.
      map.value.on('zoomanim', () => { _isZoomAnimating = true; });
      map.value.on('zoomend',  () => {
        _isZoomAnimating = false;
        if (_pendingLayerSwap) {
          const fn = _pendingLayerSwap;
          _pendingLayerSwap = null;
          fn();
        }
      });

      // Click-to-inspect: show raw cell value at clicked location
      map.value.on('click', (e) => {
        if (!_currentRasterInfo) return;
        const { rawBand, is2D, xmin, xmax, ymin, ymax, width, height, data } = _currentRasterInfo;

        // Convert click lat/lng → EPSG:3857 metres → pixel col/row
        const pt  = L.CRS.EPSG3857.project(e.latlng);
        const col = Math.round((pt.x - xmin) / (xmax - xmin) * (width  - 1));
        const row = Math.round((ymax - pt.y)  / (ymax - ymin) * (height - 1));

        if (col < 0 || col >= width || row < 0 || row >= height) return;

        const raw = is2D ? rawBand[row][col] : rawBand[row * width + col];
        const NODATA = -9999;

        let content;
        if (raw === null || raw === NODATA) {
          content = '<span style="color:#888">No data</span>';
        } else if (data.intOrFloat === 'int') {
          const label = window.legend_registry?.[data.legendKey]?.legend?.[Math.round(raw)];
          content = label
            ? `<b>${label}</b><br><span style="color:#888">code ${Math.round(raw)}</span>`
            : `<b>${Math.round(raw)}</b>`;
        } else {
          const fmt = v => Math.abs(v) >= 1e4 ? v.toExponential(3)
                         : Math.abs(v) >= 1    ? v.toPrecision(5)
                         :                       v.toPrecision(4);
          content = `<b>${fmt(raw)}</b>`;
        }

        L.popup({ closeButton: true, className: 'luto-value-popup' })
          .setLatLng(e.latlng)
          .setContent(content)
          .openOn(map.value);
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

      // Cancel any pending region timeout from a previous rapid call
      if (_pendingRegionTimeout !== null) {
        clearTimeout(_pendingRegionTimeout);
        _pendingRegionTimeout = null;
      }
      const mySeq = ++_updateSeq;

      // Snapshot region now — selectedRegion.value may change before the callbacks fire
      const regionSnapshot = selectedRegion.value;

      // Fade out existing elements first
      fadeOutExistingElements().then(() => {
        if (mySeq !== _updateSeq) return;   // superseded by a newer updateMap call

        // Remove existing rectangles after fade out
        if (boundingBox.value) {
          map.value.removeLayer(boundingBox.value);
          boundingBox.value = null;
        }

        // Calculate bounds for smooth transition
        const centroidBbox = props.regionType === 'STATE' ? window.AUS_STATE_centroid_bbox : window.NRM_AUS_centroid_bbox;
        const bbox = centroidBbox?.[regionSnapshot]?.bounding_box;
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
        _pendingRegionTimeout = setTimeout(() => {
          _pendingRegionTimeout = null;
          if (mySeq !== _updateSeq) return;   // superseded while waiting
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
    const addRegionLayer = (region = selectedRegion.value) => {
      // Skip adding region overlay for AUSTRALIA
      if (region === 'AUSTRALIA') {
        return;
      }

      // Find the actual region feature from the appropriate GeoJSON data
      const geoJsonData = props.regionType === 'STATE' ? window.AUS_STATE : window.NRM_AUS;
      const regionProp = props.regionType === 'STATE' ? 'STATE_NAME' : 'NRM_REGION';
      const regionLayer = geoJsonData?.features.find(feature =>
        feature.properties[regionProp] === region
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
    let _currentRasterInfo   = null;  // { rawBand, is2D, xmin, xmax, ymin, ymax, width, height, data }
    let _loadSeq = 0;
    let _updateSeq = 0;          // guards updateMap against rapid region changes
    let _pendingRegionTimeout = null;
    let _isZoomAnimating = false;  // true while Leaflet zoom animation is in flight
    let _pendingLayerSwap = null;  // deferred raster swap queued during zoom animation

    const loadMapData = async () => {
      const mySeq = ++_loadSeq;

      const data = props.mapData;
      if (!data?.tif_b64 || !data?.intOrFloat) {
        if (_currentRasterOverlay) {
          map.value.removeLayer(_currentRasterOverlay);
          _currentRasterOverlay = null;
        }
        _currentRasterInfo = null;
        return;
      }

      // Decode base64 → parse GeoTIFF (already EPSG:3857, no reprojection needed)
      const bytes     = Uint8Array.from(atob(data.tif_b64), c => c.charCodeAt(0));
      const georaster = await parseGeoraster(bytes.buffer);

      // Discard if a newer load was triggered while we were awaiting parseGeoraster
      if (mySeq !== _loadSeq) return;

      const { width, height, values, xmin, xmax, ymin, ymax } = georaster;
      const rawBand = values[0];
      const is2D    = typeof rawBand[0] !== 'number';
      _currentRasterInfo = { rawBand, is2D, xmin, xmax, ymin, ymax, width, height, data };

      // Pre-render all pixels to a canvas once — zoom/pan is then instant (static image)
      const canvas  = document.createElement('canvas');
      canvas.width  = width;
      canvas.height = height;
      const ctx       = canvas.getContext('2d');
      const imageData = ctx.createImageData(width, height);
      const px        = imageData.data;
      const colorFn   = getColorFn(data);
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

      // Discard if superseded during the pixel loop
      if (mySeq !== _loadSeq) return;

      // Convert EPSG:3857 extent to LatLng for Leaflet imageOverlay
      const sw = L.CRS.EPSG3857.unproject(L.point(xmin, ymin));
      const ne = L.CRS.EPSG3857.unproject(L.point(xmax, ymax));

      // Build the new overlay before swapping — eliminates blank-map flash
      const newOverlay = L.imageOverlay(canvas.toDataURL(), [sw, ne]);

      // Defer the actual layer swap if a Leaflet zoom animation is in flight.
      // Removing a layer mid-animation leaves _animateZoom firing on a null-_map
      // layer, causing the "Cannot read properties of null" crash.
      const applySwap = () => {
        if (_currentRasterOverlay) {
          map.value.removeLayer(_currentRasterOverlay);
        }
        _currentRasterOverlay = newOverlay.addTo(map.value);
        const imgEl = _currentRasterOverlay.getElement();
        if (imgEl) {
          imgEl.style.imageRendering = 'pixelated';
          imgEl.style.imageRendering = '-moz-crisp-edges';
          imgEl.style.imageRendering = 'crisp-edges';
        }
      };

      if (_isZoomAnimating) {
        // Queue swap for after zoom ends; capture mySeq so a stale deferred call is dropped.
        _pendingLayerSwap = () => { if (mySeq === _loadSeq) applySwap(); };
      } else {
        applySwap();
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
      await new Promise(resolve => setTimeout(resolve, 50)); // let UI update before heavy work

      try {
        const data      = props.mapData;
        const bytes     = Uint8Array.from(atob(data.tif_b64), c => c.charCodeAt(0));
        const georaster = await parseGeoraster(bytes.buffer);
        const { width, height, values, xmin, xmax, ymin, ymax, noDataValue } = georaster;

        const rawBand = values[0];
        const is2D    = typeof rawBand[0] !== 'number';

        // EPSG:3857 ↔ GDA94 (EPSG:4283) share the same ellipsoid — no datum shift.
        // Inverse Mercator: metres → degrees
        const R          = 6378137;
        const RAD        = Math.PI / 180;
        const merc2lng   = x => x / R / RAD;
        const merc2lat   = y => (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) / RAD;
        // Forward Mercator: degrees → metres
        const lng2merc   = lng => lng * RAD * R;
        const lat2merc   = lat => R * Math.log(Math.tan(Math.PI / 4 + lat * RAD / 2));

        // Output grid corners in GDA94 degrees
        const westLng  = merc2lng(xmin);
        const eastLng  = merc2lng(xmax);
        const southLat = merc2lat(ymin);
        const northLat = merc2lat(ymax);
        const lngPerPx = (eastLng - westLng)   / width;
        const latPerPx = (northLat - southLat)  / height;

        // Precompute source column for each output column — O(w) forward-Mercator calls
        const srcXfrac = new Float64Array(width);
        for (let col = 0; col < width; col++) {
          const xMerc   = lng2merc(westLng + (col + 0.5) * lngPerPx);
          srcXfrac[col] = (xMerc - xmin) / (xmax - xmin) * (width - 1);
        }

        // Precompute source row for each output row — O(h) forward-Mercator calls
        const srcYrow = new Int32Array(height);
        for (let row = 0; row < height; row++) {
          const yMerc   = lat2merc(northLat - (row + 0.5) * latPerPx);
          srcYrow[row]  = Math.round((ymax - yMerc) / (ymax - ymin) * (height - 1));
        }

        // Nearest-neighbour resample into the GDA94 output grid
        const NODATA  = -9999;
        const isInt   = data.intOrFloat === 'int';
        const outData = isInt
          ? new Int16Array(width * height).fill(NODATA)
          : new Float32Array(width * height).fill(NODATA);

        for (let row = 0; row < height; row++) {
          const srcRow = srcYrow[row];
          if (srcRow < 0 || srcRow >= height) continue;
          for (let col = 0; col < width; col++) {
            const srcCol = Math.round(srcXfrac[col]);
            if (srcCol < 0 || srcCol >= width) continue;
            const val = is2D ? rawBand[srcRow][srcCol] : rawBand[srcRow * width + srcCol];
            if (val !== null && val !== NODATA && val !== noDataValue)
              outData[row * width + col] = val;
          }
        }

        const buf = buildGeoTiff({
          width, height,
          bitsPerSample: isInt ? 16 : 32,
          sampleFormat:  isInt ? 2  : 3,   // 2=signed int, 3=IEEE float
          westLng, northLat, lngPerPx, latPerPx,
          rawData: new Uint8Array(outData.buffer, outData.byteOffset, outData.byteLength),
          nodata: NODATA,
        });

        const blob = new Blob([buf], { type: 'image/tiff' });
        const url  = URL.createObjectURL(blob);
        const n    = props.fileName || 'luto_layer_gda94';
        const a    = Object.assign(document.createElement('a'), {
          href: url, download: (n.length > 100 ? n.slice(0, 97) + '...' : n) + '.tif'
        });
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } finally {
        isExporting.value = false;
      }
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
              Export GeoTIFF (GDA94)
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