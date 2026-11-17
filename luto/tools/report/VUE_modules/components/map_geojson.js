// Home map: the picture of the run and its region picker. Shows the NRM regions or the states
// (prop `regionLevel`) over an optional raster layer (prop `mapData`, e.g. the land-use map);
// the regions are outlines, so the raster under them shows. A click selects a region, a second click
// on it returns to AUSTRALIA. The map can be dragged and zoomed; the house button fits the country again.
window.map_geojson = {
    name: 'MapGeojson',
    props: {
        height: {
            type: String,
            default: '500px',
        },
        regionLevel: {
            type: String,
            default: 'NRM',          // 'NRM', 'State', 'River Region' or 'Drainage Division'
        },
        // A map layer leaf {cells_b64, intOrFloat, legendKey, ...}, drawn under the regions
        mapData: {
            type: Object,
            default: null,
        },
        // px kept clear of the country at the left / top / bottom of the map, for controls laid over it
        padLeft: {
            type: Number,
            default: 8,
        },
        padTop: {
            type: Number,
            default: 8,
        },
        padBottom: {
            type: Number,
            default: 8,
        },
        // The place names over the map
        showLabels: {
            type: Boolean,
            default: true,
        },
    },
    setup(props) {
        const { ref, onMounted, onUnmounted, watch, nextTick, inject } = Vue;

        const mapElement = ref(null);
        const mapInstance = ref(null);
        const activeRegionName = inject('globalSelectedRegion');
        const hoverTooltip = ref(null);
        const geoJSONLayer = ref(null);
        const australiaBounds = L.latLngBounds(window.AUSTRALIA_VIEW.bounds)         // helpers.js;
        let resizeObserver = null;
        let rasterOverlay = null;
        let rasterLoadCount = 0;

        // The whole country, as large as the element allows
        const fitView = () => {
            const map = mapInstance.value;
            if (!map) return;
            map.invalidateSize();
            map.fitBounds(australiaBounds, { animate: false, paddingTopLeft: [props.padLeft, props.padTop], paddingBottomRight: [8, props.padBottom] });
        };

        // The boundaries of each level and the property that names a region in them
        const LEVELS = {
            'NRM': { data: 'NRM_AUS', nameField: 'NRM_REGION' },
            'State': { data: 'AUS_STATE', nameField: 'STATE_NAME' },
            // the water regions of the run (settings.WATER_REGION_DEF); Home loads the file before it asks for the level
            'River Region': { data: 'WATER_RIVREG', nameField: 'NAME' },
            'Drainage Division': { data: 'WATER_DRAINDIV', nameField: 'NAME' },
        };

        // Outlines over a raster: the selected region gets a heavy outline, the raster stays visible
        const outlineStyle = {
            color: "#4b5563",
            fillColor: "#ffffff",
            fillOpacity: 0,
            weight: 0.5,
        };

        // The selected region: a thin dark line over a soft white halo (a wider pale line under it,
        // `haloStyle`), which reads over any colour of the raster without a heavy border
        const outlineHighlightStyle = {
            color: "#1f2937",
            fillColor: "#ffffff",
            fillOpacity: 0,
            weight: 1.5,
        };
        const MASK_OPACITY = 0.7;
        const haloStyle = { color: "#ffffff", opacity: 0.75, weight: 5.5, fill: false, interactive: false };
        let haloLayer = null;

        // The style of a region now
        const styleOf = (regionName) => {
            if (regionName === activeRegionName.value) return outlineHighlightStyle;
            // With a region selected, every other region goes under a faint white mask
            return activeRegionName.value === 'AUSTRALIA' ? outlineStyle : { ...outlineStyle, fillOpacity: MASK_OPACITY };
        };

        const restyle = () => {
            const map = mapInstance.value;
            if (!geoJSONLayer.value || !map) return;
            if (haloLayer) { map.removeLayer(haloLayer); haloLayer = null; }
            geoJSONLayer.value.eachLayer(layer => {
                const isSelected = layer.options.regionName === activeRegionName.value;
                layer.setStyle(styleOf(layer.options.regionName));
                if (!isSelected) return;
                // the halo first, then the region's own line on top of it
                haloLayer = L.geoJSON(layer.feature, { style: haloStyle, interactive: false, smoothFactor: 0.3 }).addTo(map);
                layer.bringToFront();
            });
        };

        const removeTooltip = () => {
            if (hoverTooltip.value) {
                mapInstance.value.removeLayer(hoverTooltip.value);
                hoverTooltip.value = null;
            }
        };

        // (Re)draw the regions of the current level
        const buildLayer = () => {
            const map = mapInstance.value;
            const level = LEVELS[props.regionLevel] || LEVELS['NRM'];
            if (!map || !window[level.data]) return;

            removeTooltip();
            if (geoJSONLayer.value) map.removeLayer(geoJSONLayer.value);

            geoJSONLayer.value = Vue.markRaw(L.geoJSON(window[level.data], {
                // Leaflet thins an outline's points at every zoom (1 px by default); less of it keeps the shape steady from one zoom to the next
                smoothFactor: 0.3,
                style: (feature) => styleOf(feature.properties[level.nameField]),
                onEachFeature: (feature, layer) => {
                    const regionName = feature.properties[level.nameField];
                    layer.options.regionName = regionName;

                    layer.on({
                        mousemove: (e) => {
                            removeTooltip();
                            hoverTooltip.value = Vue.markRaw(L.tooltip({ permanent: false, direction: "top" })
                                .setContent(regionName)
                                .setLatLng(e.latlng)
                                .addTo(map));
                        },
                        mouseout: (e) => {
                            removeTooltip();
                        },
                        click: () => {
                            // A second click on the selected region goes back to the whole country
                            activeRegionName.value = regionName === activeRegionName.value ? 'AUSTRALIA' : regionName;
                        },
                    });
                },
            }).addTo(map));
            restyle();
        };

        const removeRaster = () => {
            if (rasterOverlay) {
                mapInstance.value.removeLayer(rasterOverlay);
                rasterOverlay = null;
            }
        };

        // Draw the raster layer of `mapData` under the regions (the GeoTIFF is in EPSG:3857 already)
        const drawRaster = async () => {
            const map = mapInstance.value;
            const loadCount = ++rasterLoadCount;
            const data = props.mapData;
            if (!map) return;
            if (!window.MapCells.hasData(data) || !data?.intOrFloat) { removeRaster(); return; }

            const georaster = await window.MapCells.decode(data);          // helpers.js
            if (loadCount !== rasterLoadCount) return;            // a newer layer was asked for meanwhile

            const { width, height, values, xmin, xmax, ymin, ymax } = georaster;
            const band = values[0];
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');
            const imageData = context.createImageData(width, height);
            const pixels = imageData.data;
            const colorFn = getColorFn(data);                      // regions_map.js
            const isRows = typeof band[0] !== 'number';
            for (let row = 0; row < height; row++) {
                for (let col = 0; col < width; col++) {
                    const rgba = colorFn(isRows ? band[row][col] : band[row * width + col]);
                    if (!rgba) continue;
                    const offset = (row * width + col) << 2;
                    pixels[offset] = rgba[0]; pixels[offset + 1] = rgba[1]; pixels[offset + 2] = rgba[2]; pixels[offset + 3] = rgba[3];
                }
            }
            context.putImageData(imageData, 0, 0);

            const southWest = L.CRS.EPSG3857.unproject(L.point(xmin, ymin));
            const northEast = L.CRS.EPSG3857.unproject(L.point(xmax, ymax));
            const newOverlay = L.imageOverlay(canvas.toDataURL(), [southWest, northEast], { interactive: false });
            removeRaster();
            rasterOverlay = newOverlay.addTo(map);
            rasterOverlay.bringToBack();
            const imageElement = rasterOverlay.getElement();
            if (imageElement) imageElement.style.imageRendering = 'pixelated';
        };

        onMounted(() => {

            // The map opens on the whole country and can be dragged and zoomed
            const map = L.map(mapElement.value, {
                zoomControl: false,
                attributionControl: false,
                zoomSnap: 0.05,
                zoomDelta: 0.5,
                // The region outlines are drawn on a canvas: one bitmap, scaled with the basemap tiles while a zoom
                // animates. As SVG paths (Leaflet's default) the browser redrew ~60 detailed outlines every frame
                // of the animation, and they lagged behind the basemap. The padding keeps a zoom out inside the bitmap.
                renderer: L.canvas({ padding: 1 }),
                minZoom: 3,
                maxZoom: 10,
            });
            L.control.zoom({ position: 'topright' }).addTo(map);

            // Back to the whole country, under the zoom buttons
            const ResetControl = L.Control.extend({
                options: { position: 'topright' },
                onAdd() {
                    const container = L.DomUtil.create('div', 'leaflet-bar');
                    const link = L.DomUtil.create('a', '', container);
                    link.href = '#';
                    link.title = 'Whole country';
                    link.setAttribute('role', 'button');
                    link.innerHTML = '&#8962;';
                    link.style.fontSize = '18px';
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.on(link, 'click', (event) => { L.DomEvent.preventDefault(event); fitView(); });
                    return container;
                },
            });
            new ResetControl().addTo(map);
            map.setView(window.AUSTRALIA_VIEW.center, 3.9, { animate: false });
            // markRaw: Vue must not wrap a Leaflet object in a reactive proxy. Through a proxy Leaflet's own
            // bookkeeping goes wrong on a zoom (the selected region's outline was left at the old scale).
            mapInstance.value = Vue.markRaw(map);

            // Basemap in two layers (the same free public CARTO key as the Map View, embedded on purpose):
            // the land and sea without labels at the bottom, and the place names in their own pane ABOVE
            // the raster and the region outlines, so a map layer never covers them. The pane takes no
            // pointer events: a click still reaches the region under a label.
            L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png?key=cb1_2lje_1_eb930ec7809f7b2ebec6a65f', {
                subdomains: 'abcd',
                maxZoom: 20,
            }).addTo(map);
            map.createPane('labels');
            map.getPane('labels').style.zIndex = 450;
            map.getPane('labels').style.pointerEvents = 'none';
            map.getPane('labels').style.display = props.showLabels ? '' : 'none';
            L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png?key=cb1_2lje_1_eb930ec7809f7b2ebec6a65f', {
                subdomains: 'abcd',
                maxZoom: 20,
                pane: 'labels',
            }).addTo(map);
            L.control.attribution({ prefix: false, position: 'bottomright' })
                .addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
                .addTo(map);

            buildLayer();
            drawRaster();

            // Refit when the element takes its size, and whenever that changes (window, sidebar)
            resizeObserver = new ResizeObserver(fitView);
            resizeObserver.observe(mapElement.value);
        });

        onUnmounted(() => { if (resizeObserver) resizeObserver.disconnect(); });

        watch(() => props.showLabels, (show) => {
            const pane = mapInstance.value?.getPane('labels');
            if (pane) pane.style.display = show ? '' : 'none';
        });
        watch(() => props.regionLevel, buildLayer);
        watch(() => props.mapData, drawRaster);
        watch(() => [props.padLeft, props.padTop, props.padBottom], fitView);
        watch(activeRegionName, restyle);                 // also when the region is set from outside the map

        // Watch for height changes and update map size
        watch(
            () => props.height,
            async () => {
                if (mapInstance.value) {
                    await nextTick();
                    setTimeout(fitView, 50);
                }
            },
            { immediate: false }
        );

        return {
            mapElement,
            activeRegionName,
            props,
        };
    },
    template: /*html*/`
      <div>
        <div ref="mapElement" :style="{ background: 'transparent', height: props.height + ' !important', width: '100%' }"></div>
      </div>
    `,
};
