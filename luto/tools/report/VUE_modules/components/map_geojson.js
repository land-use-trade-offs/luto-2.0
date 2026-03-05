window.map_geojson = {
    name: 'MapGeojson',
    props: {
        height: {
            type: String,
            default: '500px',
        },
        selectRankingColors: {
            type: Object,
        },
    },
    setup(props) {
        const { ref, onMounted, watch, nextTick, inject } = Vue;

        const mapElement = ref(null);
        const mapInstance = ref(null);
        const activeRegionName = inject('globalSelectedRegion');
        const hoverTooltip = ref(null);
        const geoJSONLayer = ref(null);
        const featureStyles = ref({});
        const australiaBounds = L.latLngBounds([-42, 113], [-12, 154]);

        const defaultStyle = {
            color: "#fefefe",
            fillColor: "#d2d7dd",
            fillOpacity: 0.5,
            weight: 1.5,
        };

        const highlightStyle = {
            color: "#0b0b0b",
            fillColor: "#0b0b0b",
            fillOpacity: 0.5,
            weight: 0.1,
        };

        onMounted(() => {

            // Initialize basic map with disabled controls
            const map = L.map(mapElement.value, {
                zoomControl: false,
                attributionControl: false,
                zoomSnap: 0.1,
                dragging: false,
                scrollWheelZoom: false,
                doubleClickZoom: false,
            });

            // Set view to Australia
            map.setView(australiaBounds.getCenter(), 3.9, { animate: false });

            // Store map instance for later use
            mapInstance.value = map;

            // Get style function for each feature
            const getFeatureStyle = (feature) => {
                const regionName = feature.properties.NRM_REGION;
                let style = { ...defaultStyle };

                if (props.selectRankingColors && props.selectRankingColors[regionName] && typeof props.selectRankingColors[regionName] === 'string') {
                    style.fillColor = props.selectRankingColors[regionName];
                }

                featureStyles.value[regionName] = style;
                return style;
            };

            // Add GeoJSON layer with mouse effects
            geoJSONLayer.value = L.geoJSON(window['NRM_AUS'], {
                style: getFeatureStyle,
                onEachFeature: (feature, layer) => {
                    layer.options.regionName = feature.properties.NRM_REGION;

                    // Set initial style
                    if (activeRegionName.value === feature.properties.NRM_REGION) {
                        layer.setStyle(highlightStyle);
                    }

                    // Mouse events
                    layer.on({
                        mousemove: (e) => {
                            const layer_e = e.target;

                            if (layer_e._path) {
                                layer_e._path.style.cursor = 'default';
                            }

                            // Remove previous tooltip
                            if (hoverTooltip.value) {
                                map.removeLayer(hoverTooltip.value);
                                hoverTooltip.value = null;
                            }

                            // Create hover tooltip
                            hoverTooltip.value = L.tooltip({
                                permanent: false,
                                direction: "top",
                            });
                            hoverTooltip.value.setContent(feature.properties.NRM_REGION);
                            hoverTooltip.value.setLatLng(e.latlng);
                            hoverTooltip.value.addTo(map);
                        },
                        mouseout: (e) => {
                            const layer_e = e.target;

                            // Remove tooltip
                            if (hoverTooltip.value) {
                                map.removeLayer(hoverTooltip.value);
                                hoverTooltip.value = null;
                            }

                            if (layer_e.options.regionName === activeRegionName.value) {
                                layer_e.setStyle(highlightStyle);
                            }
                        },
                        click: (e) => {
                            const layer_e = e.target;

                            // Toggle selection
                            if (layer_e.options.regionName === activeRegionName.value) {
                                // Deselect - restore original style
                                const regionName = layer_e.options.regionName;
                                if (featureStyles.value[regionName]) {
                                    layer_e.setStyle(featureStyles.value[regionName]);
                                } else {
                                    layer_e.setStyle(defaultStyle);
                                }
                                activeRegionName.value = 'AUSTRALIA';
                                return;
                            }

                            // Remove highlight from all regions
                            geoJSONLayer.value.eachLayer(function (layer) {
                                const regionName = layer.options.regionName;
                                if (featureStyles.value[regionName]) {
                                    layer.setStyle(featureStyles.value[regionName]);
                                } else {
                                    layer.setStyle(defaultStyle);
                                }
                            });

                            // Highlight new selection
                            layer_e.setStyle(highlightStyle);
                            activeRegionName.value = layer_e.options.regionName;
                        },
                    });
                },
            }).addTo(map);

            // Ensure map size is correct after initialization
            setTimeout(() => {
                map.invalidateSize();
            }, 100);

        });


        // Function to update map colors when selectRankingColors changes
        const updateMapStyles = () => {
            if (!geoJSONLayer.value) return;

            geoJSONLayer.value.eachLayer(function (layer) {
                const regionName = layer.options.regionName;

                // Skip currently selected region
                if (regionName === activeRegionName.value) return;

                // Update style with new colors
                let style = { ...defaultStyle };
                if (props.selectRankingColors && props.selectRankingColors[regionName] && typeof props.selectRankingColors[regionName] === 'string') {
                    style.fillColor = props.selectRankingColors[regionName];
                }

                featureStyles.value[regionName] = style;
                layer.setStyle(style);
            });
        };

        // Watch for ranking color changes
        watch(() => props.selectRankingColors, updateMapStyles, { deep: true });

        // Watch for height changes and update map size
        watch(
            () => props.height,
            async (newHeight) => {
                if (mapInstance.value) {
                    await nextTick();
                    setTimeout(() => {
                        mapInstance.value.invalidateSize();
                    }, 50);
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
    template: `
      <div>
        <div ref="mapElement" :style="{ background: 'transparent', height: props.height + ' !important', width: '100%' }"></div>
      </div>
    `,
};