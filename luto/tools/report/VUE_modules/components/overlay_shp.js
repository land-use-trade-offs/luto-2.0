/**
 * Vector overlays for <regions-map>.
 *
 * Everything about an overlay — its label, colours, popup text, legend and whether it
 * draws to canvas — lives in `registry` below. `regions_map.js` knows none of it; it
 * asks for a controller, renders whatever toggle buttons and legends come back, and
 * lets the controller own the Leaflet layer lifecycle.
 *
 * To add an overlay:
 *   1. add an entry to `registry` here;
 *   2. have the view load its GeoJSON and pass `:overlays="[{ id: '<key>', data: <geojson> }]"`.
 *
 * Views pass overlays in the order they should stack — later entries draw on top.
 */
window.OverlayShp = {

  registry: {

    // GBF2 priority degraded areas — a single dissolved mask, always drawn (no toggle)
    // because it defines the area the GBF2 metric is scored over.
    gbf2_mask: {
      label: 'GBF2 Priority Degraded Areas',
      color: '#666666',
      toggleable: false,
      style: () => ({ color: '#555', weight: 1.5, fillColor: '#666', fillOpacity: 0.35, opacity: 0.7 }),
    },

    // AEMO Renewable Energy Zones — a few dozen large polygons.
    rez: {
      label: 'Renewable Energy Zones',
      color: '#f59e0b',
      icon: '<path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>',
      style: () => ({ color: '#f59e0b', weight: 1.5, fillColor: '#f59e0b', fillOpacity: 0.12, opacity: 0.85 }),
    },

    // Past NECMA + GBCMA vegetation works, dissolved to one feature per CMA — parcels
    // overlap where a site was worked more than once, and stacked semi-transparent fills
    // render as dark banding. Still canvas-drawn: each feature is a MultiPolygon with
    // thousands of narrow parts.
    cma_veg: {
      label: 'CMA Vegetation Works',
      color: '#15803d',
      icon: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 21v-8m0 0C12 8 9 4 4 3c0 5 3 9 8 10zm0 0c0-5 3-9 8-10 0 5-3 9-8 10z"/>',
      canvas: true,
      legend: {
        title: 'Past vegetation works',
        items: [
          { color: '#15803d', label: 'NECMA' },
          { color: '#0d9488', label: 'GBCMA' },
        ],
      },
      style: (feature) => {
        const color = feature.properties?.SOURCE === 'GBCMA' ? '#0d9488' : '#15803d';
        return { color, weight: 1, fillColor: color, fillOpacity: 0.55, opacity: 0.9 };
      },
      popup: (feature) => {
        const { SOURCE, N_PARCELS, AREA_HA, YR_MIN, YR_MAX } = feature.properties || {};
        // NECMA supplied no dates at all, so the year range is GBCMA-only.
        const years = (YR_MIN && YR_MAX) ? `${YR_MIN}–${YR_MAX}` : 'no dates supplied';
        return `<div style="font-size:0.72rem"><strong>${SOURCE || 'CMA'} vegetation works</strong>`
          + `<br>${Number(N_PARCELS).toLocaleString()} parcels`
          + `<br>${Number(AREA_HA).toLocaleString()} ha (overlaps merged)`
          + `<br>Completed: ${years}</div>`;
      },
    },

  },

  /**
   * Build a controller that keeps Leaflet layers in step with the overlays a view supplies.
   *
   * @param {Function} getOverlays  () => Array<{id, data}>  — the map's `overlays` prop
   * @param {Function} getMap       () => L.Map | null
   * @returns {{items, toggle, sync}} `items` drives the toggle buttons and legends,
   *          `sync` must be called once the map exists.
   */
  createController(getOverlays, getMap) {
    const { ref, computed, watch } = Vue;

    const hidden = ref({});                 // id -> true when the user has switched it off
    const drawn = new Map();                // id -> { layer, data } currently on the map

    const resolved = computed(() => {
      const out = [];
      for (const entry of getOverlays() || []) {
        if (!entry?.id || !entry.data) continue;
        const def = this.registry[entry.id];
        if (!def) {
          console.warn(`[OverlayShp] no registry entry for overlay "${entry.id}" — skipped`);
          continue;
        }
        out.push({ ...def, id: entry.id, data: entry.data });
      }
      return out;
    });

    const isVisible = (id) => !hidden.value[id];

    const items = computed(() => resolved.value.map((o) => ({
      id: o.id,
      label: o.label,
      color: o.color,
      icon: o.icon || null,
      legend: o.legend || null,
      toggleable: o.toggleable !== false,
      visible: isVisible(o.id),
    })));

    const toggle = (id) => { hidden.value = { ...hidden.value, [id]: !hidden.value[id] }; };

    const sync = () => {
      const map = getMap();
      if (!map) return;
      const wanted = new Map(resolved.value.map((o) => [o.id, o]));

      // Drop layers that are gone, hidden, or whose GeoJSON has been swapped out.
      for (const [id, held] of drawn) {
        const want = wanted.get(id);
        if (!want || !isVisible(id) || want.data !== held.data) {
          map.removeLayer(held.layer);
          drawn.delete(id);
        }
      }

      // Add whatever is missing, in the order the view listed them.
      for (const o of resolved.value) {
        if (drawn.has(o.id) || !isVisible(o.id)) continue;
        const opts = { style: o.style };
        if (o.canvas) opts.renderer = L.canvas({ padding: 0.5 });
        if (o.popup) opts.onEachFeature = (feature, layer) => layer.bindPopup(o.popup(feature));
        drawn.set(o.id, { layer: L.geoJSON(o.data, opts).addTo(map), data: o.data });
      }
    };

    watch([resolved, hidden], sync, { deep: true });

    return { items, toggle, sync };
  },
};
