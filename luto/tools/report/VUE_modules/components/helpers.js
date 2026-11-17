
// Expose view state to window._debug for console debugging
// Usage: window._debug.Home.ChartData.value
window._debug = {};

// The view of the whole country every map of the report opens on: the box the Home map fits, and its
// centre (the Mercator centre of the box, so the country sits in the middle of the map element)
window.AUSTRALIA_VIEW = {
    bounds: [[-43.8, 112.8], [-10.5, 153.8]],
    center: [-28.4, 133.3],
};

window.loadScript = (src, name) => {
    return new Promise((resolve, reject) => {
        const existingScript = document.querySelector(`script[src="${src}"]`);

        if (existingScript && window[name]) {
            resolve();
            return;
        }

        const script = document.createElement("script");
        script.src = src;
        document.head.appendChild(script);

        script.onload = async () => {
            const timeout = 60000;
            const startTime = Date.now();
            while (!window[name]) {
                if (Date.now() - startTime > timeout) {
                    reject(new Error(`Global variable ${name} not available within timeout`));
                    return;
                }
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            resolve();
        };

        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    });
};

// One load per src while it is in flight: the watchers of one click often ask for the same file.
const _scriptInflight = new Map();

window.loadScriptWithTracking = (src, name, viewName) => {
    // The view owns the data either way, so leaving it releases the data.
    if (viewName) window.MemoryService.registerViewData(viewName, [name]);

    // Already loaded (by the sidebar, Home, or an earlier call): no second fetch and parse.
    if (window[name]) return Promise.resolve();
    if (_scriptInflight.has(src)) return _scriptInflight.get(src);

    const promise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = src;
        document.head.appendChild(script);
        if (viewName) window.MemoryService.registerViewScript(viewName, src, script);

        script.onload = async () => {
            const timeout = 60000;
            const startTime = Date.now();
            while (!window[name]) {
                if (Date.now() - startTime > timeout) {
                    reject(new Error(`Global variable ${name} not available within timeout`));
                    return;
                }
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            resolve();
        };

        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    }).finally(() => _scriptInflight.delete(src));

    _scriptInflight.set(src, promise);
    return promise;
};


/**
 * Factory that creates a reactive per-combo map-layer loader for a view.
 *
 * Each call to `ensureComboLayer(layerPrefix, comboValues)` loads the JS file
 * named `<layerPrefix>__<safe(dim1)>__...__<safe(dimN)>.js`, stores the
 * year-keyed layer data in `currentLayerData`, and releases the previous
 * combo's data for garbage collection.
 *
 * Usage in a view:
 *   const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);
 *   // selectMapData: computed(() => currentLayerData.value?.[selectYear.value] ?? {})
 *
 * @param {string} viewName  Passed to loadScriptWithTracking for memory tracking.
 * @returns {{ currentLayerData: Ref<object|null>, loadedComboKey: Ref<string|null>, ensureComboLayer: Function }}
 */
window.createMapLayerLoader = function(viewName) {
    const { ref } = Vue;
    const _safeKey = (s) => String(s).replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');

    const currentLayerData = ref(null);
    const loadedComboKey   = ref(null);

    // A view's cascade watchers often ask for the same combo several times in one tick, and a
    // slow file can land after the user has moved on. One load per key is shared (`inflight`),
    // and only the most recently requested key (`latestKey`) may set the layer.
    const inflight = new Map();
    let latestKey  = null;

    async function ensureComboLayer(layerPrefix, comboValues, pageRange = null) {
        // pageRange: [start, end] for paged SNES/ECNES/NVIS layers; null for standard layers.
        // The var name never includes the page so each new page overwrites the window variable,
        // keeping only the current page in memory.
        const pageKey = pageRange ? `||page_${pageRange[0]}_${pageRange[1]}` : '';
        const key     = `${layerPrefix}||${comboValues.join('||')}${pageKey}`;
        latestKey = key;
        if (loadedComboKey.value === key && currentLayerData.value) return;

        const varName  = `${layerPrefix}__${comboValues.map(_safeKey).join('__')}`;
        const filename = pageRange
            ? `${varName}_${pageRange[0]}_${pageRange[1]}.js`
            : `${varName}.js`;

        const filePath = `data/map_layers/${filename}`;
        currentLayerData.value = null;
        if (!inflight.has(key)) {
            inflight.set(key, (async () => {
                try {
                    await window.loadScriptWithTracking(filePath, varName, viewName);
                    return window[varName] ?? {};
                } catch (err) {
                    // A combo listed in the index with no file: show an empty map, not a stuck one.
                    console.warn(`Map layer not loaded: ${filePath}`, err);
                    return {};
                } finally {
                    delete window[varName];
                    inflight.delete(key);
                }
            })());
        }
        const layer = await inflight.get(key);
        if (latestKey !== key) return;
        currentLayerData.value = layer;
        loadedComboKey.value = key;
    }

    return { currentLayerData, loadedComboKey, ensureComboLayer };
};


/**
 * Layer toggle for the last row of selector buttons (land use / commodity).
 *
 * A click on another button selects it and shows its map layer; a click on the selected
 * button switches its layer off, and the next one switches it back on. A hidden layer's
 * button loses its fill and gets a faint dashed border. The chart is not affected.
 *
 * Usage (in a view's setup()):
 *   const { layerOn, clickLayer, layerButtonClass, layerButtonTitle } =
 *     window.createLayerToggle(selectLanduse, [currentLayerData]);
 *   // selectMapData: computed(() => layerOn.value ? (currentLayerData.value?.[year] ?? {}) : {})
 *   // button: @click="clickLayer(val)" :title="layerButtonTitle(val)" :class="layerButtonClass(val)"
 *
 * @param {Ref<string>} selected  The view's selection ref for that row.
 * @param {Ref[]} showAgainOn     Refs whose change shows the layer again (the loaded layer data:
 *                                any other choice loads a new layer).
 */
window.createLayerToggle = function(selected, showAgainOn = []) {
    const layerOn = Vue.ref(true);

    const clickLayer = (val) => {
        if (selected.value === val) { layerOn.value = !layerOn.value; }
        else { selected.value = val; layerOn.value = true; }
    };
    if (showAgainOn.length > 0) Vue.watch(showAgainOn, () => { layerOn.value = true; });

    const layerButtonClass = (val) => selected.value !== val ? 'border-transparent'
        : (layerOn.value ? 'bg-sky-500 text-white border-transparent' : 'border-dashed border-sky-400');
    const layerButtonTitle = (val) => selected.value !== val ? ''
        : (layerOn.value ? 'Click to hide this layer' : 'Layer hidden: click to show it');

    return { layerOn, clickLayer, layerButtonClass, layerButtonTitle };
};


/**
 * Decode a map layer leaf into a raster the map components draw, read and export.
 *
 * A layer is stored as a 1D vector of the model's cells (`cells_b64`: deflated, base64), not as a picture:
 * the report holds ONE raster of cell numbers (data/map_layers/map_cell_index.js; the input layers, which have
 * a resolution of their own, a second one named by the leaf's `cellIndex`), and a layer is painted by
 * looking every pixel's cell up in the vector. Int layers hold int16 codes; float layers hold one signed byte
 * per cell: 0 = no value, ±1 … ±levels = the value as a share of the layer's range (raw_min_max), restored here
 * to within one level, so a value read off the map or exported is good for LOOKING AT, not for analysis.
 *
 * Returns { width, height, values: [band], xmin, xmax, ymin, ymax, noDataValue }, `band` a flat typed array
 * in row order (the shape parseGeoraster returns, which a report written before 2026-11-12 still goes through:
 * its leaves hold `tif_b64`).
 */
window.MapCells = (() => {
    const NODATA = -9999;
    const indexPromises = {};          // name of the cell-number raster -> its promise

    const inflate = async (b64) => {
        const bytes = Uint8Array.from(atob(b64), char => char.charCodeAt(0));
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
        return new Response(stream).arrayBuffer();
    };

    // The cell-number raster, read once and kept for the life of the page
    const loadIndex = (name = 'map_cell_index') => {
        if (!indexPromises[name]) {
            indexPromises[name] = (async () => {
                await window.loadScriptWithTracking(`data/map_layers/${name}.js`, name);
                const meta = window[name];
                return { ...meta, index: new Int32Array(await inflate(meta.index_b64)) };
            })();
            indexPromises[name].catch(() => { delete indexPromises[name]; });
        }
        return indexPromises[name];
    };

    const hasData = (leaf) => !!(leaf?.cells_b64 || leaf?.tif_b64);

    const decode = async (leaf) => {
        if (leaf.tif_b64) {
            const bytes = Uint8Array.from(atob(leaf.tif_b64), char => char.charCodeAt(0));
            return parseGeoraster(bytes.buffer);
        }
        const [grid, buffer] = await Promise.all([loadIndex(leaf.cellIndex), inflate(leaf.cells_b64)]);
        const { width, height, index, levels, nodata } = grid;
        const isInt = leaf.intOrFloat === 'int';
        const cells = isInt ? new Int16Array(buffer) : new Int8Array(buffer);
        const band = isInt ? new Int16Array(width * height) : new Float32Array(width * height);
        const [low, high] = leaf.raw_min_max ?? [0, 1];
        for (let pixel = 0; pixel < band.length; pixel++) {
            const cell = index[pixel];
            if (cell < 0) { band[pixel] = NODATA; continue; }
            const stored = cells[cell];
            if (isInt) band[pixel] = stored;
            else if (stored === nodata) band[pixel] = NODATA;
            else band[pixel] = stored > 0 ? stored / levels * high : stored < 0 ? -stored / levels * low : 0;
        }
        return { width, height, values: [band], xmin: grid.xmin, xmax: grid.xmax, ymin: grid.ymin, ymax: grid.ymax, noDataValue: NODATA };
    };

    return { decode, hasData, NODATA };
})();
