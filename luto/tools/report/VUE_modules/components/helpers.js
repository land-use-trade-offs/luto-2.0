
// Expose view state to window._debug for console debugging
// Usage: window._debug.Home.ChartData.value
window._debug = {};

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
