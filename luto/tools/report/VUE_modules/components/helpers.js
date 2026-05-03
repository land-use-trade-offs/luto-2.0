
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

window.loadScriptWithTracking = (src, name, viewName) => {
    return new Promise((resolve, reject) => {
        // Always create new script element for aggressive cleanup approach
        const script = document.createElement("script");
        script.src = src;
        document.head.appendChild(script);

        // Register both data and script with the view for aggressive cleanup
        if (viewName) {
            window.MemoryService.registerViewData(viewName, [name]);
            window.MemoryService.registerViewScript(viewName, src, script);
        }

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

    async function ensureComboLayer(layerPrefix, comboValues) {
        const key = `${layerPrefix}||${comboValues.join('||')}`;
        if (loadedComboKey.value === key && currentLayerData.value) return;

        const varName  = `${layerPrefix}__${comboValues.map(_safeKey).join('__')}`;
        const filePath = `data/map_layers/${varName}.js`;

        // Release previous combo data before loading (allows GC)
        currentLayerData.value = null;
        await window.loadScriptWithTracking(filePath, varName, viewName);
        currentLayerData.value = window[varName] ?? null;
        delete window[varName];   // Release global pin for GC
        loadedComboKey.value = key;
    }

    return { currentLayerData, loadedComboKey, ensureComboLayer };
};
