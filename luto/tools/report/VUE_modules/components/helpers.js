
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
            const timeout = 5000;
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
            const timeout = 5000;
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

