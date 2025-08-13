
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

window.loadDataset = async (datasetName, timeout = 5000) => {
    try {
        // Load only the dataset script, not Chart_default_options
        // This prevents Chart_default_options from being reloaded and potentially overwriting custom options
        await loadScript(`./data/${datasetName}.js`, datasetName);

        // Wait until the dataset is available in the window object or timeout occurs
        const start = Date.now();
        while (!window[datasetName]) {
            if (Date.now() - start > timeout) {
                throw new Error(`Timeout waiting for dataset ${datasetName}`);
            }
            // Pause briefly to allow the dataset to load
            await new Promise((resolve) => setTimeout(resolve, 10));
        }
        return window[datasetName];
    } catch (error) {
        console.error(`Error loading dataset ${datasetName}:`, error);
    }
};
