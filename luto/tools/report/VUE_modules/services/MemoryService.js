window.MemoryService = {
  // Track which view owns which data objects and script elements
  viewDataRegistry: new Map(),
  viewScriptRegistry: new Map(),

  // Register data loaded by a view
  registerViewData(viewName, dataNames) {
    if (!this.viewDataRegistry.has(viewName)) {
      this.viewDataRegistry.set(viewName, new Set());
    }
    dataNames.forEach(name => {
      this.viewDataRegistry.get(viewName).add(name);
    });
  },

  // Register script element loaded by a view
  registerViewScript(viewName, src, scriptElement) {
    if (!this.viewScriptRegistry.has(viewName)) {
      this.viewScriptRegistry.set(viewName, new Map());
    }
    this.viewScriptRegistry.get(viewName).set(src, scriptElement);
  },

  // AGGRESSIVE CLEANUP - Remove ALL data and scripts when leaving a view
  cleanupViewData(viewName) {
    // Clean up data objects
    const dataNames = this.viewDataRegistry.get(viewName);
    if (dataNames) {
      dataNames.forEach(name => {
        delete window[name];
      });
      this.viewDataRegistry.delete(viewName);
    }

    // Clean up script elements
    const scriptMap = this.viewScriptRegistry.get(viewName);
    if (scriptMap) {
      scriptMap.forEach((scriptElement, src) => {
        if (scriptElement && scriptElement.parentNode) {
          scriptElement.parentNode.removeChild(scriptElement);
        }
      });
      this.viewScriptRegistry.delete(viewName);
    }

    // Force garbage collection if available
    if (window.gc) {
      window.gc();
    }
  },

  // Get memory usage info (optional debugging)
  getMemoryInfo() {
    const totalDataObjects = Array.from(this.viewDataRegistry.values())
      .reduce((sum, set) => sum + set.size, 0);
    const totalScriptElements = Array.from(this.viewScriptRegistry.values())
      .reduce((sum, map) => sum + map.size, 0);

    return {
      activeViews: this.viewDataRegistry.size,
      totalDataObjects,
      totalScriptElements
    };
  }
};