window.DataConstructor = class DataConstructor {
    constructor() {
        this.leafPaths = new Map();
        this.data = {};
    }

    /**
     * Load data and extract all leaf paths
     * @param {Object} dataObject - The hierarchical data object
     */
    loadData(dataObject) {
        this.data = dataObject;
        this.leafPaths.clear();
        this._extractLeafPaths(dataObject, []);
    }

    /**
     * Recursively extract all paths to leaf nodes (non-object values)
     * @param {any} obj - Current object being traversed
     * @param {Array} currentPath - Current path from root
     * @private
     */
    _extractLeafPaths(obj, currentPath) {
        if (obj === null || obj === undefined) {
            return;
        }

        // If it's not an object or is an array, treat it as a leaf
        if (typeof obj !== 'object' || Array.isArray(obj)) {
            const pathKey = currentPath.join('.');
            this.leafPaths.set(pathKey, {
                path: [...currentPath],
                value: obj
            });
            return;
        }

        // Recursively traverse object properties
        for (const [key, value] of Object.entries(obj)) {
            this._extractLeafPaths(value, [...currentPath, key]);
        }
    }

    /**
     * Query data using level parameters
     * @param {Object} queryParams - Object with level_1, level_2, etc. parameters
     * @returns {any} - The queried data or null if not found
     */
    query(queryParams) {
        // Build path from query parameters
        const path = this._buildPathFromQuery(queryParams);

        if (path.length === 0) {
            return null;
        }

        // Navigate to the data using the path
        return this._getValueByPath(this.data, path);
    }

    /**
     * Build path array from query parameters
     * @param {Object} queryParams - Query parameters object
     * @returns {Array} - Path array
     * @private
     */
    _buildPathFromQuery(queryParams) {
        const path = [];
        let level = 1;

        while (queryParams[`level_${level}`] !== undefined) {
            const value = queryParams[`level_${level}`];
            if (value !== null && value !== undefined && value !== '') {
                path.push(value);
            }
            level++;
        }

        return path;
    }

    /**
     * Get value by following a path through the object
     * @param {Object} obj - Object to navigate
     * @param {Array} path - Path to follow
     * @returns {any} - Value at path or null if not found
     * @private
     */
    _getValueByPath(obj, path) {
        let current = obj;

        for (const key of path) {
            if (current === null || current === undefined || typeof current !== 'object') {
                return null;
            }

            if (!(key in current)) {
                return null;
            }

            current = current[key];
        }

        return current;
    }

    /**
     * Get all leaf paths for debugging/inspection
     * @returns {Array} - Array of leaf path objects
     */
    getAllLeafPaths() {
        return Array.from(this.leafPaths.values());
    }

    /**
     * Find paths that match a partial query
     * @param {Object} queryParams - Partial query parameters
     * @returns {Array} - Array of matching leaf paths
     */
    findMatchingPaths(queryParams) {
        const queryPath = this._buildPathFromQuery(queryParams);
        const matches = [];

        for (const leafInfo of this.leafPaths.values()) {
            if (this._pathMatches(leafInfo.path, queryPath)) {
                matches.push(leafInfo);
            }
        }

        return matches;
    }

    /**
     * Check if a leaf path matches a query path (prefix match)
     * @param {Array} leafPath - Full path to leaf
     * @param {Array} queryPath - Query path (may be partial)
     * @returns {boolean} - True if matches
     * @private
     */
    _pathMatches(leafPath, queryPath) {
        if (queryPath.length > leafPath.length) {
            return false;
        }

        for (let i = 0; i < queryPath.length; i++) {
            if (leafPath[i] !== queryPath[i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Get available values at a specific level
     * @param {Object} queryParams - Query parameters for previous levels
     * @param {number} targetLevel - Level to get available values for (1-based)
     * @returns {Array} - Array of available values
     */
    getAvailableValues(queryParams, targetLevel) {
        const basePath = this._buildPathFromQuery(queryParams);
        const values = new Set();

        for (const leafInfo of this.leafPaths.values()) {
            if (this._pathMatches(leafInfo.path, basePath) &&
                leafInfo.path.length > targetLevel - 1) {
                values.add(leafInfo.path[targetLevel - 1]);
            }
        }

        return Array.from(values);
    }

    /**
     * Get available keys at the next level given current path
     * @param {Object} fixedLevels - Object with fixed level values (e.g., {level_1: "map_area_Am"})
     * @returns {Array} - Array of available keys for the next level
     */
    getAvailableKeysAtNextLevel(fixedLevels) {
        const currentPath = this._buildPathFromQuery(fixedLevels);
        const nextLevel = currentPath.length + 1;

        return this.getAvailableValues(fixedLevels, nextLevel);
    }

    /**
     * Get available keys at a specific level with a more intuitive interface
     * @param {Object} fixedLevels - Object with fixed level values
     * @param {number} targetLevel - Target level to get keys for (1-based, optional)
     * @returns {Array} - Array of available keys
     */
    getKeysAtLevel(fixedLevels, targetLevel = null) {
        if (targetLevel === null) {
            // If no target level specified, get keys for next level
            return this.getAvailableKeysAtNextLevel(fixedLevels);
        }

        return this.getAvailableValues(fixedLevels, targetLevel);
    }

    /**
     * Navigate through the data structure step by step
     * @param {Object} currentPath - Current path as key-value pairs
     * @returns {Object} - Object with available keys and current data
     */
    explore(currentPath = {}) {
        const pathArray = this._buildPathFromQuery(currentPath);
        const currentData = this._getValueByPath(this.data, pathArray);
        const nextLevelKeys = this.getAvailableKeysAtNextLevel(currentPath);

        return {
            currentPath: pathArray,
            currentData: currentData,
            nextLevelKeys: nextLevelKeys,
            isLeaf: nextLevelKeys.length === 0,
            dataType: typeof currentData
        };
    }

    /**
     * Query with flexible parameter names (backward compatibility)
     * @param {Object} params - Parameters object
     * @returns {any} - Queried data
     */
    flexQuery(params) {
        // Convert various parameter formats to level_N format
        const normalizedParams = {};

        // Handle level_1, level_2, etc.
        for (const [key, value] of Object.entries(params)) {
            if (key.startsWith('level_')) {
                normalizedParams[key] = value;
            }
        }

        // Handle positional parameters if level_N not provided
        if (Object.keys(normalizedParams).length === 0) {
            const values = Object.values(params);
            for (let i = 0; i < values.length; i++) {
                normalizedParams[`level_${i + 1}`] = values[i];
            }
        }

        return this.query(normalizedParams);
    }
}