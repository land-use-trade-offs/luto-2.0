window.TransitionView = {
  name: 'TransitionView',
  setup() {
    const { ref, computed, onMounted, onUnmounted, inject, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["Transition"];
    const mapRegister = window.MapService.mapCategories["Transition"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Transition";

    // Global region state
    const selectRegion = inject("globalSelectedRegion");

    // Category (always "Area")
    const availableCategories = ["Area"];
    const selectCategory = ref("Area");

    // SubCat: derived from map registry keys
    const availableSubCats = Object.keys(mapRegister);
    const selectSubCat = ref(availableSubCats[0] || "Ag2Ag");

    // Year state (index-driven slider)
    const yearIndex = ref(0);
    const selectYear = ref("2020");
    const availableYears = ref([]);

    // Water selection state
    const availableFromWater = ref([]);
    const availableToWater = ref([]);
    const selectFromWater = ref("ALL");
    const selectToWater = ref("ALL");

    // UI state
    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    // Cell selection for heatmap-driven map filter: { xi, yi } or null
    const selectedCell = ref(null);

    // ----------------------------------------------------------------
    // Computed
    // ----------------------------------------------------------------
    const currentMapName = computed(() => mapRegister[selectSubCat.value]?.["name"] || "");
    const currentChartName = computed(() => chartRegister[selectSubCat.value]?.["name"] || "");

    // Map leaf: From-water -> To-water -> from_lu -> to_lu -> year
    // When a cell is selected in the preview heatmap, filter to that from/to LU pair.
    const selectMapData = computed(() => {
      if (!dataLoaded.value || !currentMapName.value) return {};
      const mapData = window[currentMapName.value];
      const cell = selectedCell.value;
      const leaf = selectChartLeaf.value;
      const fromLu = (cell && leaf) ? (leaf.y_categories[cell.yi] || 'ALL') : 'ALL';
      const toLu = (cell && leaf) ? (leaf.x_categories[cell.xi] || 'ALL').replace(/<br>/g, ' ') : 'ALL';
      // Try specific LU pair, fall back to ALL→ALL
      return mapData?.[selectFromWater.value]?.[selectToWater.value]?.[fromLu]?.[toLu]?.[selectYear.value]
        || mapData?.[selectFromWater.value]?.[selectToWater.value]?.['ALL']?.['ALL']?.[selectYear.value]
        || {};
    });

    // Chart leaf: region -> from_water -> to_water -> year
    const selectChartLeaf = computed(() => {
      if (!dataLoaded.value || !currentChartName.value) return null;
      const chartData = window[currentChartName.value];
      return chartData?.[selectRegion.value]?.[selectFromWater.value]?.[selectToWater.value]?.[selectYear.value] || null;
    });

    // Preview data: when a cell is selected, dim all other non-null cells.
    const DIM_COLOR = 'rgba(210,210,210,0.45)';
    const previewChartData = computed(() => {
      const leaf = selectChartLeaf.value;
      if (!leaf) return [];
      const cell = selectedCell.value;
      if (!cell) return leaf.data;
      return leaf.data.map(p => {
        const xi = Array.isArray(p) ? p[0] : p.x;
        const yi = Array.isArray(p) ? p[1] : p.y;
        const val = Array.isArray(p) ? p[2] : p.value;
        if (xi === cell.xi && yi === cell.yi) {
          // Keep selected cell using colorAxis (array form)
          return Array.isArray(p) ? p : [xi, yi, val];
        }
        // Dim everything else
        return { x: xi, y: yi, value: val, color: DIM_COLOR };
      });
    });

    // ----------------------------------------------------------------
    // Load data for current subcat
    // ----------------------------------------------------------------
    const loadSubCatData = async (subCat) => {
      const mapEntry = mapRegister[subCat];
      const chartEntry = chartRegister[subCat];
      if (!mapEntry || !chartEntry) return;

      isLoadingData.value = true;
      dataLoaded.value = false;
      await Promise.all([
        loadScript(mapEntry["path"], mapEntry["name"], VIEW_NAME),
        loadScript(chartEntry["path"], chartEntry["name"], VIEW_NAME),
      ]);
      isLoadingData.value = false;

      // Derive water options from chart data (AUSTRALIA has all combinations)
      const chartData = window[chartEntry["name"]];
      const refRegion = 'AUSTRALIA';
      availableFromWater.value = Object.keys(chartData?.[refRegion] || {});
      selectFromWater.value = availableFromWater.value.includes("ALL") ? "ALL" : (availableFromWater.value[0] || "ALL");

      availableToWater.value = Object.keys(chartData?.[refRegion]?.[selectFromWater.value] || {});
      selectToWater.value = availableToWater.value.includes("ALL") ? "ALL" : (availableToWater.value[0] || "ALL");

      await nextTick(() => { dataLoaded.value = true; });
    };

    // ----------------------------------------------------------------
    // Memory cleanup
    // ----------------------------------------------------------------
    onUnmounted(() => {
      window.MemoryService.cleanupViewData(VIEW_NAME);
    });

    // ----------------------------------------------------------------
    // Mount
    // ----------------------------------------------------------------
    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      availableYears.value = window.Supporting_info.years.map(String);
      selectYear.value = availableYears.value[0] || "2020";
      yearIndex.value = 0;
      await loadSubCatData(selectSubCat.value);
    });

    // ----------------------------------------------------------------
    // Watchers
    // ----------------------------------------------------------------
    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };

    // Cell click handler for compact preview heatmap
    const nullMessage = ref(null);  // timed message for NaN cells
    let _nullMsgTimer = null;
    const handlePreviewClick = ({ xi, yi, value }) => {
      // Ignore clicks on null/NaN cells
      if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
        if (_nullMsgTimer) clearTimeout(_nullMsgTimer);
        nullMessage.value = 'This transition does not exist';
        _nullMsgTimer = setTimeout(() => { nullMessage.value = null; }, 2500);
        return;
      }
      nullMessage.value = null;
      const cell = selectedCell.value;
      if (cell && cell.xi === xi && cell.yi === yi) {
        selectedCell.value = null;  // toggle off
      } else {
        selectedCell.value = { xi, yi };
      }
    };

    // Clear selection when water/year/subcat changes
    watch([selectFromWater, selectToWater, selectSubCat, yearIndex], () => {
      selectedCell.value = null;
    });

    watch(yearIndex, (newIdx) => {
      selectYear.value = availableYears.value[newIdx];
    });

    watch(selectSubCat, (newSubCat) => {
      loadSubCatData(newSubCat);
    });

    watch(selectFromWater, (newFW) => {
      const chartData = window[currentChartName.value];
      const refRegion = 'AUSTRALIA';
      availableToWater.value = Object.keys(chartData?.[refRegion]?.[newFW] || {});
      if (!availableToWater.value.includes(selectToWater.value)) {
        selectToWater.value = availableToWater.value.includes("ALL") ? "ALL" : (availableToWater.value[0] || "ALL");
      }
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears,
      availableCategories, selectCategory,
      availableSubCats, selectSubCat,
      availableFromWater, availableToWater,
      selectFromWater, selectToWater,
      selectMapData, selectChartLeaf, previewChartData,
      selectedCell, handlePreviewClick, nullMessage,
      dataLoaded, isLoadingData,
      isDrawerOpen, toggleDrawer,
    };
    window._debug = window._debug || {};
    window._debug[VIEW_NAME] = _state;
    return _state;
  },

  template: /*html*/`
    <div class="relative w-full h-screen">

      <!-- Region dropdown -->
      <div class="absolute w-[262px] top-32 left-[20px] z-[9999] bg-white/70 rounded-lg shadow-lg">
        <filterable-dropdown region-type="NRM"></filterable-dropdown>
      </div>

      <!-- Year slider -->
      <div class="absolute top-[200px] left-[20px] z-[1001] w-[262px] bg-white/70 p-2 rounded-lg">
        <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
        <el-slider
          v-if="availableYears && availableYears.length > 0"
          v-model="yearIndex"
          size="small"
          :min="0"
          :max="availableYears.length - 1"
          :step="1"
          :format-tooltip="index => availableYears[index]"
          :show-stops="true"
          @input="(index) => { yearIndex = index; selectYear = availableYears[index]; }"
        />
      </div>

      <!-- Filters + heatmap preview column -->
      <div class="absolute top-[285px] left-[20px] w-[262px] z-[1001] flex flex-col gap-2">

        <!-- Filters box -->
        <div class="flex flex-col space-y-2 bg-white/70 p-2 rounded-lg">

          <!-- Category buttons -->
          <div class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
            <button v-for="val in availableCategories" :key="val"
              @click="selectCategory = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectCategory === val}">
              {{ val }}
            </button>
          </div>

          <!-- SubCat buttons -->
          <div class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">Sub-Category:</span>
            <button v-for="val in availableSubCats" :key="val"
              @click="selectSubCat = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectSubCat === val}">
              {{ val }}
            </button>
          </div>

          <!-- From-water buttons -->
          <div v-if="dataLoaded && availableFromWater.length > 0" class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">From Water:</span>
            <button v-for="val in availableFromWater" :key="val"
              @click="selectFromWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectFromWater === val}">
              {{ val }}
            </button>
          </div>

          <!-- To-water buttons -->
          <div v-if="dataLoaded && availableToWater.length > 0" class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">To Water:</span>
            <button v-for="val in availableToWater" :key="val"
              @click="selectToWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectToWater === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Independent heatmap preview (click cell to filter map) -->
        <div style="width:350px; display:flex; flex-direction:column; gap:4px;">
          <!-- Heatmap -->
          <div class="bg-white/90 rounded-lg" style="width:350px; height:350px; overflow:hidden; position:relative;">
            <div v-if="!dataLoaded || isLoadingData"
              class="flex items-center justify-center h-full text-gray-400 text-xs">
              Loading...
            </div>
            <div v-else-if="!selectChartLeaf || selectChartLeaf.data.length === 0"
              class="flex items-center justify-center h-full text-gray-400 text-xs">
              No transitions for this selection
            </div>
            <heatmap-container
              v-else
              :xCats="selectChartLeaf.x_categories"
              :yCats="selectChartLeaf.y_categories"
              :data="previewChartData"
              :maxVal="selectChartLeaf.max_val"
              :show-axis-labels="false"
              :show-data-labels="false"
              :on-cell-click="handlePreviewClick"
              style="width:100%; height:100%;">
            </heatmap-container>
          </div>
          <!-- Selection / error label below the heatmap -->
          <div style="min-height:1.4rem; text-align:center; font-size:0.6rem; line-height:1.3;">
            <span v-if="nullMessage" class="text-amber-600">{{ nullMessage }}</span>
            <span v-else-if="selectedCell && selectChartLeaf" class="text-gray-600">
              {{ selectChartLeaf.y_categories[selectedCell.yi] }} → {{ selectChartLeaf.x_categories[selectedCell.xi] }}
              <span class="text-sky-500 ml-1">(click again to clear)</span>
            </span>
          </div>
        </div>
      </div>

      <!-- Map + slide-out drawer -->
      <div style="position:relative; width:100%; height:100%; overflow:hidden;">

        <!-- Loading overlay -->
        <div v-if="isLoadingData"
          class="absolute inset-0 z-[2000] flex items-center justify-center bg-white/60 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-2 text-gray-600 text-sm font-medium">
            <svg class="animate-spin h-8 w-8 text-sky-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            Loading transition data...
          </div>
        </div>

        <!-- Map -->
        <regions-map
          :mapData="selectMapData"
          region-type="STATE"
          style="width:100%; height:100%;">
        </regions-map>

        <!-- Drawer toggle button -->
        <button
          @click="toggleDrawer"
          class="absolute top-5 z-[1001] p-2.5 bg-white border border-gray-300 rounded cursor-pointer transition-all duration-300 ease-in-out"
          :class="isDrawerOpen ? 'right-[66%]' : 'right-5'">
          {{ isDrawerOpen ? '\u2192' : '\u2190' }}
        </button>

        <!-- Heatmap drawer -->
        <div :style="{
          position: 'absolute',
          top: '10px',
          bottom: '10px',
          right: isDrawerOpen ? '0px' : '-100%',
          width: '66.666%',
          background: 'transparent',
          transition: 'right 0.3s ease',
          zIndex: 1000,
          padding: '60px 20px 20px 20px',
          boxSizing: 'border-box',
        }">
          <div v-if="dataLoaded && (!selectChartLeaf || selectChartLeaf.data.length === 0)"
            class="flex items-center justify-center h-full text-gray-400 text-sm bg-white/80 rounded-lg">
            No transitions for {{ selectRegion }} - {{ selectFromWater }} to {{ selectToWater }} - {{ selectYear }}
          </div>
          <heatmap-container
            v-else-if="dataLoaded && selectChartLeaf"
            :xCats="selectChartLeaf.x_categories"
            :yCats="selectChartLeaf.y_categories"
            :data="selectChartLeaf.data"
            :maxVal="selectChartLeaf.max_val"
            :exportable="true"
            :zoomable="true"
            :draggable="true"
            style="width:100%; height:calc(100% - 20px);">
          </heatmap-container>
        </div>
      </div>

    </div>
  `,
};