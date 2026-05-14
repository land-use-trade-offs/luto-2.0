window.TransitionView = {
  name: 'TransitionView',
  setup() {
    const { ref, computed, onMounted, onUnmounted, inject, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Transition"];
    const mapRegister = window.MapService.mapCategories["Transition"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Transition";

    const selectRegion = inject("globalSelectedRegion");

    const availableCategories = Object.keys(mapRegister);
    const selectCategory = ref(availableCategories[0] || "Area");

    const availableSubCats = computed(() => Object.keys(mapRegister[selectCategory.value] || {}));
    const selectSubCat = ref("");

    const yearIndex = ref(0);
    const selectYear = ref("");
    const availableYears = ref([]);

    const availableFromWater = ref([]);
    const availableToWater = ref([]);
    const selectFromWater = ref("ALL");
    const selectToWater = ref("ALL");

    const availableCostTypes = ref([]);
    const selectCostType = ref("ALL");

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    const selectedCell = ref(null);

    const isAreaMode = computed(() => selectCategory.value === "Area");
    const currentChartName = computed(() => chartRegister[selectCategory.value]?.[selectSubCat.value]?.["name"] || "");

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    // Chart leaf — area mode: region→from_water→to_water→year
    //              cost mode: region→cost_type→year
    const selectChartLeaf = computed(() => {
      if (!dataLoaded.value || !currentChartName.value) return null;
      const chartData = window[currentChartName.value];
      if (isAreaMode.value) {
        return chartData?.[selectRegion.value]?.[selectFromWater.value]?.[selectToWater.value]?.[selectYear.value] || null;
      } else {
        return chartData?.[selectRegion.value]?.[selectCostType.value]?.[selectYear.value] || null;
      }
    });

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
        if (xi === cell.xi && yi === cell.yi) return Array.isArray(p) ? p : [xi, yi, val];
        return { x: xi, y: yi, value: val, color: DIM_COLOR };
      });
    });

    function _getCombo(cat, fromWater, toWater, costType, fromLu, toLu) {
      if (cat === "Area") return [fromWater, toWater, fromLu, toLu];
      return [costType, fromLu, toLu];
    }

    const _loadSubCatData = async (cat, subCat) => {
      const mapEntry = mapRegister[cat]?.[subCat];
      const chartEntry = chartRegister[cat]?.[subCat];
      if (!mapEntry || !chartEntry) return;

      isLoadingData.value = true;
      dataLoaded.value = false;
      window.MemoryService.cleanupViewData(VIEW_NAME);

      await Promise.all([
        loadScript(mapEntry.indexPath, mapEntry.indexName, VIEW_NAME),
        loadScript(chartEntry.path, chartEntry.name, VIEW_NAME),
      ]);
      isLoadingData.value = false;

      const chartData = window[chartEntry.name];
      const refRegion = 'AUSTRALIA';
      let defaultCombo;

      if (cat === "Area") {
        availableFromWater.value = Object.keys(chartData?.[refRegion] || {});
        selectFromWater.value = availableFromWater.value.includes("ALL") ? "ALL" : (availableFromWater.value[0] || "ALL");
        availableToWater.value = Object.keys(chartData?.[refRegion]?.[selectFromWater.value] || {});
        selectToWater.value = availableToWater.value.includes("ALL") ? "ALL" : (availableToWater.value[0] || "ALL");
        availableYears.value = Object.keys(chartData?.[refRegion]?.[selectFromWater.value]?.[selectToWater.value] || {}).sort();
        defaultCombo = [selectFromWater.value, selectToWater.value, 'ALL', 'ALL'];
      } else {
        availableCostTypes.value = Object.keys(chartData?.[refRegion] || {});
        selectCostType.value = availableCostTypes.value.includes("ALL") ? "ALL" : (availableCostTypes.value[0] || "ALL");
        availableYears.value = Object.keys(chartData?.[refRegion]?.[selectCostType.value] || {}).sort();
        defaultCombo = [selectCostType.value, 'ALL', 'ALL'];
      }

      yearIndex.value = 0;
      selectYear.value = availableYears.value[0] || "";
      selectedCell.value = null;

      await ensureComboLayer(mapEntry.layerPrefix, defaultCombo);
      dataLoaded.value = true;
    };

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    onMounted(async () => {
      selectSubCat.value = availableSubCats.value[0];
      if (selectSubCat.value) {
        await _loadSubCatData(selectCategory.value, selectSubCat.value);
      }
    });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };

    const nullMessage = ref(null);
    let _nullMsgTimer = null;
    const handlePreviewClick = async ({ xi, yi, value }) => {
      if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
        if (_nullMsgTimer) clearTimeout(_nullMsgTimer);
        nullMessage.value = 'This transition does not exist';
        _nullMsgTimer = setTimeout(() => { nullMessage.value = null; }, 2500);
        return;
      }
      nullMessage.value = null;
      const cell = selectedCell.value;
      const newCell = (cell && cell.xi === xi && cell.yi === yi) ? null : { xi, yi };
      selectedCell.value = newCell;

      const leaf = selectChartLeaf.value;
      const fromLu = (newCell && leaf) ? (leaf.y_categories[newCell.yi] || 'ALL') : 'ALL';
      const toLu = (newCell && leaf) ? ((leaf.x_categories[newCell.xi] || 'ALL').replace(/<br>/g, ' ')) : 'ALL';

      const mapEntry = mapRegister[selectCategory.value]?.[selectSubCat.value];
      if (!mapEntry) return;
      await ensureComboLayer(mapEntry.layerPrefix, _getCombo(selectCategory.value, selectFromWater.value, selectToWater.value, selectCostType.value, fromLu, toLu));
    };

    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });

    watch(selectCategory, (newCat) => {
      const subs = Object.keys(mapRegister[newCat] || {});
      selectSubCat.value = subs[0];
      if (subs[0]) _loadSubCatData(newCat, subs[0]);
    });

    watch(selectSubCat, (newSubCat) => {
      _loadSubCatData(selectCategory.value, newSubCat);
    });

    watch(selectFromWater, async (newFW) => {
      if (!isAreaMode.value || !selectSubCat.value || !dataLoaded.value) return;
      const chartData = window[currentChartName.value];
      availableToWater.value = Object.keys(chartData?.['AUSTRALIA']?.[newFW] || {});
      if (!availableToWater.value.includes(selectToWater.value)) {
        selectToWater.value = availableToWater.value.includes("ALL") ? "ALL" : (availableToWater.value[0] || "ALL");
      }
      selectedCell.value = null;
      const mapEntry = mapRegister["Area"]?.[selectSubCat.value];
      if (mapEntry) await ensureComboLayer(mapEntry.layerPrefix, [newFW, selectToWater.value, 'ALL', 'ALL']);
    });

    watch(selectToWater, async (newTW) => {
      if (!isAreaMode.value || !selectSubCat.value || !dataLoaded.value) return;
      selectedCell.value = null;
      const mapEntry = mapRegister["Area"]?.[selectSubCat.value];
      if (mapEntry) await ensureComboLayer(mapEntry.layerPrefix, [selectFromWater.value, newTW, 'ALL', 'ALL']);
    });

    watch(selectCostType, async (ct) => {
      if (isAreaMode.value || !selectSubCat.value || !dataLoaded.value) return;
      selectedCell.value = null;
      const mapEntry = mapRegister["Cost"]?.[selectSubCat.value];
      if (mapEntry) await ensureComboLayer(mapEntry.layerPrefix, [ct, 'ALL', 'ALL']);
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears,
      availableCategories, selectCategory,
      availableSubCats, selectSubCat,
      isAreaMode,
      availableFromWater, availableToWater, selectFromWater, selectToWater,
      availableCostTypes, selectCostType,
      selectMapData, selectChartLeaf, previewChartData,
      selectedCell, handlePreviewClick, nullMessage,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
    };
    const _fn = v => String(v).trim().replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    _state.mapFileName = computed(() =>
      [VIEW_NAME, selectCategory.value, selectSubCat.value, selectYear.value]
        .filter(Boolean).map(_fn).filter(Boolean).join('__')
    );
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

          <!-- Category buttons (Area / Cost) -->
          <div class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
            <button v-for="val in availableCategories" :key="val"
              @click="selectCategory = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectCategory === val}">
              {{ val }}
            </button>
          </div>

          <!-- SubCat buttons (scoped by category) -->
          <div class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">Sub-Category:</span>
            <button v-for="val in availableSubCats" :key="val"
              @click="selectSubCat = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectSubCat === val}">
              {{ val }}
            </button>
          </div>

          <!-- Area mode: From-water buttons -->
          <div v-if="isAreaMode && dataLoaded && availableFromWater.length > 0" class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">From Water:</span>
            <button v-for="val in availableFromWater" :key="val"
              @click="selectFromWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectFromWater === val}">
              {{ val }}
            </button>
          </div>

          <!-- Area mode: To-water buttons -->
          <div v-if="isAreaMode && dataLoaded && availableToWater.length > 0" class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">To Water:</span>
            <button v-for="val in availableToWater" :key="val"
              @click="selectToWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectToWater === val}">
              {{ val }}
            </button>
          </div>

          <!-- Cost mode: Cost-type buttons -->
          <div v-if="!isAreaMode && dataLoaded && availableCostTypes.length > 0" class="flex flex-wrap gap-1">
            <span class="text-[0.8rem] mr-1 font-medium">Cost Type:</span>
            <button v-for="val in availableCostTypes" :key="val"
              @click="selectCostType = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectCostType === val}">
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
              :value-type="selectCategory"
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
          :file-name="mapFileName"
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
            No transitions for this selection
          </div>
          <heatmap-container
            v-else-if="dataLoaded && selectChartLeaf"
            :xCats="selectChartLeaf.x_categories"
            :yCats="selectChartLeaf.y_categories"
            :data="selectChartLeaf.data"
            :maxVal="selectChartLeaf.max_val"
            :value-type="selectCategory"
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