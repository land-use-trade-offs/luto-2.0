window.ProductionView = {
  name: 'ProductionView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Production"];
    const mapRegister = window.MapService.mapCategories["Production"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Production";

    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    const availableYears = ref([]);
    const availableUnit = {
      Area: "Hectares", Economics: "AUD", GHG: "Mt CO2e",
      Water: "ML", Production: "Tonnes",
      Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    const availableCategories = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);  // stores commodity values

    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectCommodity = ref("");

    const previousSelections = ref({
      "Sum": { water: "", landuse: "" },
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    const selectChartData = computed(() => {
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const commodity = selectCommodity.value;
      const region = selectRegion.value;
      if (!dataLoaded.value) return {};

      const rawChart = window[chartRegister[cat]?.["name"]];
      if (!rawChart) return {};
      const chartData = rawChart[region];
      let seriesData;

      if (cat === "Sum" || cat === "Ag") {
        seriesData = (chartData?.[water] || []).filter(s => commodity === "ALL" || s.name === commodity);
      } else if (cat === "Ag Mgt") {
        seriesData = (chartData?.[water]?.[commodity] || []).filter(s => agMgt === "ALL" || s.name === agMgt);
      } else if (cat === "Non-Ag") {
        seriesData = (chartData || []).filter(s => commodity === "ALL" || s.name === commodity);
      }

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["Production"] } },
        series: seriesData || [],
      };
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    function getTree(cat) {
      const t = window[mapRegister[cat]?.indexName]?.tree;
      if (t !== undefined) return t;
      return cat === "Non-Ag" ? [] : {};
    }

    async function ensureIndexLoaded(cat) {
      const mapEntry = mapRegister[cat];
      if (mapEntry && !window[mapEntry.indexName]) {
        isLoadingData.value = true;
        await loadScript(mapEntry.indexPath, mapEntry.indexName, VIEW_NAME);
        isLoadingData.value = false;
      }
    }

    async function loadAllCharts() {
      const pending = [];
      for (const entry of Object.values(chartRegister)) {
        if (entry?.name && !window[entry.name])
          pending.push(loadScript(entry.path, entry.name, VIEW_NAME));
      }
      if (pending.length > 0) await Promise.all(pending);
    }

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;

      const initCat = "Sum";
      await Promise.all([ensureIndexLoaded(initCat), loadAllCharts()]);

      // Sum tree: { water: [commodity] }
      const tree = getTree(initCat);
      availableWater.value = Object.keys(tree);
      selectWater.value = availableWater.value[0] || '';
      availableLanduse.value = tree[selectWater.value] || [];
      selectCommodity.value = availableLanduse.value[0] || '';

      await ensureComboLayer(mapRegister[initCat].layerPrefix, [selectWater.value, selectCommodity.value]);
      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };
    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });

    watch(selectCategory, async (newCat, oldCat) => {
      if (oldCat === "Sum") previousSelections.value["Sum"] = { water: selectWater.value, landuse: selectCommodity.value };
      if (oldCat === "Ag") previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectCommodity.value };
      if (oldCat === "Ag Mgt") previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectCommodity.value };
      if (oldCat === "Non-Ag") previousSelections.value["Non-Ag"] = { landuse: selectCommodity.value };

      const curWater = selectWater.value, curLanduse = selectCommodity.value, curAgMgt = selectAgMgt.value;
      await ensureIndexLoaded(newCat);
      const tree = getTree(newCat);

      if (newCat === "Non-Ag") {
        availableWater.value = [];
        availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        const prev = previousSelections.value["Non-Ag"]?.landuse || curLanduse;
        selectCommodity.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Non-Ag"].layerPrefix, [selectCommodity.value]);
      } else {
        // Sum, Ag, Ag Mgt all have water → commodity
        if (newCat === "Ag Mgt") {
          availableAgMgt.value = Object.keys(tree);
          const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt || curAgMgt;
          selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');
          const subTree = tree[selectAgMgt.value] || {};
          availableWater.value = Object.keys(subTree);
          const prevWater = previousSelections.value["Ag Mgt"].water || curWater;
          selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
          availableLanduse.value = subTree[selectWater.value] || [];
          const prevLanduse = previousSelections.value["Ag Mgt"].landuse || curLanduse;
          selectCommodity.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, selectCommodity.value]);
        } else {
          availableWater.value = Object.keys(tree);
          const prevWater = previousSelections.value[newCat]?.water || curWater;
          selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
          availableLanduse.value = tree[selectWater.value] || [];
          const prevLanduse = previousSelections.value[newCat]?.landuse || curLanduse;
          selectCommodity.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister[newCat].layerPrefix, [selectWater.value, selectCommodity.value]);
        }
      }
    });

    watch(selectAgMgt, async (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const tree = getTree("Ag Mgt");
      availableWater.value = Object.keys(tree[newAgMgt] || {});
      const prevWater = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
      availableLanduse.value = tree[newAgMgt]?.[selectWater.value] || [];
      const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
      selectCommodity.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [newAgMgt, selectWater.value, selectCommodity.value]);
    });

    watch(selectWater, async (newWater) => {
      const cat = selectCategory.value;
      if (!cat || !mapRegister[cat]) return;   // guard: fires before selectCategory is set during onMounted
      if (cat === "Non-Ag") return;
      previousSelections.value[cat] = { ...(previousSelections.value[cat] || {}), water: newWater };
      const tree = getTree(cat);
      const subTree = (cat === "Ag Mgt") ? (tree[selectAgMgt.value] || {}) : tree;
      availableLanduse.value = subTree[newWater] || [];
      const prevLanduse = previousSelections.value[cat]?.landuse;
      selectCommodity.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      const comboArgs = (cat === "Ag Mgt")
        ? [selectAgMgt.value, newWater, selectCommodity.value]
        : [newWater, selectCommodity.value];
      await ensureComboLayer(mapRegister[cat].layerPrefix, comboArgs);
    });

    watch(selectCommodity, async (newCommodity) => {
      const cat = selectCategory.value;
      if (cat === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newCommodity;
        await ensureComboLayer(mapRegister["Non-Ag"].layerPrefix, [newCommodity]);
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newCommodity;
        await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, newCommodity]);
      } else if (cat) {
        previousSelections.value[cat] = { ...(previousSelections.value[cat] || {}), landuse: newCommodity };
        await ensureComboLayer(mapRegister[cat].layerPrefix, [selectWater.value, newCommodity]);
      }
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears, availableCategories,
      availableAgMgt, availableWater, availableLanduse,
      selectCategory, selectAgMgt, selectWater, selectCommodity,
      selectMapData, selectChartData,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
    };
    const _fn = v => String(v).trim().replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    _state.mapFileName = computed(() =>
      [VIEW_NAME, selectCategory.value, selectAgMgt.value, selectWater.value, selectCommodity.value, selectYear.value]
        .filter(Boolean).map(_fn).filter(Boolean).join('__')
    );
    window._debug[VIEW_NAME] = _state;
    return _state;
  },
  template: /*html*/`
    <div class="relative w-full h-screen">

      <!-- Region selection dropdown -->
      <div class="absolute w-[262px] top-32 left-[20px] z-50 bg-white/70 rounded-lg shadow-lg max-w-xs z-[9999]">
        <filterable-dropdown></filterable-dropdown>
      </div>

      <!-- Year slider -->
      <div class="absolute top-[200px] left-[20px] z-[1001] w-[262px] bg-white/70 p-2 rounded-lg items-center">
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

      <!-- Data selection controls container -->
      <div class="absolute top-[285px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

        <!-- Category buttons (always visible) -->
        <div class="flex space-x-1">
          <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
          <button v-for="(val, key) in availableCategories" :key="key"
            @click="selectCategory = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
            :class="{'bg-sky-500 text-white': selectCategory === val}">
            {{ val }}
          </button>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category) -->
        <div v-if="dataLoaded && selectCategory === 'Ag Mgt' && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>

        <!-- Water options (for Sum, Ag and Ag Mgt) -->
        <div v-if="dataLoaded && (selectCategory === 'Sum' || selectCategory === 'Ag' || selectCategory === 'Ag Mgt') && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Landuse options -->
        <div v-if="dataLoaded" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
          <button v-for="(val, key) in availableLanduse" :key="key"
            @click="selectCommodity = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectCommodity === val}">
            {{ val }}
          </button>
        </div>
      </div>

      <!-- Map container with slide-out chart drawer -->
      <div style="position: relative; width: 100%; height: 100%; overflow: hidden;">

        <!-- Loading overlay shown while lazy-loading a new map file -->
        <div v-if="isLoadingData"
          class="absolute inset-0 z-[2000] flex items-center justify-center bg-white/60 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-2 text-gray-600 text-sm font-medium">
            <svg class="animate-spin h-8 w-8 text-sky-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            Loading map data…
          </div>
        </div>

        <!-- Map component takes full space -->
        <regions-map
          :mapData="selectMapData"
          :file-name="mapFileName"
          :show-legend="!isDrawerOpen"
          style="width: 100%; height: 100%;">
        </regions-map>

        <!-- Drawer toggle button -->
        <button
          @click="toggleDrawer"
          class="absolute top-5 z-[1001] p-2.5 bg-white border border-gray-300 rounded cursor-pointer transition-all duration-300 ease-in-out"
          :class="isDrawerOpen ? 'right-[420px]' : 'right-5'">
          {{ isDrawerOpen ? '→' : '←' }}
        </button>

        <!-- Chart drawer positioned relative to map -->
        <div
          :style="{
            position: 'absolute',
            height: '50px',
            top: '10px',
            bottom: '10px',
            right: isDrawerOpen ? '0px' : '-100%',
            width: '66.666%',
            background: 'transparent',
            transition: 'right 0.3s ease',
            zIndex: 1000,
            padding: '60px 20px 20px 20px',
            boxSizing: 'border-box'
          }">
          <chart-container
            :chartData="selectChartData"
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

    </div>
  `,
};