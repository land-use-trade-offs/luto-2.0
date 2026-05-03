window.WaterView = {
  name: 'WaterView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Water"];
    const mapRegister = window.MapService.mapCategories["Water"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Water";

    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    const availableYears = ref([]);
    const availableUnit = {
      Area: "Hectares", Economics: "AUD", GHG: "Mt CO2e",
      Water: "ML", Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    const availableCategories = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectLanduse = ref("");

    const previousSelections = ref({
      "Sum": { landuse: "" },
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    const SUM_TYPE_TO_SERIES = { 'ag': 'Agricultural Land-use', 'ag-man': 'Agricultural Management', 'non-ag': 'Non-Agricultural Land-use' };
    function formatLanduse(val) {
      return selectCategory.value === 'Sum' ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    const selectChartData = computed(() => {
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      if (!dataLoaded.value) return {};
      const chartData = window[chartRegister["NRM"]?.[cat]?.["name"]]?.[region];
      let seriesData;

      if (cat === "Sum") {
        const items = chartData || [];
        const seriesName = SUM_TYPE_TO_SERIES[landuse];
        seriesData = (landuse === "ALL" || !landuse) ? items : items.filter(s => s.name === seriesName);
      } else if (cat === "Ag") {
        seriesData = (chartData?.[water] || []).filter(s => landuse === "ALL" || s.name === landuse);
      } else if (cat === "Ag Mgt") {
        seriesData = (chartData?.[water]?.[landuse] || []).filter(s => agMgt === "ALL" || s.name === agMgt);
      } else if (cat === "Non-Ag") {
        seriesData = (chartData || []).filter(s => landuse === "ALL" || s.name === landuse);
      }

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["Water"] } },
        series: seriesData || [],
      };
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    function getTree(cat) {
      const t = window[mapRegister[cat]?.indexName]?.tree;
      if (t !== undefined) return t;
      return (cat === "Non-Ag" || cat === "Sum") ? [] : {};
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
      for (const regionDict of Object.values(chartRegister)) {
        for (const entry of Object.values(regionDict || {})) {
          if (entry?.name && !window[entry.name])
            pending.push(loadScript(entry.path, entry.name, VIEW_NAME));
        }
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

      const tree = getTree(initCat);
      availableWater.value = [];
      availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
      selectLanduse.value = availableLanduse.value[0] || '';

      await ensureComboLayer(mapRegister[initCat].layerPrefix, [selectLanduse.value]);
      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };
    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });

    watch(selectCategory, async (newCat, oldCat) => {
      if (oldCat === "Sum") previousSelections.value["Sum"] = { landuse: selectLanduse.value };
      if (oldCat === "Ag") previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
      if (oldCat === "Ag Mgt") previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
      if (oldCat === "Non-Ag") previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };

      const curWater = selectWater.value, curLanduse = selectLanduse.value, curAgMgt = selectAgMgt.value;
      await ensureIndexLoaded(newCat);
      const tree = getTree(newCat);

      if (newCat === "Sum" || newCat === "Non-Ag") {
        availableWater.value = [];
        availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        const prev = previousSelections.value[newCat]?.landuse || curLanduse;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister[newCat].layerPrefix, [selectLanduse.value]);
      } else if (newCat === "Ag") {
        availableWater.value = Object.keys(tree);
        const prevWater = previousSelections.value["Ag"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
        availableLanduse.value = tree[selectWater.value] || [];
        const prevLanduse = previousSelections.value["Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag"].layerPrefix, [selectWater.value, selectLanduse.value]);
      } else if (newCat === "Ag Mgt") {
        availableAgMgt.value = Object.keys(tree);
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt || curAgMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');
        availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
        availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);
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
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [newAgMgt, selectWater.value, selectLanduse.value]);
    });

    watch(selectWater, async (newWater) => {
      const cat = selectCategory.value;
      if (cat === "Ag") {
        previousSelections.value["Ag"].water = newWater;
        const tree = getTree("Ag");
        availableLanduse.value = tree[newWater] || [];
        const prevLanduse = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag"].layerPrefix, [newWater, selectLanduse.value]);
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const tree = getTree("Ag Mgt");
        availableLanduse.value = tree[selectAgMgt.value]?.[newWater] || [];
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, newWater, selectLanduse.value]);
      }
    });

    watch(selectLanduse, async (newLanduse) => {
      const cat = selectCategory.value;
      if (cat === "Sum") {
        previousSelections.value["Sum"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Sum"].layerPrefix, [newLanduse]);
      } else if (cat === "Ag") {
        previousSelections.value["Ag"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Ag"].layerPrefix, [selectWater.value, newLanduse]);
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, newLanduse]);
      } else if (cat === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Non-Ag"].layerPrefix, [newLanduse]);
      }
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears, availableCategories,
      availableAgMgt, availableWater, availableLanduse,
      selectCategory, selectAgMgt, selectWater, selectLanduse,
      selectMapData, selectChartData, formatLanduse,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
    };
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

        <!-- Water options (Ag and Ag Mgt) -->
        <div v-if="selectCategory !== 'Non-Ag' && selectCategory !== 'Sum' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
          <span class="text-[0.8rem] mr-1 font-medium">{{ selectCategory === 'Sum' ? 'Type:' : 'Landuse:' }}</span>
          <button v-for="(val, key) in availableLanduse" :key="key"
            @click="selectLanduse = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectLanduse === val}">
            {{ formatLanduse(val) }}
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