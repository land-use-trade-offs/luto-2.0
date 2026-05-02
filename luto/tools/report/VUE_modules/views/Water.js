window.WaterView = {
  name: 'WaterView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["Water"];
    const mapRegister = window.MapService.mapCategories["Water"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Water";

    // Global selection state
    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    // Available variables
    const availableYears = ref([]);
    const availableUnit = {
      Area: "Hectares",
      Economics: "AUD",
      GHG: "Mt CO2e",
      Water: "ML",
      Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    // Available selections
    const availableCategories = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    // Map selection state
    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectLanduse = ref("");

    // Previous selections memory
    const previousSelections = ref({
      "Sum": { landuse: "" },
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    // UI state
    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const triggerVersion = ref(0);
    const isDrawerOpen = ref(false);

    // Sum-tab: Type key → display label and series name
    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    const SUM_TYPE_TO_SERIES = { 'ag': 'Agricultural Land-use', 'ag-man': 'Agricultural Management', 'non-ag': 'Non-Agricultural Land-use' };
    function formatLanduse(val) {
      return selectCategory.value === 'Sum' ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    // Reactive data
    // map_water_yield_Sum:   Type → Year
    // map_water_yield_Ag:    Water → LU → Year
    // map_water_yield_Am:    AgMgt → Water → LU → Year
    // map_water_yield_NonAg: LU → Year
    const selectMapData = computed(() => {
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const year = selectYear.value;
      void triggerVersion.value;
      if (!dataLoaded.value) return {};
      const mapData = window[mapRegister[cat]["name"]];
      if (cat === "Sum") {
        return mapData?.[landuse]?.[year] || {};
      } else if (cat === "Ag") {
        return mapData?.[water]?.[landuse]?.[year] || {};
      } else if (cat === "Ag Mgt") {
        return mapData?.[agMgt]?.[water]?.[landuse]?.[year] || {};
      } else if (cat === "Non-Ag") {
        return mapData?.[landuse]?.[year] || {};
      }
      return {};
    });

    // Water_Sum_NRM chart:   Region → [series(name=TypeDisplayLabel)]
    // Water_Ag_NRM chart:    Region → Water → [series(name=LU)]
    // Water_Am_NRM chart:    Region → Water → LU → [series(name=AgMgt)]
    // Water_NonAg_NRM chart: Region → [series(name=LU)]
    const selectChartData = computed(() => {
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      void triggerVersion.value;
      if (!dataLoaded.value) return {};
      const chartData = window[chartRegister["NRM"][cat]["name"]]?.[region];
      let seriesData;

      if (cat === "Sum") {
        // Sum: region → [series(name=TypeDisplayLabel)]
        const items = chartData || [];
        const seriesName = SUM_TYPE_TO_SERIES[landuse];
        seriesData = (landuse === "ALL" || !landuse)
          ? items : items.filter(s => s.name === seriesName);
      } else if (cat === "Ag") {
        seriesData = chartData?.[water] || [];
        seriesData = seriesData.filter(s => landuse === "ALL" || s.name === landuse);
      } else if (cat === "Ag Mgt") {
        seriesData = chartData?.[water]?.[landuse] || [];
        seriesData = seriesData.filter(s => agMgt === "ALL" || s.name === agMgt);
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

    // Memory cleanup on component unmount
    onUnmounted(() => {
      window.MemoryService.cleanupViewData(VIEW_NAME);
    });

    // ── Lazy loader (maps only) ──────────────────────────────────────────────
    async function ensureDataLoaded(cat) {
      const mapEntry = mapRegister[cat];
      if (mapEntry && !window[mapEntry.name]) {
        isLoadingData.value = true;
        await loadScript(mapEntry.path, mapEntry.name, VIEW_NAME);
        isLoadingData.value = false;
      }
    }

    // Pre-load ALL chart files on mount (they are small)
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

      // Load initial map + ALL charts in parallel
      const initCat = availableCategories[0]; // "Sum"
      await Promise.all([ensureDataLoaded(initCat), loadAllCharts()]);

      // Cascade initial selections synchronously (Sum: Type only)
      // Cascade initial selections synchronously (Sum: Type only)
      const initData = window[mapRegister[initCat]["name"]];
      availableWater.value = [];
      availableLanduse.value = Object.keys(initData || {});
      selectLanduse.value = availableLanduse.value[0] || '';

      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    // Watchers and methods
    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    // Progressive selection chain watchers
    watch(selectCategory, async (newCategory, oldCategory) => {
      // Save previous selections before switching
      if (oldCategory === "Sum") {
        previousSelections.value["Sum"] = { landuse: selectLanduse.value };
      } else if (oldCategory === "Ag") {
        previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Ag Mgt") {
        previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Non-Ag") {
        previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
      }

      // Remember current selections for cross-category restore
      const curWater = selectWater.value;
      const curLanduse = selectLanduse.value;
      const curAgMgt = selectAgMgt.value;

      // Only fetch if not already loaded (avoid unnecessary microtask yield)
      const _me = mapRegister[newCategory];
      if (_me && !window[_me.name]) {
        await ensureDataLoaded(newCategory);
      }

      const sumData = window[mapRegister["Sum"]["name"]];
      const agData = window[mapRegister["Ag"]["name"]];
      const amData = window[mapRegister["Ag Mgt"]["name"]];
      const nonAgData = window[mapRegister["Non-Ag"]["name"]];

      if (newCategory === "Sum") {
        // Cascade: Type only
        availableWater.value = [];
        availableLanduse.value = Object.keys(sumData || {});
        const prevLanduse = previousSelections.value["Sum"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (newCategory === "Ag") {
        // Cascade: Water → LU
        availableWater.value = Object.keys(agData || {});
        const prevWater = previousSelections.value["Ag"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(agData?.[selectWater.value] || {});
        const prevLanduse = previousSelections.value["Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (newCategory === "Ag Mgt") {
        // Cascade: AgMgt → Water → LU
        availableAgMgt.value = Object.keys(amData || {});
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt || curAgMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');

        availableWater.value = Object.keys(amData?.[selectAgMgt.value] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[selectWater.value] || {});
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (newCategory === "Non-Ag") {
        // Cascade: LU only
        availableLanduse.value = Object.keys(nonAgData || {});
        const prevLanduse = previousSelections.value["Non-Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
      triggerVersion.value++;
    });

    watch(selectAgMgt, (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const amData = window[mapRegister["Ag Mgt"]["name"]];

      availableWater.value = Object.keys(amData?.[newAgMgt] || {});
      const prevWater = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

      availableLanduse.value = Object.keys(amData?.[newAgMgt]?.[selectWater.value] || {});
      const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
    });

    watch(selectWater, (newWater) => {
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].water = newWater;
        const agData = window[mapRegister["Ag"]["name"]];

        availableLanduse.value = Object.keys(agData?.[newWater] || {});
        const prevLanduse = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const amData = window[mapRegister["Ag Mgt"]["name"]];

        availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[newWater] || {});
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
    });

    watch(selectLanduse, (newLanduse) => {
      if (selectCategory.value === "Sum") {
        previousSelections.value["Sum"].landuse = newLanduse;
      } else if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].landuse = newLanduse;
      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newLanduse;
      } else if (selectCategory.value === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newLanduse;
      }
    });

    const _state = {
      yearIndex,
      selectYear,
      selectRegion,

      availableYears,
      availableCategories,
      availableAgMgt,
      availableWater,
      availableLanduse,

      selectCategory,
      selectAgMgt,
      selectWater,
      selectLanduse,

      selectMapData,
      selectChartData,
      formatLanduse,

      dataLoaded, isLoadingData,
      isDrawerOpen,
      toggleDrawer,
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