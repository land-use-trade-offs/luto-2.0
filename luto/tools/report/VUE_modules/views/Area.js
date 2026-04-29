window.AreaView = {
  name: 'AreaView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["Area"];    // ChartService has been registered in index.html       [ChartService.js]
    const mapRegister = window.MapService.mapCategories["Area"];          // MapService was registered in the index.html          [MapService.js]
    const loadScript = window.loadScriptWithTracking;                     // Enhanced loadScript with memory tracking             [helpers.js]

    // View identification for memory management
    const VIEW_NAME = "Area";

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
    const availableCategories = ["Ag", "Ag Mgt", "Non-Ag"];
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
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });


    // UI state
    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const triggerVersion = ref(0); // incremented after every cascade to force computed re-run
    const isDrawerOpen = ref(false);


    //  Reactive data
    const selectMapData = computed(() => {
      // Read all reactive deps upfront so Vue always tracks them as dependencies,
      // even when we return early before the data is available.
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const year = selectYear.value;
      void triggerVersion.value; // re-run after every cascade (window[] is not reactive)
      if (!dataLoaded.value) return {};

      const rawData = window[mapRegister[cat]?.["name"]];
      if (!rawData) return {};
      let mapData = JSON.parse(JSON.stringify(rawData));

      if (cat === "Ag") {
        return mapData[water]?.[landuse]?.[year] ?? {};
      }
      else if (cat === "Ag Mgt") {
        return mapData?.[agMgt]?.[water]?.[landuse]?.[year] ?? {};
      }
      else if (cat === "Non-Ag") {
        return mapData[landuse]?.[year] ?? {};
      }
      return {};
    });


    const selectChartData = computed(() => {
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      void triggerVersion.value; // re-run after every cascade
      if (!dataLoaded.value) return {};

      const rawChart = window[chartRegister[cat]?.["name"]];
      if (!rawChart) return {};
      let chartData = rawChart[region];
      let seriesData;

      if (cat === "Ag") {
        seriesData = chartData[water];
        seriesData = seriesData.filter(serie => (landuse === "ALL" || serie.name === landuse));
      }
      else if (cat === "Ag Mgt") {
        seriesData = chartData[water][landuse];
        seriesData = seriesData.filter(serie => (agMgt === "ALL" || serie.name === agMgt));
      } else if (cat === "Non-Ag") {
        seriesData = chartData;
        seriesData = seriesData.filter(serie => (landuse === "ALL" || serie.name === landuse));
      }

      return {
        ...window["Chart_default_options"],
        chart: {
          height: 440,
        },
        yAxis: {
          title: {
            text: availableUnit[cat],
          },
        },
        series: seriesData || [],
      };
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

      // Load initial map + ALL charts in parallel
      const initCat = availableCategories[0]; // "Ag"
      await Promise.all([ensureDataLoaded(initCat), loadAllCharts()]);

      // Cascade initial selections synchronously so computed has all values when dataLoaded=true
      const initData = window[mapRegister[initCat]["name"]];
      availableWater.value = Object.keys(initData || {});
      selectWater.value = availableWater.value[0] || '';
      availableLanduse.value = Object.keys(initData?.[selectWater.value] || {});
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
      if (oldCategory) {
        if (oldCategory === "Ag") {
          previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
        } else if (oldCategory === "Ag Mgt") {
          previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
        } else if (oldCategory === "Non-Ag") {
          previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
        }
      }

      // Remember current selections for cross-category restore
      const curWater = selectWater.value;
      const curLanduse = selectLanduse.value;
      const curAgMgt = selectAgMgt.value;

      // Only fetch if not already loaded (avoid unnecessary microtask yield that leaves
      // computed functions with stale selectWater/selectLanduse refs)
      const _me = mapRegister[newCategory];
      if (_me && !window[_me.name]) {
        await ensureDataLoaded(newCategory);
      }

      // Handle ALL downstream variables with cascading pattern
      if (newCategory === "Ag Mgt") {
        availableAgMgt.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]]);
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt || curAgMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');

        availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value]);
        const prevWater = previousSelections.value["Ag Mgt"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][selectWater.value]);
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      } else if (newCategory === "Ag") {
        availableWater.value = Object.keys(window[mapRegister["Ag"]["name"]]);
        const prevWater = previousSelections.value["Ag"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][selectWater.value]);
        const prevLanduse = previousSelections.value["Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      } else if (newCategory === "Non-Ag") {
        availableLanduse.value = Object.keys(window[mapRegister["Non-Ag"]["name"]]);
        const prevLanduse = previousSelections.value["Non-Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
      triggerVersion.value++;
    });


    watch(selectWater, (newWater) => {
      // Save current water selection
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].water = newWater;
      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
      }

      // Handle ALL downstream variables
      if (selectCategory.value === "Ag") {
        availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][newWater]);
        const prevLanduse = previousSelections.value["Ag"].landuse || selectLanduse.value;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      } else if (selectCategory.value === "Ag Mgt") {
        availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][newWater]);
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse || selectLanduse.value;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
    });

    watch(selectAgMgt, (newAgMgt) => {
      // Save current agMgt selection
      if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].agMgt = newAgMgt;

        // Handle ALL downstream variables with cascading pattern
        availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt]);
        const prevWater = previousSelections.value["Ag Mgt"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt][selectWater.value]);
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse || selectLanduse.value;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
    });

    watch(selectLanduse, (newLanduse) => {
      // Save current landuse selection
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].landuse = newLanduse;
      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newLanduse;
      } else if (selectCategory.value === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newLanduse;
      }
    });

    // Memory cleanup on component unmount
    onUnmounted(() => {
      window.MemoryService.cleanupViewData(VIEW_NAME);
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
        <div v-if="selectCategory === 'Ag Mgt' && dataLoaded && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>
        

        <!-- Water options -->
        <div v-if="selectCategory !== 'Non-Ag' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
            @click="selectLanduse = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectLanduse === val}">
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
