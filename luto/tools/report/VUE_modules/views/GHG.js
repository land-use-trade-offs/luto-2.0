window.GHGView = {
  name: 'GHGView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["GHG"];
    const mapRegister = window.MapService.mapCategories["GHG"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "GHG";

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
    const availableSource = ref([]);  // Ag only: Water → Source → LU
    const availableLanduse = ref([]);

    // Map selection state
    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectSource = ref("");  // Ag only
    const selectLanduse = ref("");

    // Previous selections memory
    const previousSelections = ref({
      "Ag": { water: "", source: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);

    // Reactive data
    // map_GHG_Ag:    Water → Source → LU → Year
    // map_GHG_Am:    AgMgt → Water → LU → Year
    // map_GHG_NonAg: LU → Year
    const selectMapData = computed(() => {
      if (!dataLoaded.value) return {};
      const mapData = window[mapRegister[selectCategory.value]["name"]];
      if (selectCategory.value === "Ag") {
        return mapData?.[selectWater.value]?.[selectSource.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      } else if (selectCategory.value === "Ag Mgt") {
        return mapData?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      } else if (selectCategory.value === "Non-Ag") {
        return mapData?.[selectLanduse.value]?.[selectYear.value] || {};
      }
      return {};
    });

    // GHG_Ag chart:    Region → Water → Source → [series(name=LU)]
    // GHG_Am chart:    Region → Water → LU → [series(name=AgMgt)]
    // GHG_NonAg chart: Region → [series(name=LU)]
    const selectChartData = computed(() => {
      if (!dataLoaded.value) return {};
      const chartData = window[chartRegister[selectCategory.value]["name"]]?.[selectRegion.value];
      let seriesData;

      if (selectCategory.value === "Ag") {
        seriesData = chartData?.[selectWater.value]?.[selectSource.value] || [];
        seriesData = seriesData.filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      } else if (selectCategory.value === "Ag Mgt") {
        seriesData = chartData?.[selectWater.value]?.[selectLanduse.value] || [];
        seriesData = seriesData.filter(s => selectAgMgt.value === "ALL" || s.name === selectAgMgt.value);
      } else if (selectCategory.value === "Non-Ag") {
        seriesData = (chartData || []).filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      }

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["GHG"] } },
        series: seriesData || [],
      };
    });

    // Memory cleanup on component unmount
    onUnmounted(() => {
      window.MemoryService.cleanupViewData(VIEW_NAME);
    });

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      // Load data
      await loadScript(mapRegister["Ag"]["path"], mapRegister["Ag"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Ag Mgt"]["path"], mapRegister["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Non-Ag"]["path"], mapRegister["Non-Ag"]["name"], VIEW_NAME);
      await loadScript(chartRegister["Ag"]["path"], chartRegister["Ag"]["name"], VIEW_NAME);
      await loadScript(chartRegister["Ag Mgt"]["path"], chartRegister["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(chartRegister["Non-Ag"]["path"], chartRegister["Non-Ag"]["name"], VIEW_NAME);

      // Initial selections
      availableYears.value = window.Supporting_info.years;
      selectCategory.value = availableCategories[0];

      await nextTick(() => {
        dataLoaded.value = true;
      });
    });

    // Watchers and methods
    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    // Progressive selection chain watchers
    watch(selectCategory, (newCategory, oldCategory) => {
      // Save previous selections before switching
      if (oldCategory === "Ag") {
        previousSelections.value["Ag"] = { water: selectWater.value, source: selectSource.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Ag Mgt") {
        previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Non-Ag") {
        previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
      }

      const agData = window[mapRegister["Ag"]["name"]];
      const amData = window[mapRegister["Ag Mgt"]["name"]];
      const nonAgData = window[mapRegister["Non-Ag"]["name"]];

      if (newCategory === "Ag") {
        // Cascade: Water → Source → LU
        availableWater.value = Object.keys(agData || {});
        const prevWater = previousSelections.value["Ag"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableSource.value = Object.keys(agData?.[selectWater.value] || {});
        const prevSource = previousSelections.value["Ag"].source;
        selectSource.value = (prevSource && availableSource.value.includes(prevSource)) ? prevSource : (availableSource.value[0] || '');

        availableLanduse.value = Object.keys(agData?.[selectWater.value]?.[selectSource.value] || {});
        const prevLanduse = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (newCategory === "Ag Mgt") {
        // Cascade: AgMgt → Water → LU
        availableAgMgt.value = Object.keys(amData || {});
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');

        availableWater.value = Object.keys(amData?.[selectAgMgt.value] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[selectWater.value] || {});
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');

      } else if (newCategory === "Non-Ag") {
        // Cascade: LU only
        availableLanduse.value = Object.keys(nonAgData || {});
        const prevLanduse = previousSelections.value["Non-Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      }
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

        availableSource.value = Object.keys(agData?.[newWater] || {});
        const prevSource = previousSelections.value["Ag"].source;
        selectSource.value = (prevSource && availableSource.value.includes(prevSource)) ? prevSource : (availableSource.value[0] || '');

        availableLanduse.value = Object.keys(agData?.[newWater]?.[selectSource.value] || {});
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

    watch(selectSource, (newSource) => {
      if (selectCategory.value !== "Ag") return;
      previousSelections.value["Ag"].source = newSource;
      const agData = window[mapRegister["Ag"]["name"]];

      availableLanduse.value = Object.keys(agData?.[selectWater.value]?.[newSource] || {});
      const prevLanduse = previousSelections.value["Ag"].landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
    });

    watch(selectLanduse, (newLanduse) => {
      if (selectCategory.value === "Ag") {
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
      availableSource,
      availableLanduse,

      selectCategory,
      selectAgMgt,
      selectWater,
      selectSource,
      selectLanduse,

      selectMapData,
      selectChartData,

      dataLoaded,
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
        <div v-if="selectCategory !== 'Non-Ag' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Source options (Ag only: emission type) -->
        <div v-if="selectCategory === 'Ag' && dataLoaded && availableSource.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Source:</span>
          <button v-for="(val, key) in availableSource" :key="key"
            @click="selectSource = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectSource === val}">
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

        <!-- Map component takes full space -->
        <regions-map
          :mapData="selectMapData"
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
