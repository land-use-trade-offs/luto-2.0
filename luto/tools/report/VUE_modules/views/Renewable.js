window.RenewableView = {
  name: 'RenewableView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["Renewable"];
    const mapRegister = window.MapService.mapCategories["Renewable"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Renewable";

    // Global selection state
    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    // Available variables
    const availableYears = ref([]);

    // Available selections — Am only: AgMgt → Water → LU
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    // Map selection state
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectLanduse = ref("");

    // Previous selections memory
    const previousSelections = ref({ agMgt: "", water: "", landuse: "" });

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);

    // map_renewable_energy_Am: AgMgt → Water → LU → Year
    const selectMapData = computed(() => {
      if (!dataLoaded.value) return {};
      const mapData = window[mapRegister["Ag Mgt"]["name"]];
      return mapData?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
    });

    // Renewable_energy_Am chart: Region → AgMgt → Water → [series(name=LU)]
    const selectChartData = computed(() => {
      if (!dataLoaded.value) return {};
      const chartData = window[chartRegister["Ag Mgt"]["name"]]?.[selectRegion.value];
      let seriesData = chartData?.[selectAgMgt.value]?.[selectWater.value] || [];
      seriesData = seriesData.filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: "MWh" } },
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

      // Load map and chart data
      await loadScript(mapRegister["Ag Mgt"]["path"], mapRegister["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(chartRegister["Ag Mgt"]["path"], chartRegister["Ag Mgt"]["name"], VIEW_NAME);

      // Initial selections
      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;

      const amData = window[mapRegister["Ag Mgt"]["name"]];
      availableAgMgt.value = Object.keys(amData || {});
      selectAgMgt.value = availableAgMgt.value[0] || '';

      availableWater.value = Object.keys(amData?.[selectAgMgt.value] || {});
      selectWater.value = availableWater.value[0] || '';

      availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[selectWater.value] || {});
      selectLanduse.value = availableLanduse.value[0] || '';

      await nextTick(() => {
        dataLoaded.value = true;
      });
    });

    // Watchers
    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    watch(selectAgMgt, (newAgMgt) => {
      previousSelections.value.agMgt = newAgMgt;
      const amData = window[mapRegister["Ag Mgt"]["name"]];

      availableWater.value = Object.keys(amData?.[newAgMgt] || {});
      const prevWater = previousSelections.value.water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

      availableLanduse.value = Object.keys(amData?.[newAgMgt]?.[selectWater.value] || {});
      const prevLanduse = previousSelections.value.landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
    });

    watch(selectWater, (newWater) => {
      previousSelections.value.water = newWater;
      const amData = window[mapRegister["Ag Mgt"]["name"]];

      availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[newWater] || {});
      const prevLanduse = previousSelections.value.landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
    });

    watch(selectLanduse, (newLanduse) => {
      previousSelections.value.landuse = newLanduse;
    });

    const _state = {
      yearIndex,
      selectYear,
      selectRegion,

      availableYears,
      availableAgMgt,
      availableWater,
      availableLanduse,

      selectAgMgt,
      selectWater,
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

        <!-- Ag Mgt options -->
        <div v-if="dataLoaded && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>

        <!-- Water options -->
        <div v-if="dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Landuse options -->
        <div v-if="dataLoaded && availableLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
