window.RenewableView = {
  name: 'RenewableView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Renewable"];
    const mapRegister = window.MapService.mapCategories["Renewable"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Renewable";

    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");
    const availableRegionLevels = ['region_state', 'region_NRM'];
    const selectRegionLevel = ref('region_state');

    const availableYears = ref([]);
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectLanduse = ref("");

    const previousSelections = ref({ agMgt: "", water: "", landuse: "" });

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    const selectChartData = computed(() => {
      if (!dataLoaded.value) return {};
      const chartData = window[chartRegister["Ag Mgt"]?.["name"]]?.[selectRegionLevel.value]?.[selectRegion.value];
      let seriesData = (chartData?.[selectAgMgt.value]?.[selectWater.value] || [])
        .filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: "MWh" } },
        series: seriesData || [],
      };
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    const rezOverlay = ref(null);
    const mapOverlays = computed(() =>
      rezOverlay.value ? [{ id: 'rez', data: rezOverlay.value }] : []
    );

    function getTree() {
      return window[mapRegister["Ag Mgt"]?.indexName]?.tree ?? {};
    }

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      isLoadingData.value = true;
      await Promise.all([
        loadScript(mapRegister["Ag Mgt"].indexPath, mapRegister["Ag Mgt"].indexName, VIEW_NAME),
        loadScript(chartRegister["Ag Mgt"].path, chartRegister["Ag Mgt"].name, VIEW_NAME),
        loadScript("./data/geo/REZ.js", "RENEWABLE_REZ", VIEW_NAME),
      ]);
      rezOverlay.value = window.RENEWABLE_REZ || null;
      isLoadingData.value = false;

      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;

      // Tree: { am: { lm: [lu] } }
      const tree = getTree();
      availableAgMgt.value = Object.keys(tree);
      selectAgMgt.value = availableAgMgt.value[0] || '';
      availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
      selectWater.value = availableWater.value[0] || '';
      availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
      selectLanduse.value = availableLanduse.value[0] || '';

      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);
      dataLoaded.value = true;
    });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };
    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });
    watch(selectRegionLevel, () => { selectRegion.value = 'AUSTRALIA'; });

    watch(selectAgMgt, async (newAgMgt) => {
      previousSelections.value.agMgt = newAgMgt;
      const tree = getTree();
      availableWater.value = Object.keys(tree[newAgMgt] || {});
      const prevWater = previousSelections.value.water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
      availableLanduse.value = tree[newAgMgt]?.[selectWater.value] || [];
      const prevLanduse = previousSelections.value.landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [newAgMgt, selectWater.value, selectLanduse.value]);
    });

    watch(selectWater, async (newWater) => {
      previousSelections.value.water = newWater;
      const tree = getTree();
      availableLanduse.value = tree[selectAgMgt.value]?.[newWater] || [];
      const prevLanduse = previousSelections.value.landuse;
      selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, newWater, selectLanduse.value]);
    });

    watch(selectLanduse, async (newLanduse) => {
      previousSelections.value.landuse = newLanduse;
      await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, newLanduse]);
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableRegionLevels, selectRegionLevel,
      availableYears, availableAgMgt, availableWater, availableLanduse,
      selectAgMgt, selectWater, selectLanduse,
      selectMapData, selectChartData,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
      rezOverlay, mapOverlays,
    };
    const _fn = v => String(v).trim().replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    _state.mapFileName = computed(() =>
      [VIEW_NAME, selectAgMgt.value, selectWater.value, selectLanduse.value, selectYear.value]
        .filter(Boolean).map(_fn).filter(Boolean).join('__')
    );
    window._debug[VIEW_NAME] = _state;
    return _state;
  },
  template: /*html*/`
    <div class="relative w-full h-screen">

      <!-- Region level tabs + Region selection dropdown -->
      <div class="absolute w-[262px] top-28 left-[20px] z-[9999] max-w-xs">
        <!-- Drawer-style region level tabs -->
        <div class="flex gap-1 ml-2 mb-0">
          <button v-for="lvl in availableRegionLevels" :key="lvl"
            @click="selectRegionLevel = lvl"
            class="px-2 py-0.5 text-[0.65rem] font-medium rounded-t-md border border-b-0 transition-colors"
            :class="selectRegionLevel === lvl
              ? 'bg-white/90 border-gray-300 text-sky-600'
              : 'bg-white/40 border-gray-200 text-gray-500 hover:bg-white/60'">
            {{ lvl === 'region_state' ? 'State' : 'NRM' }}
          </button>
        </div>
        <!-- Dropdown panel -->
        <div class="bg-white/70 rounded-lg shadow-lg">
          <filterable-dropdown
            :key="selectRegionLevel"
            :region-type="selectRegionLevel === 'region_state' ? 'STATE' : 'NRM'">
          </filterable-dropdown>
        </div>
      </div>

      <!-- Year slider -->
      <div class="absolute top-[240px] left-[20px] z-[1001] w-[262px] bg-white/70 p-2 rounded-lg items-center">
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
      <div class="absolute top-[325px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

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
          :region-type="selectRegionLevel === 'region_state' ? 'STATE' : 'NRM'"
          :overlays="mapOverlays"
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