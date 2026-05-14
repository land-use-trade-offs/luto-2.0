window.EconomicsView = {
  name: 'EconomicsView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch } = Vue;

    const chartRegister = window.ChartService.chartCategories["Economics"];
    const mapRegister = window.MapService.mapCategories["Economics"];
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Economics";

    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    const availableYears = ref([]);
    const availableUnit = { Economics: "AUD" };
    const availableCategories = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableMapTypes = ref([]);
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableSource = ref([]);
    const availableLanduse = ref([]);

    const selectCategory = ref("");
    const selectMapType = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectSource = ref("");
    const selectLanduse = ref("");

    const previousSelections = ref({
      "Sum": { mapType: "", landuse: "" },
      "Ag": { mapType: "", water: "", source: "", landuse: "" },
      "Ag Mgt": { mapType: "", agMgt: "", water: "", landuse: "" },
      "Non-Ag": { mapType: "", landuse: "" },
    });

    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    const SUM_TYPE_TO_SERIES = { 'ag': 'Agricultural Land-use', 'ag-man': 'Agricultural Management', 'non-ag': 'Non-Agricultural Land-use' };
    function formatLanduse(val) {
      return selectCategory.value === 'Sum' ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const isDrawerOpen = ref(false);

    const hasSourceLevel = computed(() =>
      selectCategory.value === "Ag" && selectMapType.value !== "Profit"
    );
    const TRANSITION_AG = ["Transition (Ag2Ag)", "Transition (NonAg2Ag)"];
    const TRANSITION_NONAG = ["Transition (Ag2NonAg)", "Transition (NonAg2NonAg)"];
    const isTransition = computed(() =>
      (selectCategory.value === "Ag" && TRANSITION_AG.includes(selectMapType.value)) ||
      (selectCategory.value === "Non-Ag" && TRANSITION_NONAG.includes(selectMapType.value))
    );

    // ── Per-combo map layer loader ──────────────────────────────────────────
    const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);

    const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

    const emptyChart = () => ({
      ...window["Chart_default_options"],
      chart: { height: 440 },
      yAxis: { title: { text: availableUnit["Economics"] } },
      series: [],
    });

    const selectChartData = computed(() => {
      const cat = selectCategory.value;
      const mt = selectMapType.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const source = selectSource.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      const hasSrc = hasSourceLevel.value;
      if (!dataLoaded.value) return emptyChart();
      const effectiveMt = cat === "Sum" ? "Profit" : mt;
      const chartEntry = chartRegister[cat]?.[effectiveMt];
      if (!chartEntry) return emptyChart();
      const chartData = window[chartEntry.name]?.[region];
      if (!chartData) return emptyChart();

      let seriesData;
      if (cat === "Sum") {
        const seriesName = SUM_TYPE_TO_SERIES[landuse];
        seriesData = (landuse === "ALL" || !landuse) ? chartData : chartData.filter(s => s.name === seriesName);
      } else if (cat === "Ag") {
        let items;
        if (hasSrc && effectiveMt !== "Profit") {
          items = chartData?.[source || "ALL"]?.[water];
        } else {
          items = chartData?.[water];
        }
        seriesData = (items && items.length)
          ? ((landuse === "ALL" || !landuse) ? items : items.filter(s => s.name === landuse))
          : [];
      } else if (cat === "Ag Mgt") {
        // chart: agMgt → water → [series by LU]
        const items = chartData?.[agMgt]?.[water];
        seriesData = (items && items.length)
          ? ((landuse === "ALL" || !landuse) ? items : items.filter(s => s.name === landuse))
          : [];
      } else if (cat === "Non-Ag") {
        seriesData = (landuse && landuse !== "ALL") ? chartData?.filter(s => s.name === landuse) : chartData;
      }
      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["Economics"] } },
        series: seriesData || [],
      };
    });

    // ── Helpers ──────────────────────────────────────────────────────────────
    function getTree(cat, mapType) {
      return window[mapRegister[cat]?.[mapType]?.indexName]?.tree ?? (cat === "Non-Ag" ? [] : {});
    }

    async function ensureIndexLoaded(cat, mapType) {
      const entry = mapRegister[cat]?.[mapType];
      if (entry && !window[entry.indexName]) {
        isLoadingData.value = true;
        await loadScript(entry.indexPath, entry.indexName, VIEW_NAME);
        isLoadingData.value = false;
      }
    }

    function saveSelections(cat) {
      if (!cat) return;
      if (cat === "Sum") previousSelections.value["Sum"] = { mapType: selectMapType.value, landuse: selectLanduse.value };
      if (cat === "Ag") previousSelections.value["Ag"] = { mapType: selectMapType.value, water: selectWater.value, source: selectSource.value, landuse: selectLanduse.value };
      if (cat === "Ag Mgt") previousSelections.value["Ag Mgt"] = { mapType: selectMapType.value, agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
      if (cat === "Non-Ag") previousSelections.value["Non-Ag"] = { mapType: selectMapType.value, landuse: selectLanduse.value };
    }

    async function cascadeAll(cat, mapType) {
      const tree = getTree(cat, mapType);
      const curW = selectWater.value, curL = selectLanduse.value, curS = selectSource.value, curAm = selectAgMgt.value;

      if (cat === "Sum") {
        availableWater.value = []; selectWater.value = ''; availableSource.value = []; selectSource.value = '';
        availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        const prev = previousSelections.value["Sum"]?.landuse || curL;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Sum"][mapType].layerPrefix, [selectLanduse.value]);

      } else if (cat === "Ag") {
        if (TRANSITION_AG.includes(mapType)) {
          availableWater.value = []; selectWater.value = 'ALL';
          const subTree = tree['ALL'] ?? {};
          if (Array.isArray(subTree)) {
            availableSource.value = []; selectSource.value = '';
            availableLanduse.value = subTree;
            const prev = previousSelections.value["Ag"]?.landuse || curL;
            selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
            await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, ['ALL', selectLanduse.value]);
          } else {
            availableSource.value = Object.keys(subTree);
            const prevS = previousSelections.value["Ag"]?.source || curS;
            selectSource.value = (prevS && availableSource.value.includes(prevS)) ? prevS : (availableSource.value[0] || '');
            availableLanduse.value = subTree[selectSource.value] || [];
            const prev = previousSelections.value["Ag"]?.landuse || curL;
            selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
            await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, ['ALL', selectSource.value, selectLanduse.value]);
          }
        } else if (mapType === "Profit") {
          availableWater.value = Object.keys(tree);
          const prevW = previousSelections.value["Ag"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableSource.value = []; selectSource.value = '';
          availableLanduse.value = tree[selectWater.value] || [];
          const prev = previousSelections.value["Ag"]?.landuse || curL;
          selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [selectWater.value, selectLanduse.value]);
        } else {
          // Revenue/Cost: { water: { source: [lu] } }
          availableWater.value = Object.keys(tree);
          const prevW = previousSelections.value["Ag"]?.water || curW;
          selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
          availableSource.value = Object.keys(tree[selectWater.value] || {});
          const prevS = previousSelections.value["Ag"]?.source || curS;
          selectSource.value = (prevS && availableSource.value.includes(prevS)) ? prevS : (availableSource.value[0] || '');
          availableLanduse.value = tree[selectWater.value]?.[selectSource.value] || [];
          const prev = previousSelections.value["Ag"]?.landuse || curL;
          selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [selectWater.value, selectSource.value, selectLanduse.value]);
        }

      } else if (cat === "Ag Mgt") {
        availableAgMgt.value = Object.keys(tree);
        const prevAm = previousSelections.value["Ag Mgt"]?.agMgt || curAm;
        selectAgMgt.value = (prevAm && availableAgMgt.value.includes(prevAm)) ? prevAm : (availableAgMgt.value[0] || '');
        availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
        const prevW = previousSelections.value["Ag Mgt"]?.water || curW;
        selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
        availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
        const prevL = previousSelections.value["Ag Mgt"]?.landuse || curL;
        selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag Mgt"][mapType].layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);

      } else if (cat === "Non-Ag") {
        availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
        const prev = previousSelections.value["Non-Ag"]?.landuse || curL;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Non-Ag"][mapType].layerPrefix, [selectLanduse.value]);
      }
    }

    async function loadAllCharts() {
      const pending = [];
      for (const mapTypeDict of Object.values(chartRegister)) {
        for (const entry of Object.values(mapTypeDict || {})) {
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

      const initCat = availableCategories[0];
      const initMapType = Object.keys(mapRegister[initCat] || {})[0] || '';
      await Promise.all([ensureIndexLoaded(initCat, initMapType), loadAllCharts()]);

      availableMapTypes.value = Object.keys(mapRegister[initCat] || {});
      selectMapType.value = initMapType;
      await cascadeAll(initCat, initMapType);

      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };
    watch(yearIndex, (i) => { selectYear.value = availableYears.value[i]; });

    watch([selectCategory, selectMapType], async ([newCat, newMapType], [oldCat]) => {
      if (!newCat) return;
      if (oldCat && oldCat !== newCat) saveSelections(oldCat);
      availableMapTypes.value = Object.keys(mapRegister[newCat] || {});
      if (!availableMapTypes.value.includes(newMapType)) {
        const prev = previousSelections.value[newCat]?.mapType;
        const resolved = (prev && availableMapTypes.value.includes(prev)) ? prev : (availableMapTypes.value[0] || '');
        selectMapType.value = resolved;
        return;
      }
      await ensureIndexLoaded(newCat, newMapType);
      await cascadeAll(newCat, newMapType);
    }, { immediate: true });

    watch(selectAgMgt, async (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const mapType = selectMapType.value;
      const tree = getTree("Ag Mgt", mapType);
      availableWater.value = Object.keys(tree[newAgMgt] || {});
      const prevW = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
      availableLanduse.value = tree[newAgMgt]?.[selectWater.value] || [];
      const prevL = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');
      await ensureComboLayer(mapRegister["Ag Mgt"][mapType].layerPrefix, [newAgMgt, selectWater.value, selectLanduse.value]);
    });

    watch(selectWater, async (newWater) => {
      const cat = selectCategory.value;
      const mapType = selectMapType.value;
      if (cat === "Ag") {
        if (isTransition.value) return;
        previousSelections.value["Ag"].water = newWater;
        const tree = getTree("Ag", mapType);
        if (mapType === "Profit") {
          availableLanduse.value = tree[newWater] || [];
          const prev = previousSelections.value["Ag"].landuse;
          selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [newWater, selectLanduse.value]);
        } else {
          availableSource.value = Object.keys(tree[newWater] || {});
          const prevS = previousSelections.value["Ag"].source;
          selectSource.value = (prevS && availableSource.value.includes(prevS)) ? prevS : (availableSource.value[0] || '');
          availableLanduse.value = tree[newWater]?.[selectSource.value] || [];
          const prev = previousSelections.value["Ag"].landuse;
          selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [newWater, selectSource.value, selectLanduse.value]);
        }
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const tree = getTree("Ag Mgt", mapType);
        availableLanduse.value = tree[selectAgMgt.value]?.[newWater] || [];
        const prev = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
        await ensureComboLayer(mapRegister["Ag Mgt"][mapType].layerPrefix, [selectAgMgt.value, newWater, selectLanduse.value]);
      }
    });

    watch(selectSource, async (newSource) => {
      if (selectCategory.value !== "Ag" || !hasSourceLevel.value) return;
      previousSelections.value["Ag"].source = newSource;
      const mapType = selectMapType.value;
      const tree = getTree("Ag", mapType);
      if (TRANSITION_AG.includes(mapType)) {
        availableLanduse.value = (tree['ALL'] ?? {})[newSource] || [];
      } else {
        availableLanduse.value = tree[selectWater.value]?.[newSource] || [];
      }
      const prev = previousSelections.value["Ag"].landuse;
      selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
      const waterKey = TRANSITION_AG.includes(mapType) ? 'ALL' : selectWater.value;
      await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [waterKey, newSource, selectLanduse.value]);
    });

    watch(selectLanduse, async (newLanduse) => {
      const cat = selectCategory.value;
      const mapType = selectMapType.value;
      if (!cat || !mapType) return;
      if (cat === "Sum") {
        previousSelections.value["Sum"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Sum"][mapType].layerPrefix, [newLanduse]);
      } else if (cat === "Ag") {
        previousSelections.value["Ag"].landuse = newLanduse;
        if (mapType === "Profit") {
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [selectWater.value, newLanduse]);
        } else if (TRANSITION_AG.includes(mapType)) {
          const args = availableSource.value.length > 0
            ? ['ALL', selectSource.value, newLanduse] : ['ALL', newLanduse];
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, args);
        } else {
          await ensureComboLayer(mapRegister["Ag"][mapType].layerPrefix, [selectWater.value, selectSource.value, newLanduse]);
        }
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Ag Mgt"][mapType].layerPrefix, [selectAgMgt.value, selectWater.value, newLanduse]);
      } else if (cat === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newLanduse;
        await ensureComboLayer(mapRegister["Non-Ag"][mapType].layerPrefix, [newLanduse]);
      }
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears, availableCategories, availableMapTypes,
      availableAgMgt, availableWater, availableSource, availableLanduse,
      selectCategory, selectMapType, selectAgMgt, selectWater, selectSource, selectLanduse,
      hasSourceLevel, isTransition,
      selectMapData, selectChartData, formatLanduse,
      dataLoaded, isLoadingData, isDrawerOpen, toggleDrawer,
    };
    const _fn = v => String(v).trim().replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    _state.mapFileName = computed(() =>
      [VIEW_NAME, selectCategory.value, selectMapType.value, selectAgMgt.value, selectWater.value, selectSource.value, selectLanduse.value, selectYear.value]
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

        <!-- 1. Category -->
        <div class="flex space-x-1">
          <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
          <button v-for="(val, key) in availableCategories" :key="key"
            @click="selectCategory = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
            :class="{'bg-sky-500 text-white': selectCategory === val}">
            {{ val }}
          </button>
        </div>

        <!-- 2. Map Type (dynamic per category, hidden for Sum since it only has Profit) -->
        <div v-if="selectCategory !== 'Sum' && dataLoaded && availableMapTypes.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Map Type:</span>
          <button v-for="(val, key) in availableMapTypes" :key="key"
            @click="selectMapType = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectMapType === val}">
            {{ val }}
          </button>
        </div>

        <!-- 3. Ag Mgt (only for Ag Mgt category) -->
        <div v-if="selectCategory === 'Ag Mgt' && dataLoaded && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>

        <!-- 4. Water (Ag and Ag Mgt, not Transition or Non-Ag or Sum) -->
        <div v-if="!isTransition && selectCategory !== 'Non-Ag' && selectCategory !== 'Sum' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- 5. Source (Ag non-Profit only) -->
        <div v-if="hasSourceLevel && dataLoaded && availableSource.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Source:</span>
          <button v-for="(val, key) in availableSource" :key="key"
            @click="selectSource = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectSource === val}">
            {{ val }}
          </button>
        </div>

        <!-- 6. Landuse -->
        <div v-if="dataLoaded && availableLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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