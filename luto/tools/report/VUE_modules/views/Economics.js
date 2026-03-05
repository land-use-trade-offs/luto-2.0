window.EconomicsView = {
  name: 'EconomicsView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.ChartService.chartCategories["Economics"];
    const mapRegister = window.MapService.mapCategories["Economics"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Economics";

    // Global selection state
    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    // Available variables
    const availableYears = ref([]);
    const availableUnit = { Economics: "AUD" };

    // Selection options (static & dynamic)
    const availableCategories = ["Ag", "Ag Mgt", "Non-Ag"];
    const availableMapTypes = ref([]);   // populated from mapRegister[category] keys
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableSource = ref([]);     // only for Ag non-Profit map types
    const availableLanduse = ref([]);

    // Current selections
    const selectCategory = ref("");
    const selectMapType = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectSource = ref("");
    const selectLanduse = ref("");

    // Previous selection memory per category
    const previousSelections = ref({
      "Ag":     { mapType: "", water: "", source: "", landuse: "" },
      "Ag Mgt": { mapType: "", agMgt: "", water: "", landuse: "" },
      "Non-Ag": { mapType: "", landuse: "" },
    });

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);

    // Source level is only present for Ag + non-Profit map types
    const hasSourceLevel = computed(() =>
      selectCategory.value === "Ag" && selectMapType.value !== "Profit"
    );

    // Transition map types have LU as their first dim (from_lu), not lm/Water.
    // Collapse it by always selecting 'ALL' and hiding the Water row.
    const isTransition = computed(() =>
      selectCategory.value === "Ag" &&
      (selectMapType.value === "Transition (Ag2Ag)" || selectMapType.value === "Transition (Ag2NonAg)")
    );

    // ── Computed map data ────────────────────────────────────────────────────
    const selectMapData = computed(() => {
      if (!dataLoaded.value) return {};
      const mapData = window[mapRegister[selectCategory.value]?.[selectMapType.value]?.name];
      if (!mapData) return {};
      const yr = String(selectYear.value);
      const cat = selectCategory.value;

      if (cat === "Ag") {
        if (hasSourceLevel.value) {
          return mapData?.[selectWater.value]?.[selectSource.value]?.[selectLanduse.value]?.[yr] || {};
        } else {
          return mapData?.[selectWater.value]?.[selectLanduse.value]?.[yr] || {};
        }
      } else if (cat === "Ag Mgt") {
        return mapData?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[yr] || {};
      } else if (cat === "Non-Ag") {
        return mapData?.[selectLanduse.value]?.[yr] || {};
      }
      return {};
    });

    const selectChartData = computed(() => {
      if (!dataLoaded.value) return {};
      const cat = selectCategory.value;
      const chartData = window[chartRegister[cat]?.name]?.[selectRegion.value];
      let seriesData;
      if (cat === "Ag" || cat === "Ag Mgt") {
        seriesData = chartData?.["ALL"]?.["ALL"];
      } else if (cat === "Non-Ag") {
        seriesData = chartData;
      }
      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["Economics"] } },
        series: seriesData || [],
      };
    });

    // ── Helpers ──────────────────────────────────────────────────────────────

    // Save current selections for a category before switching away
    function saveSelections(cat) {
      if (!cat) return;
      if (cat === "Ag") {
        previousSelections.value["Ag"] = {
          mapType: selectMapType.value,
          water: selectWater.value,
          source: selectSource.value,
          landuse: selectLanduse.value,
        };
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"] = {
          mapType: selectMapType.value,
          agMgt: selectAgMgt.value,
          water: selectWater.value,
          landuse: selectLanduse.value,
        };
      } else if (cat === "Non-Ag") {
        previousSelections.value["Non-Ag"] = {
          mapType: selectMapType.value,
          landuse: selectLanduse.value,
        };
      }
    }

    // Cascade source + landuse from water level (Ag only)
    function cascadeAgFromWater(mapData, water, isSource) {
      const waterData = mapData?.[water] || {};
      if (isSource) {
        availableSource.value = Object.keys(waterData);
        const prev = previousSelections.value["Ag"].source;
        selectSource.value = (prev && availableSource.value.includes(prev)) ? prev : (availableSource.value[0] || '');
        availableLanduse.value = Object.keys(waterData[selectSource.value] || {});
      } else {
        availableSource.value = [];
        selectSource.value = '';
        availableLanduse.value = Object.keys(waterData);
      }
      const prev = previousSelections.value["Ag"].landuse;
      selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
    }

    // Full cascade from the top level for a given category + mapType
    function cascadeAll(cat, mapType) {
      const mapData = window[mapRegister[cat]?.[mapType]?.name];
      if (!mapData) return;

      if (cat === "Ag") {
        if (mapType === "Transition (Ag2Ag)" || mapType === "Transition (Ag2NonAg)") {
          availableWater.value = [];
          selectWater.value = 'ALL';
          cascadeAgFromWater(mapData, 'ALL', true);
        } else {
          availableWater.value = Object.keys(mapData);
          const prev = previousSelections.value["Ag"].water;
          selectWater.value = (prev && availableWater.value.includes(prev)) ? prev : (availableWater.value[0] || '');
          cascadeAgFromWater(mapData, selectWater.value, mapType !== "Profit");
        }

      } else if (cat === "Ag Mgt") {
        availableAgMgt.value = Object.keys(mapData);
        const prevAm = previousSelections.value["Ag Mgt"].agMgt;
        selectAgMgt.value = (prevAm && availableAgMgt.value.includes(prevAm)) ? prevAm : (availableAgMgt.value[0] || '');

        const agMgtData = mapData[selectAgMgt.value] || {};
        availableWater.value = Object.keys(agMgtData);
        const prevW = previousSelections.value["Ag Mgt"].water;
        selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(agMgtData[selectWater.value] || {});
        const prevL = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');

      } else if (cat === "Non-Ag") {
        availableLanduse.value = Object.keys(mapData);
        const prev = previousSelections.value["Non-Ag"].landuse;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
      }
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────
    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      // Load all economics map layers
      for (const mapTypes of Object.values(mapRegister)) {
        for (const entry of Object.values(mapTypes)) {
          await loadScript(entry.path, entry.name, VIEW_NAME);
        }
      }
      // Load chart data
      await loadScript(chartRegister["Ag"]["path"],     chartRegister["Ag"]["name"],     VIEW_NAME);
      await loadScript(chartRegister["Ag Mgt"]["path"], chartRegister["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(chartRegister["Non-Ag"]["path"], chartRegister["Non-Ag"]["name"], VIEW_NAME);

      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;
      selectCategory.value = availableCategories[0]; // triggers cascade watcher

      await nextTick(() => { dataLoaded.value = true; });
    });

    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    // ── Watchers ─────────────────────────────────────────────────────────────
    const toggleDrawer = () => { isDrawerOpen.value = !isDrawerOpen.value; };

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    // Combined watcher: Category + MapType drive all downstream options
    watch([selectCategory, selectMapType], ([newCat, newMapType], [oldCat]) => {
      if (!newCat) return;

      // Save previous selections when category changes
      if (oldCat && oldCat !== newCat) {
        saveSelections(oldCat);
      }

      // Update available map types for the current category
      availableMapTypes.value = Object.keys(mapRegister[newCat] || {});

      // If current mapType is invalid for this category, restore previous or use first
      if (!availableMapTypes.value.includes(newMapType)) {
        const prev = previousSelections.value[newCat]?.mapType;
        const resolved = (prev && availableMapTypes.value.includes(prev))
          ? prev
          : (availableMapTypes.value[0] || '');
        selectMapType.value = resolved;
        return; // Watcher re-fires with the resolved mapType
      }

      cascadeAll(newCat, newMapType);
    }, { immediate: true });

    // AgMgt → Water → Landuse (Ag Mgt only)
    watch(selectAgMgt, (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const mapData = window[mapRegister["Ag Mgt"][selectMapType.value]?.name];
      const agMgtData = mapData?.[newAgMgt] || {};
      availableWater.value = Object.keys(agMgtData);
      const prevW = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevW && availableWater.value.includes(prevW)) ? prevW : (availableWater.value[0] || '');
      availableLanduse.value = Object.keys(agMgtData[selectWater.value] || {});
      const prevL = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevL && availableLanduse.value.includes(prevL)) ? prevL : (availableLanduse.value[0] || '');
    });

    // Water → Source → Landuse
    watch(selectWater, (newWater) => {
      const cat = selectCategory.value;
      if (cat === "Ag") {
        if (selectMapType.value === "Transition (Ag2Ag)" || selectMapType.value === "Transition (Ag2NonAg)") return;
        previousSelections.value["Ag"].water = newWater;
        const mapData = window[mapRegister["Ag"][selectMapType.value]?.name];
        cascadeAgFromWater(mapData, newWater, selectMapType.value !== "Profit");
      } else if (cat === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const mapData = window[mapRegister["Ag Mgt"][selectMapType.value]?.name];
        availableLanduse.value = Object.keys(mapData?.[selectAgMgt.value]?.[newWater] || {});
        const prev = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
      }
    });

    // Source → Landuse (Ag non-Profit only)
    watch(selectSource, (newSource) => {
      if (selectCategory.value !== "Ag" || !hasSourceLevel.value) return;
      previousSelections.value["Ag"].source = newSource;
      const mapData = window[mapRegister["Ag"][selectMapType.value]?.name];
      availableLanduse.value = Object.keys(mapData?.[selectWater.value]?.[newSource] || {});
      const prev = previousSelections.value["Ag"].landuse;
      selectLanduse.value = (prev && availableLanduse.value.includes(prev)) ? prev : (availableLanduse.value[0] || '');
    });

    // Landuse → just save
    watch(selectLanduse, (newLanduse) => {
      const cat = selectCategory.value;
      if (cat === "Ag")     previousSelections.value["Ag"].landuse = newLanduse;
      else if (cat === "Ag Mgt") previousSelections.value["Ag Mgt"].landuse = newLanduse;
      else if (cat === "Non-Ag") previousSelections.value["Non-Ag"].landuse = newLanduse;
    });

    const _state = {
      yearIndex, selectYear, selectRegion,
      availableYears, availableCategories, availableMapTypes,
      availableAgMgt, availableWater, availableSource, availableLanduse,
      selectCategory, selectMapType, selectAgMgt, selectWater, selectSource, selectLanduse,
      hasSourceLevel, isTransition,
      selectMapData, selectChartData,
      dataLoaded, isDrawerOpen, toggleDrawer,
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

        <!-- 2. Map Type (dynamic per category) -->
        <div v-if="dataLoaded && availableMapTypes.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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

        <!-- 4. Water (Ag and Ag Mgt only, not Transition) -->
        <div v-if="!isTransition && selectCategory !== 'Non-Ag' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
