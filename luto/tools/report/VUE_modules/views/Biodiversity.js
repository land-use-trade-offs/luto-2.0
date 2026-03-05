window.BiodiversityView = {
  name: 'BiodiversityView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service — top-level keys are metric names
    const chartRegister = window.ChartService.chartCategories["Biodiversity"];
    const mapRegister = window.MapService.mapCategories["Biodiversity"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Biodiversity";

    // Global selection state
    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    // Available variables
    const availableYears = ref([]);
    const availableUnit = {
      Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    // Metric display labels (internal keys must match MapService/ChartService)
    const METRIC_LABELS = {
      'quality': 'Quality',
      'GBF2': 'GBF2',
      'GBF3_NVIS': 'GBF3 NVIS',
      'GBF3_IBRA': 'GBF3 IBRA',
      'GBF4_SNES': 'GBF4 SNES',
      'GBF4_ECNES': 'GBF4 ECNES',
      'GBF8_GROUP': 'GBF8 Group',
      'GBF8_SPECIES': 'GBF8 Species',
    };

    // Metric → Supporting_info setting key (null = always available)
    const METRIC_TO_SETTING = {
      'quality': null,
      'GBF2': 'BIODIVERSITY_TARGET_GBF_2',
      'GBF3_NVIS': 'BIODIVERSITY_TARGET_GBF_3_NVIS',
      'GBF3_IBRA': 'BIODIVERSITY_TARGET_GBF_3_IBRA',
      'GBF4_SNES': 'BIODIVERSITY_TARGET_GBF_4_SNES',
      'GBF4_ECNES': 'BIODIVERSITY_TARGET_GBF_4_ECNES',
      'GBF8_GROUP': 'BIODIVERSITY_TARGET_GBF_8',
      'GBF8_SPECIES': 'BIODIVERSITY_TARGET_GBF_8',
    };

    // Available selections
    const availableMetrics = ref(['quality']);
    const availableCategories = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    // Map selection state
    const selectMetric = ref("quality");
    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectLanduse = ref("");

    // Previous selections memory (per category)
    const previousSelections = ref({
      "Sum": { landuse: "" },
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    // Display labels for the "Sum" category's Type dimension
    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    // Map Type key → series name in BIO_*_overview_sum chart data
    const SUM_TYPE_TO_SERIES = {
      'ag': 'Agricultural Land-use',
      'ag-man': 'Agricultural Management',
      'non-ag': 'Non-Agricultural Land-use',
    };
    function formatLanduse(val) {
      return (selectCategory.value === 'Sum') ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);

    // Cascade helper — reads from mapRegister[selectMetric]
    function doCascade(category) {
      const mr = mapRegister[selectMetric.value];
      const sumData = window[mr?.["Sum"]?.["name"]];
      const agData = window[mr?.["Ag"]?.["name"]];
      const amData = window[mr?.["Ag Mgt"]?.["name"]];
      const nonAgData = window[mr?.["Non-Ag"]?.["name"]];

      if (category === "Sum") {
        availableAgMgt.value = [];
        availableWater.value = [];
        availableLanduse.value = Object.keys(sumData || {});
        const prevLU = previousSelections.value["Sum"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Ag") {
        availableWater.value = Object.keys(agData || {});
        const prevWater = previousSelections.value["Ag"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(agData?.[selectWater.value] || {});
        const prevLU = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Ag Mgt") {
        availableAgMgt.value = Object.keys(amData || {});
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');

        availableWater.value = Object.keys(amData?.[selectAgMgt.value] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[selectWater.value] || {});
        const prevLU = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Non-Ag") {
        availableLanduse.value = Object.keys(nonAgData || {});
        const prevLU = previousSelections.value["Non-Ag"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
      }
    }

    // Reactive data
    // map_bio_*_All:   Type → Year  (Sum category)
    // map_bio_*_Ag:    Water → LU → Year
    // map_bio_*_Am:    AgMgt → Water → LU → Year
    // map_bio_*_NonAg: LU → Year
    const selectMapData = computed(() => {
      if (!dataLoaded.value) return {};
      const mr = mapRegister[selectMetric.value];
      const mapData = window[mr?.[selectCategory.value]?.["name"]];
      if (selectCategory.value === "Sum") {
        return mapData?.[selectLanduse.value]?.[selectYear.value] || {};
      } else if (selectCategory.value === "Ag") {
        return mapData?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      } else if (selectCategory.value === "Ag Mgt") {
        return mapData?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      } else if (selectCategory.value === "Non-Ag") {
        return mapData?.[selectLanduse.value]?.[selectYear.value] || {};
      }
      return {};
    });

    // BIO_*_Ag chart:    Region → Water → [series(name=LU)]
    // BIO_*_Am chart:    Region → AgMgt → Water → [series(name=LU)]
    // BIO_*_NonAg chart: Region → [series(name=LU)]
    // Sum: no chart data available
    const selectChartData = computed(() => {
      if (!dataLoaded.value) return {};
      const cr = chartRegister[selectMetric.value];
      const chartData = window[cr?.[selectCategory.value]?.["name"]]?.[selectRegion.value];
      let seriesData;

      if (selectCategory.value === "Sum") {
        const sumEntry = chartRegister[selectMetric.value]?.['overview']?.['sum'];
        const sumData = window[sumEntry?.['name']]?.[selectRegion.value] || [];
        const filterName = SUM_TYPE_TO_SERIES[selectLanduse.value];
        seriesData = filterName ? sumData.filter(s => s.name === filterName) : sumData;
      } else if (selectCategory.value === "Ag") {
        seriesData = chartData?.[selectWater.value] || [];
        seriesData = seriesData.filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      } else if (selectCategory.value === "Ag Mgt") {
        seriesData = chartData?.[selectAgMgt.value]?.[selectWater.value] || [];
        seriesData = seriesData.filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      } else if (selectCategory.value === "Non-Ag") {
        seriesData = (chartData || []).filter(s => selectLanduse.value === "ALL" || s.name === selectLanduse.value);
      }

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        yAxis: { title: { text: availableUnit["Biodiversity"] } },
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

      // Determine enabled metrics from run scenario settings
      const runScenario = {};
      (window.Supporting_info.model_run_settings || []).forEach(s => {
        runScenario[s.parameter] = s.val;
      });

      const enabledMetrics = ['quality'];
      for (const [metric, settingKey] of Object.entries(METRIC_TO_SETTING)) {
        if (metric === 'quality') continue;
        if (settingKey && runScenario[settingKey] !== 'off' && mapRegister[metric]) {
          enabledMetrics.push(metric);
        }
      }
      availableMetrics.value = enabledMetrics;

      // Load all enabled metrics
      for (const metric of enabledMetrics) {
        const mr = mapRegister[metric];
        const cr = chartRegister[metric];
        if (mr) {
          if (mr["Sum"]) await loadScript(mr["Sum"]["path"], mr["Sum"]["name"], VIEW_NAME);
          await loadScript(mr["Ag"]["path"], mr["Ag"]["name"], VIEW_NAME);
          await loadScript(mr["Ag Mgt"]["path"], mr["Ag Mgt"]["name"], VIEW_NAME);
          await loadScript(mr["Non-Ag"]["path"], mr["Non-Ag"]["name"], VIEW_NAME);
        }
        if (cr) {
          await loadScript(cr["Ag"]["path"], cr["Ag"]["name"], VIEW_NAME);
          await loadScript(cr["Ag Mgt"]["path"], cr["Ag Mgt"]["name"], VIEW_NAME);
          await loadScript(cr["Non-Ag"]["path"], cr["Non-Ag"]["name"], VIEW_NAME);
          if (cr["overview"]?.["sum"]) await loadScript(cr["overview"]["sum"]["path"], cr["overview"]["sum"]["name"], VIEW_NAME);
        }
      }

      // Initial selections
      availableYears.value = window.Supporting_info.years;
      selectMetric.value = enabledMetrics[0];
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

    // Metric change: re-cascade based on current category
    watch(selectMetric, () => {
      doCascade(selectCategory.value);
    });

    // Progressive selection chain watchers
    watch(selectCategory, (newCategory, oldCategory) => {
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
      doCascade(newCategory);
    });

    watch(selectAgMgt, (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const amData = window[mapRegister[selectMetric.value]?.["Ag Mgt"]?.["name"]];

      availableWater.value = Object.keys(amData?.[newAgMgt] || {});
      const prevWater = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

      availableLanduse.value = Object.keys(amData?.[newAgMgt]?.[selectWater.value] || {});
      const prevLU = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
    });

    watch(selectWater, (newWater) => {
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].water = newWater;
        const agData = window[mapRegister[selectMetric.value]?.["Ag"]?.["name"]];

        availableLanduse.value = Object.keys(agData?.[newWater] || {});
        const prevLU = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const amData = window[mapRegister[selectMetric.value]?.["Ag Mgt"]?.["name"]];

        availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[newWater] || {});
        const prevLU = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
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

      METRIC_LABELS,
      availableYears,
      availableMetrics,
      availableCategories,
      availableAgMgt,
      availableWater,
      availableLanduse,

      selectMetric,
      selectCategory,
      selectAgMgt,
      selectWater,
      selectLanduse,

      formatLanduse,
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

        <!-- Metric buttons (always visible) -->
        <div class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Metric:</span>
          <button v-for="(val, key) in availableMetrics" :key="key"
            @click="selectMetric = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectMetric === val}">
            {{ METRIC_LABELS[val] || val }}
          </button>
        </div>

        <!-- Category buttons -->
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

        <!-- Water options (Ag and Ag Mgt only) -->
        <div v-if="selectCategory !== 'Non-Ag' && selectCategory !== 'Sum' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Landuse options (for Sum: shows Type dimension values) -->
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
