window.ProductionView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.DataService.chartCategories["Production"];
    const mapRegister = window.MapService.mapCategories["Production"];
    const loadScript = window.loadScript;

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
      Production: "Tonnes",
      Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    // Available selections - Production has different structure than Area
    const availableCategories = ["Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]); // Only for Ag Mgt category

    // Map selection state - simpler than Area
    const selectCategory = ref("");
    const selectAgMgt = ref(""); // Only used for Ag Mgt category

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);
    
    // Production data structure is much simpler - check data readiness
    const mapReady = computed(() => {
      if (!selectCategory.value) {
        return false;
      }
      // Ag Mgt needs AgMgt selection, others just need category
      if (selectCategory.value === "Ag Mgt" && !selectAgMgt.value) {
        return false;
      }
      const dataName = mapRegister[selectCategory.value]?.["name"];
      return dataName && window[dataName];
    });
    
    const chartReady = computed(() => {
      if (!selectCategory.value || !selectRegion.value) {
        return false;
      }
      const dataName = chartRegister[selectCategory.value]?.["name"];
      return dataName && window[dataName] && window[dataName][selectRegion.value];
    });

    // Reactive data - Production has simpler structure
    const mapData = computed(() => window[mapRegister[selectCategory.value]["name"]]);
    const chartData = computed(() => window[chartRegister[selectCategory.value]["name"]][selectRegion.value]);
    
    const selectMapData = computed(() => {
      if (!mapReady.value) {
        return {};
      }
      // Production data structure is simpler than Area
      if (selectCategory.value === "Ag" || selectCategory.value === "Non-Ag") {
        // Ag and Non-Ag: Year > {img_str, bounds, min_max}
        return mapData.value[selectYear.value];
      }
      else if (selectCategory.value === "Ag Mgt") {
        // Ag Mgt: AgMgt > Year > {img_str, bounds, min_max}
        return mapData.value?.[selectAgMgt.value]?.[selectYear.value];
      }
    });
    
    const selectChartData = computed(() => {
      if (!chartReady.value) {
        return {};
      }
      // Production chart data is simple: Region > [series array]
      const seriesData = chartData.value;

      return {
        ...window["Chart_default_options"],
        chart: {
          height: 440,
        },
        yAxis: {
          title: {
            text: availableUnit["Production"],
          },
        },
        series: seriesData || [],
        colors: window["Supporting_info"].colors,
      };
    });

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info");
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options");

      // Load data
      await loadScript(mapRegister["Ag"]["path"], mapRegister["Ag"]["name"]);
      await loadScript(mapRegister["Ag Mgt"]["path"], mapRegister["Ag Mgt"]["name"]);
      await loadScript(mapRegister["Non-Ag"]["path"], mapRegister["Non-Ag"]["name"]);
      await loadScript(chartRegister["Ag"]["path"], chartRegister["Ag"]["name"]);
      await loadScript(chartRegister["Ag Mgt"]["path"], chartRegister["Ag Mgt"]["name"]);
      await loadScript(chartRegister["Non-Ag"]["path"], chartRegister["Non-Ag"]["name"]);

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

    // Production has simpler selection logic than Area
    watch(selectCategory, (newCategory) => {
      // Only Ag Mgt has sub-levels
      if (newCategory === "Ag Mgt") {
        availableAgMgt.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]] || {});
        selectAgMgt.value = availableAgMgt.value[0] || '';
      } else {
        // Ag and Non-Ag have no sub-levels
        availableAgMgt.value = [];
        selectAgMgt.value = '';
      }
    });

    return {
      yearIndex,
      selectYear,
      selectRegion,

      availableYears,
      availableCategories,
      availableAgMgt,

      selectCategory,
      selectAgMgt,

      selectMapData,
      selectChartData,

      dataLoaded,
      isDrawerOpen,
      toggleDrawer,
    };
  },
  template: `
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
          :show-tooltip="false"
          :min="0"
          :max="availableYears.length - 1"
          :step="1"
          :format-tooltip="index => availableYears[index]"
          :marks="availableYears.reduce((acc, year, index) => ({ ...acc, [index]: year }), {})"
          @input="(index) => { yearIndex = index; selectYear = availableYears[index]; }"
        />
      </div>

      <!-- Data selection controls container -->
      <div class="absolute top-[285px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

        <!-- Category buttons (always visible) -->
        <div class="flex items-center">
          <div class="flex space-x-1">
            <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
            <button v-for="(val, key) in availableCategories" :key="key"
              @click="selectCategory = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
              :class="{'bg-sky-500 text-white': selectCategory === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category) -->
        <div 
          class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Production Method:</span>
            <button v-for="(val, key) in availableAgMgt" :key="key"
              @click="selectAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectAgMgt === val}">
              {{ val }}
            </button>
          </div>
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