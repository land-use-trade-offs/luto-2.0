window.GHGView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.DataService.chartCategories["GHG"];
    const mapRegister = window.MapService.mapCategories["GHG"];
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

    // UI state
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);
    const mapReady = computed(() => {
      if (!selectCategory.value || !selectWater.value || !selectLanduse.value) {
        return false;
      }
      if (selectCategory.value === "Ag Mgt" && !selectAgMgt.value) {
        return false;
      }
      const dataName = mapRegister[selectCategory.value]?.["name"];
      return dataName && window[dataName];
    });
    const chartReady = computed(() => {
      if (!selectCategory.value || !selectRegion.value || !selectWater.value || !selectLanduse.value) {
        return false;
      }
      if (selectCategory.value === "Ag Mgt" && !selectAgMgt.value) {
        return false;
      }
      const dataName = chartRegister[selectCategory.value]?.["name"];
      return dataName && window[dataName] && window[dataName][selectRegion.value];
    });

    // Reactive data
    const mapData = computed(() => window[mapRegister[selectCategory.value]["name"]]);
    const chartData = computed(() => window[chartRegister[selectCategory.value]["name"]][selectRegion.value]);
    const selectMapData = computed(() => {
      if (!mapReady.value) {
        return {};
      }
      if (selectCategory.value === "Ag") {
        return mapData.value[selectWater.value][selectLanduse.value][selectYear.value];
      }
      else if (selectCategory.value === "Ag Mgt") {
        return mapData.value?.[selectAgMgt.value][selectLanduse.value][selectWater.value][selectYear.value];
      }
      else if (selectCategory.value === "Non-Ag") {
        return mapData.value[selectLanduse.value][selectYear.value];
      }
    });
    const selectChartData = computed(() => {
      if (!chartReady.value) {
        return {};
      }
      let seriesData;
      if (selectCategory.value === "Ag") {
        seriesData = chartData.value[selectWater.value];
      }
      else if (selectCategory.value === "Ag Mgt") {
        seriesData = chartData.value[selectAgMgt.value]?.[selectWater.value];
      } else if (selectCategory.value === "Non-Ag") {
        seriesData = chartData.value;
      }

      return {
        ...window["Chart_default_options"],
        chart: {
          height: 440,
        },
        yAxis: {
          title: {
            text: availableUnit["GHG"],
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

    // Progressive selection chain watchers
    watch(selectCategory, (newCategory) => {
      // Handle ALL downstream variables with cascading pattern
      if (newCategory === "Ag Mgt") {
        availableAgMgt.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]] || {});
        selectAgMgt.value = availableAgMgt.value[0] || '';
        availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value] || {});
        selectLanduse.value = availableLanduse.value[0] || '';
        availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][selectLanduse.value] || {});
        selectWater.value = availableWater.value[0] || '';
      } else if (newCategory === "Ag") {
        availableWater.value = Object.keys(window[mapRegister["Ag"]["name"]] || {});
        selectWater.value = availableWater.value[0] || '';
        availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][selectWater.value] || {});
        selectLanduse.value = availableLanduse.value[0] || 'ALL';
      } else if (newCategory === "Non-Ag") {
        availableLanduse.value = Object.keys(window[mapRegister["Non-Ag"]["name"]] || {});
        selectLanduse.value = availableLanduse.value[0] || 'ALL';
      }
    });

    watch(selectAgMgt, (newAgMgt) => {
      // Handle ALL downstream variables with cascading pattern
      if (selectCategory.value === "Ag Mgt") {
        availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt] || {});
        selectLanduse.value = availableLanduse.value[0] || '';
        availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt][selectLanduse.value] || {});
        selectWater.value = availableWater.value[0] || '';
      }
    });

    watch(selectLanduse, (newLanduse) => {
      // Handle downstream variables for Ag Mgt
      if (selectCategory.value === "Ag Mgt") {
        availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][newLanduse] || {});
        selectWater.value = availableWater.value[0] || '';
      }
    });

    watch(selectWater, (newWater) => {
      // Handle ALL downstream variables
      if (selectCategory.value === "Ag") {
        availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][newWater] || {});
        selectLanduse.value = availableLanduse.value[0] || 'ALL';
      }
    });

    return {
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
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableAgMgt" :key="key"
              @click="selectAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectAgMgt === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Water options -->
        <div 
          class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableWater" :key="key"
              @click="selectWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectWater === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Landuse options -->
        <div 
          class="flex items-start border-t border-white/10 pt-1">
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
            :selectedLanduse="selectLanduse"
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

    </div>
  `,
};