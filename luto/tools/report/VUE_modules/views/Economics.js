window.EconomicsView = {
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service
    const chartRegister = window.DataService.chartCategories["Economics"];
    const mapRegister = window.MapService.mapCategories["Economics"];
    const loadScript = window.loadScriptWithTracking;
    const mosaicMap = window.MapService.mapCategories["Dvar"]["Mosaic"];
    
    // View identification for memory management
    const VIEW_NAME = "Economics";

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

    // Available selections for Economics
    const availableCostRevenue = ["Cost", "Revenue"];
    const availableCategories = ["Ag", "Ag Mgt", "Non-Ag"];
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableLanduse = ref([]);

    // Map selection state (Cost/Revenue only affects map, not chart)
    const selectCostRevenue = ref("Cost");
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
    const isDrawerOpen = ref(false);

    // Reactive data
    const mapData = computed(() => {
      if (!dataLoaded.value) {
        return {};
      }
      let mapDataWithALL = JSON.parse(JSON.stringify(window[mapRegister[selectCostRevenue.value][selectCategory.value]?.name]));
      let mapDataWithMosaic = JSON.parse(JSON.stringify(window[mosaicMap['name']]));
      if (selectCategory.value === "Ag") {
        mapDataWithALL[selectWater.value] = {
          "ALL": mapDataWithMosaic['Agricultural Land-use'],
          ...mapDataWithALL[selectWater.value]
        }
      } else if (selectCategory.value === "Ag Mgt") {
        mapDataWithALL[selectAgMgt.value][selectWater.value] = {
          "ALL": mapDataWithMosaic['Agricultural Management'],
          ...mapDataWithALL[selectAgMgt.value][selectWater.value]
        }
      } else if (selectCategory.value === "Non-Ag") {
        mapDataWithALL = {
          "ALL": mapDataWithMosaic['Non-agricultural Land-use'],
          ...mapDataWithALL
        }
      }
      return mapDataWithALL;
    });
    const chartData = computed(() => window[chartRegister[selectCategory.value]?.name]?.[selectRegion.value]);
    const selectMapData = computed(() => {
      if (!dataLoaded.value) {
        return {};
      }
      if (selectCategory.value === "Ag") {
        return mapData.value?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      }
      else if (selectCategory.value === "Ag Mgt") {
        return mapData.value?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[selectYear.value] || {};
      }
      else if (selectCategory.value === "Non-Ag") {
        return mapData.value?.[selectLanduse.value]?.[selectYear.value] || {};
      }
      return {};
    });
    const selectChartData = computed(() => {
      if (!dataLoaded.value) {
        return {};
      }
      let seriesData;
      // Chart always shows BOTH cost and revenue (ignores selectCostRevenue)
      if (selectCategory.value === "Ag") {
        // Economics_Ag chart structure is Region → "ALL" → "ALL" → [series] (aggregated, same as Am)
        seriesData = chartData.value?.["ALL"]?.["ALL"];
      }
      else if (selectCategory.value === "Ag Mgt") {
        // Economics_Am chart structure is Region → "ALL" → "ALL" → [series] (aggregated)
        seriesData = chartData.value?.["ALL"]?.["ALL"];
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
            text: availableUnit["Economics"],
          },
        },
        series: seriesData || [],
        colors: window["Supporting_info"].colors,
      };
    });

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);
      await loadScript(mosaicMap['path'], mosaicMap['name'], VIEW_NAME);

      // Load data
      await loadScript(mapRegister["Cost"]["Ag"]["path"], mapRegister["Cost"]["Ag"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Cost"]["Ag Mgt"]["path"], mapRegister["Cost"]["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Cost"]["Non-Ag"]["path"], mapRegister["Cost"]["Non-Ag"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Revenue"]["Ag"]["path"], mapRegister["Revenue"]["Ag"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Revenue"]["Ag Mgt"]["path"], mapRegister["Revenue"]["Ag Mgt"]["name"], VIEW_NAME);
      await loadScript(mapRegister["Revenue"]["Non-Ag"]["path"], mapRegister["Revenue"]["Non-Ag"]["name"], VIEW_NAME);
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

    // Progressive selection chain watchers (replaced by combined watcher below)

    // Combined watcher for Cost/Revenue changes - directly updates options
    watch([selectCostRevenue, selectCategory], ([newCostRevenue, newCategory], [oldCostRevenue, oldCategory]) => {
      if (!newCategory) return;

      // Save previous selections before switching (only when category changes)
      if (oldCategory && oldCategory !== newCategory) {
        if (oldCategory === "Ag") {
          previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
        } else if (oldCategory === "Ag Mgt") {
          previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
        } else if (oldCategory === "Non-Ag") {
          previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
        }
      }
      
      if (newCategory === "Ag Mgt") {
        const currentMapData = window[mapRegister[newCostRevenue]["Ag Mgt"]["name"]];
        const newAvailableAgMgt = Object.keys(currentMapData || {});
        availableAgMgt.value = newAvailableAgMgt;
        
        // Restore previous AgMgt selection if valid, otherwise use first available
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt;
        selectAgMgt.value = (prevAgMgt && newAvailableAgMgt.includes(prevAgMgt)) ? prevAgMgt : (newAvailableAgMgt[0] || '');
        
        if (selectAgMgt.value) {
          availableWater.value = Object.keys(currentMapData[selectAgMgt.value] || {});
          const prevWater = previousSelections.value["Ag Mgt"].water;
          selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
          
          availableLanduse.value = ["ALL", ...Object.keys(currentMapData[selectAgMgt.value][selectWater.value] || {})];
          const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
          selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
        }
      } else if (newCategory === "Ag") {
        const currentMapData = window[mapRegister[newCostRevenue]["Ag"]["name"]];
        availableWater.value = Object.keys(currentMapData || {});
        const prevWater = previousSelections.value["Ag"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
        
        availableLanduse.value = ["ALL", ...Object.keys(currentMapData[selectWater.value] || {})];
        const prevLanduse = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
      } else if (newCategory === "Non-Ag") {
        const currentMapData = window[mapRegister[newCostRevenue]["Non-Ag"]["name"]];
        availableLanduse.value = ["ALL", ...Object.keys(currentMapData || {})];
        const prevLanduse = previousSelections.value["Non-Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
      }
    }, { immediate: true });

    watch(selectAgMgt, (newAgMgt) => {
      // Save current agMgt selection
      if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
        
        // Handle ALL downstream variables with cascading pattern
        const currentMapData = window[mapRegister[selectCostRevenue.value]["Ag Mgt"]["name"]];
        availableWater.value = Object.keys(currentMapData[newAgMgt] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
        
        availableLanduse.value = ["ALL", ...Object.keys(currentMapData[newAgMgt][selectWater.value] || {})];
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
      }
    });

    watch(selectWater, (newWater) => {
      // Save current water selection
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].water = newWater;
      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
      }

      // Handle downstream variables
      if (selectCategory.value === "Ag") {
        const currentMapData = window[mapRegister[selectCostRevenue.value]["Ag"]["name"]];
        availableLanduse.value = ["ALL", ...Object.keys(currentMapData[newWater] || {})];
        const prevLanduse = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
      } else if (selectCategory.value === "Ag Mgt") {
        const currentMapData = window[mapRegister[selectCostRevenue.value]["Ag Mgt"]["name"]];
        availableLanduse.value = ["ALL", ...Object.keys(currentMapData[selectAgMgt.value][newWater] || {})];
        const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
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

    return {
      yearIndex,
      selectYear,
      selectRegion,

      availableYears,
      availableCostRevenue,
      availableCategories,
      availableAgMgt,
      availableWater,
      availableLanduse,

      selectCostRevenue,
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

        <!-- Cost/Revenue buttons (always visible, affects MAP only) -->
        <div class="flex items-center">
          <div class="flex space-x-1">
            <span class="text-[0.8rem] mr-1 font-medium">Map Type:</span>
            <button v-for="(val, key) in availableCostRevenue" :key="key"
              @click="selectCostRevenue = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
              :class="{'bg-sky-500 text-white': selectCostRevenue === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Category buttons (always visible) -->
        <div class="flex items-center border-t border-white/10 pt-1">
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
        <div v-if="selectCategory === 'Ag Mgt'"
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
          <div v-if="selectCategory !== 'Non-Ag' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

    </div>
  `,
};