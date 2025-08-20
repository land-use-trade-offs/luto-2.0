window.WaterView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    const selectRegion = inject('globalSelectedRegion');
    const isDrawerOpen = ref(false);
    const yearIndex = ref(0);

    // Data selection and visualization state
    const selectDataset = ref({});
    const selectChartLevel = ref('Landuse');
    const mapPathName = ref({});
    const mapSelectKey = ref([]);

    // Category selection state
    const selectCategory = ref('Ag');
    const selectAgMgt = ref('Environmental Plantings');
    const selectWater = ref('dry');
    const selectLanduse = ref('Beef - modified land');
    const selectYear = ref(2020);

    const availableYears = ref([]);
    const availableCategories = ref([]);

    const dataLoaded = ref(false);

    // Centralized function to navigate nested data structure based on current selections
    const getNestedData = (path = []) => {
      // Start with the appropriate data object based on category
      let dataSource;
      if (selectCategory.value === 'Ag') {
        dataSource = window.map_water_yield_Ag;
      } else if (selectCategory.value === 'Ag Mgt') {
        dataSource = window.map_water_yield_Am;
      } else if (selectCategory.value === 'Non-Ag') {
        dataSource = window.map_water_yield_NonAg;
      }

      if (!dataSource) return null;

      // Navigate through the nested structure using the provided path
      for (const key of path) {
        if (!dataSource || !dataSource[key]) return null;
        dataSource = dataSource[key];
      }

      return dataSource;
    };

    // Get options for a specific level in the hierarchy
    const getOptionsForLevel = (level) => {
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectCategory.value !== 'Ag Mgt' || !window.map_water_yield_Am) return [];
        return Object.keys(window.map_water_yield_Am);
      }

      if (level === 'water') {
        // Water options depend on category and possibly ag mgt selection
        if (selectCategory.value === 'Ag') {
          return Object.keys(window.map_water_yield_Ag || {});
        } else if (selectCategory.value === 'Ag Mgt') {
          const agMgtData = getNestedData([selectAgMgt.value]);
          return agMgtData ? Object.keys(agMgtData) : [];
        }
        return [];
      }

      if (level === 'landuse') {
        // Landuse options depend on category and previous selections
        if (selectCategory.value === 'Ag') {
          const waterData = getNestedData([selectWater.value]);
          return waterData ? Object.keys(waterData) : [];
        } else if (selectCategory.value === 'Ag Mgt') {
          const waterData = getNestedData([selectAgMgt.value, selectWater.value]);
          return waterData ? Object.keys(waterData) : [];
        } else if (selectCategory.value === 'Non-Ag') {
          return Object.keys(window.map_water_yield_NonAg || {});
        }
        return [];
      }

      return [];
    };

    // Computed properties using the centralized functions
    const availableAgMgt = computed(() => getOptionsForLevel('agMgt'));
    const availableWater = computed(() => getOptionsForLevel('water'));
    const availableLanduse = computed(() => getOptionsForLevel('landuse'));


    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');

      // Load Water chart data files
      await loadScript("./data/Water_overview_NRM_region_1_Landuse.js", 'Water_overview_NRM_region_1_Landuse');
      await loadScript("./data/Water_overview_NRM_region_2_Type.js", 'Water_overview_NRM_region_2_Type');
      await loadScript("./data/Water_ranking.js", 'Water_ranking');

      // Load Water Ag split data
      await loadScript("./data/Water_split_Ag_NRM_region_1_Landuse.js", 'Water_split_Ag_NRM_region_1_Landuse');
      await loadScript("./data/Water_split_Ag_NRM_region_2_Water_Supply.js", 'Water_split_Ag_NRM_region_2_Water_Supply');

      // Load Water Ag Mgt split data
      await loadScript("./data/Water_split_Am_NRM_region_1_Water_Supply.js", 'Water_split_Am_NRM_region_1_Water_Supply');
      await loadScript("./data/Water_split_Am_NRM_region_2_Landuse.js", 'Water_split_Am_NRM_region_2_Landuse');
      await loadScript("./data/Water_split_Am_NRM_region_3_Agri-Management.js", 'Water_split_Am_NRM_region_3_Agri-Management');

      // Load Water Non-Ag split data
      await loadScript("./data/Water_split_NonAg_NRM_region_1_Landuse.js", 'Water_split_NonAg_NRM_region_1_Landuse');

      // Load map data for all categories
      await loadScript(`${window.MapService.mapCategories['Water']['Ag']}`, 'map_water_yield_Ag');
      await loadScript(`${window.MapService.mapCategories['Water']['Ag Mgt']}`, 'map_water_yield_Am');
      await loadScript(`${window.MapService.mapCategories['Water']['Non-Ag']}`, 'map_water_yield_NonAg');

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['Water']);

      if (selectCategory.value === 'Ag') {
        mapPathName.value = 'window.map_water_yield_Ag';
        mapSelectKey.value = [selectWater.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_water_yield_Am';
        mapSelectKey.value = [selectAgMgt.value, selectWater.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_water_yield_NonAg';
        mapSelectKey.value = [selectLanduse.value, selectYear.value];
      }

      // Update chart with initial data
      updateChartSeries();

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        // Set dataLoaded to true after all data has been processed and the DOM has updated
        dataLoaded.value = true;
      });
    });

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch([selectCategory, selectAgMgt, selectWater, selectLanduse, selectYear], () => {
      // Reset values if they're no longer valid options
      if (selectCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = getOptionsForLevel('agMgt');
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectAgMgt.value)) {
          selectAgMgt.value = validAgMgtOptions[0];
        }
      }

      const validWaterOptions = getOptionsForLevel('water');
      if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectWater.value)) {
        selectWater.value = validWaterOptions[0];
      }

      const validLanduseOptions = getOptionsForLevel('landuse');
      if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectLanduse.value)) {
        selectLanduse.value = validLanduseOptions[0];
      }

      // Set map configuration based on category
      if (selectCategory.value === 'Ag') {
        mapPathName.value = 'window.map_water_yield_Ag';
        mapSelectKey.value = [selectWater.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_water_yield_Am';
        mapSelectKey.value = [selectAgMgt.value, selectWater.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_water_yield_NonAg';
        mapSelectKey.value = [selectLanduse.value, selectYear.value];
      }

      // Force a redraw by creating a new array reference
      mapSelectKey.value = [...mapSelectKey.value];
    });

    // Refresh chart when category/level/region change
    watch([selectCategory, selectChartLevel, selectRegion], () => {
      updateChartSeries();
    });



    // Functions to get chart data and options
    const getChartData = () => {
      // Map the visible "Chart level" label to the loaded dataset key
      const chartKeyMap = {
        'Ag': {
          'Landuse': 'Water_split_Ag_NRM_region_1_Landuse',
          'Water Supply': 'Water_split_Ag_NRM_region_2_Water_Supply',
          'Overview': 'Water_overview_NRM_region_1_Landuse',
          'Type': 'Water_overview_NRM_region_2_Type',
          'Ranking': 'Water_ranking',
        },
        'Ag Mgt': {
          'Water Supply': 'Water_split_Am_NRM_region_1_Water_Supply',
          'Landuse': 'Water_split_Am_NRM_region_2_Landuse',
          'Agri-Management': 'Water_split_Am_NRM_region_3_Agri-Management',
          'Overview': 'Water_overview_NRM_region_1_Landuse',
          'Type': 'Water_overview_NRM_region_2_Type',
          'Ranking': 'Water_ranking',
        },
        'Non-Ag': {
          'Landuse': 'Water_split_NonAg_NRM_region_1_Landuse',
          'Overview': 'Water_overview_NRM_region_1_Landuse',
          'Type': 'Water_overview_NRM_region_2_Type',
          'Ranking': 'Water_ranking',
        },
      };
      return chartKeyMap[selectCategory.value]?.[selectChartLevel.value] || 'Water_overview_NRM_region_1_Landuse';
    };

    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        // Ag options only available in 'Ag' category
        if (selectCategory.value !== 'Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Water']['Ag']);
      }
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectCategory.value !== 'Ag Mgt' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Water']['Ag Mgt']);
      }
      if (level === 'nonag') {
        // Non-Ag options only available in 'Non-Ag' category
        if (selectCategory.value !== 'Non-Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Water']['Non-Ag']);
      }
      return [];
    };

    const availableChartAg = computed(() => getChartOptionsForLevel('ag'));
    const availableChartAgMgt = computed(() => getChartOptionsForLevel('agMgt'));
    const availableChartNonAg = computed(() => getChartOptionsForLevel('nonag'));

    const updateChartSeries = () => {
      const dsKey = getChartData();
      selectDataset.value = {
        ...window.Chart_default_options,
        chart: { height: 500 },
        yAxis: {
          title: {
            text: "Megalitres",
          },
        },
        series: window[dsKey][selectRegion.value],
      };
    };

    return {
      yearIndex,
      isDrawerOpen,
      toggleDrawer,
      dataLoaded,

      availableYears,
      availableCategories,
      availableAgMgt,
      availableWater,
      availableLanduse,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,
      selectChartLevel,

      selectRegion,
      selectDataset,

      selectCategory,
      selectAgMgt,
      selectWater,
      selectLanduse,
      selectYear,

      mapSelectKey,
      mapPathName,
    };
  },
  template: `
    <div class="relative w-full h-screen">

      <!-- Drawer toggle button - Controls visibility of the chart panel -->
      <button 
        @click="toggleDrawer"
        class="absolute top-5 z-[1001] p-2.5 bg-white border border-gray-300 rounded cursor-pointer transition-all duration-300 ease-in-out"
        :class="isDrawerOpen ? 'right-[420px]' : 'right-5'">
        {{ isDrawerOpen ? '→' : '←' }}
      </button>

      <!-- Region selection dropdown - Uses FilterableDropdown component -->
      <div class="absolute w-[262px] top-32 left-[20px] z-50 bg-white/70 rounded-lg shadow-lg max-w-xs z-[9999]">
        <filterable-dropdown></filterable-dropdown>
      </div>

      <!-- Year slider - Allows selection of different years in the dataset -->
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


      <!-- Data selection controls container - Categories, AgMgt, Water, Landuse selections -->
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

        <!-- Ag Mgt options (only for Ag Mgt category when drawer is closed) -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <div v-if="!isDrawerOpen && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableAgMgt" :key="key"
              @click="selectAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectAgMgt === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="isDrawerOpen && availableChartAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAgMgt" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Water options -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && !isDrawerOpen && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableWater" :key="key"
              @click="selectWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectWater === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="dataLoaded && isDrawerOpen && availableChartAg.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Landuse options -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && !isDrawerOpen && availableLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableLanduse" :key="key"
              @click="selectLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectLanduse === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="dataLoaded && isDrawerOpen && availableChartNonAg.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartNonAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>
      </div>

      
      <!-- Map container with slide-out chart drawer - Main visualization area -->
      <div style="position: relative; width: 100%; height: 100%; overflow: hidden;">
        <!-- Map component takes full space -->
        <regions-map 
          :mapPathName="mapPathName"
          :mapKey="mapSelectKey"
          style="width: 100%; height: 100%;">
        </regions-map>
        
        <!-- Chart drawer positioned relative to map -->
        <div 
          :style="{
            position: 'absolute',
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
            :chartData="selectDataset" 
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

      

    </div>
  `
};