window.EconomicsView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    const selectRegion = inject('globalSelectedRegion');
    const isDrawerOpen = ref(false);
    const yearIndex = ref(0);

    // Data selection and visualization state
    const selectDataset = ref({});
    const selectChartLevel = ref('Overview');
    const mapPathName = ref({});
    const mapSelectKey = ref([]);

    // Category selection state
    const selectCategory = ref('Ag');
    const selectEconomicsType = ref('Cost'); // New: Cost or Revenue selection
    const selectAgMgt = ref('Environmental Plantings');
    const selectWater = ref('dry');
    const selectLanduse = ref('Beef - modified land');
    const selectCostRevenueType = ref('Area cost'); // New: For Ag category, this holds the cost/revenue type
    const selectYear = ref(2020);

    const availableYears = ref([]);
    const availableCategories = ref([]);
    const availableEconomicsTypes = ref(['Cost', 'Revenue']); // New: Available economics types
    const availableCostRevenueTypes = ref([]); // New: Available cost/revenue types

    const dataLoaded = ref(false);

    // Centralized function to navigate nested data structure based on current selections
    const getNestedData = (path = []) => {
      // Start with the appropriate data object based on category and economics type
      let dataSource;
      const economicsTypeKey = selectEconomicsType.value.toLowerCase();

      if (selectCategory.value === 'Ag') {
        dataSource = economicsTypeKey === 'cost' ? window.map_cost_Ag : window.map_revenue_Ag;
      } else if (selectCategory.value === 'Ag Mgt') {
        dataSource = economicsTypeKey === 'cost' ? window.map_cost_Am : window.map_revenue_Am;
      } else if (selectCategory.value === 'Non-Ag') {
        dataSource = economicsTypeKey === 'cost' ? window.map_cost_NonAg : window.map_revenue_NonAg;
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
      const economicsTypeKey = selectEconomicsType.value.toLowerCase();
      const mapDataKey = `map_${economicsTypeKey}_`;

      if (level === 'costRevenueType') {
        // Cost/Revenue Type options only available in 'Ag' category
        if (selectCategory.value !== 'Ag') return [];
        const waterData = getNestedData([selectWater.value]);
        return waterData ? Object.keys(waterData) : [];
      }

      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectCategory.value !== 'Ag Mgt') return [];
        const mapDataSource = economicsTypeKey === 'cost' ? window.map_cost_Am : window.map_revenue_Am;
        return mapDataSource ? Object.keys(mapDataSource) : [];
      }

      if (level === 'water') {
        // Water options depend on category and possibly ag mgt selection
        if (selectCategory.value === 'Ag') {
          const mapDataSource = economicsTypeKey === 'cost' ? window.map_cost_Ag : window.map_revenue_Ag;
          return mapDataSource ? Object.keys(mapDataSource) : [];
        } else if (selectCategory.value === 'Ag Mgt') {
          const agMgtData = getNestedData([selectAgMgt.value]);
          return agMgtData ? Object.keys(agMgtData) : [];
        }
        return [];
      }

      if (level === 'landuse') {
        // Landuse options depend on category and previous selections
        if (selectCategory.value === 'Ag') {
          // For Ag, we need to navigate water -> cost/revenue type -> landuse
          const costRevenueTypeData = getNestedData([selectWater.value, selectCostRevenueType.value]);
          return costRevenueTypeData ? Object.keys(costRevenueTypeData) : [];
        } else if (selectCategory.value === 'Ag Mgt') {
          const waterData = getNestedData([selectAgMgt.value, selectWater.value]);
          return waterData ? Object.keys(waterData) : [];
        } else if (selectCategory.value === 'Non-Ag') {
          const mapDataSource = economicsTypeKey === 'cost' ? window.map_cost_NonAg : window.map_revenue_NonAg;
          return mapDataSource ? Object.keys(mapDataSource) : [];
        }
        return [];
      }

      return [];
    };

    // Computed properties using the centralized functions
    const availableAgMgt = computed(() => getOptionsForLevel('agMgt'));
    const availableWater = computed(() => getOptionsForLevel('water'));
    const availableLanduse = computed(() => getOptionsForLevel('landuse'));
    const availableCostRevenueType = computed(() => getOptionsForLevel('costRevenueType'));

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');

      // Load Economics chart data files
      await loadScript("./data/Economics_overview.js", 'Economics_overview');
      await loadScript("./data/Economics_ranking.js", 'Economics_ranking');

      // Load Economics Ag split data
      await loadScript("./data/Economics_split_Ag_1_Land-use.js", 'Economics_split_Ag_1_Land-use');
      await loadScript("./data/Economics_split_Ag_2_Type.js", 'Economics_split_Ag_2_Type');
      await loadScript("./data/Economics_split_Ag_3_Water_supply.js", 'Economics_split_Ag_3_Water_supply');

      // Load Economics Ag Mgt split data
      await loadScript("./data/Economics_split_AM_1_Management_Type.js", 'Economics_split_AM_1_Management_Type');
      await loadScript("./data/Economics_split_AM_2_Water_supply.js", 'Economics_split_AM_2_Water_supply');
      await loadScript("./data/Economics_split_AM_3_Land-use.js", 'Economics_split_AM_3_Land-use');

      // Load Economics Non-Ag split data
      await loadScript("./data/Economics_split_NonAg_1_Land-use.js", 'Economics_split_NonAg_1_Land-use');

      // Load both cost and revenue data for all categories
      await loadScript(`${window.MapService.mapCategories['Economics']['Cost']['Ag']}`, 'map_cost_Ag');
      await loadScript(`${window.MapService.mapCategories['Economics']['Cost']['Ag Mgt']}`, 'map_cost_Am');
      await loadScript(`${window.MapService.mapCategories['Economics']['Cost']['Non-Ag']}`, 'map_cost_NonAg');
      await loadScript(`${window.MapService.mapCategories['Economics']['Revenue']['Ag']}`, 'map_revenue_Ag');
      await loadScript(`${window.MapService.mapCategories['Economics']['Revenue']['Ag Mgt']}`, 'map_revenue_Am');
      await loadScript(`${window.MapService.mapCategories['Economics']['Revenue']['Non-Ag']}`, 'map_revenue_NonAg');

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['Economics']['Cost']);

      // Set map configuration based on category
      updateMapConfiguration();

      // Update chart with initial data
      updateChartSeries();

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        // Set dataLoaded to true after all data has been processed and the DOM has updated
        dataLoaded.value = true;
      });
    });

    const updateMapConfiguration = () => {
      const economicsTypeKey = selectEconomicsType.value.toLowerCase();

      if (selectCategory.value === 'Ag') {
        mapPathName.value = `window.map_${economicsTypeKey}_Ag`;
        mapSelectKey.value = [selectWater.value, selectCostRevenueType.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Ag Mgt') {
        mapPathName.value = `window.map_${economicsTypeKey}_Am`;
        mapSelectKey.value = [selectAgMgt.value, selectWater.value, selectLanduse.value, selectYear.value];
      } else if (selectCategory.value === 'Non-Ag') {
        mapPathName.value = `window.map_${economicsTypeKey}_NonAg`;
        mapSelectKey.value = [selectLanduse.value, selectYear.value];
      }

      // Force a redraw by creating a new array reference
      mapSelectKey.value = [...mapSelectKey.value];
    };

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch([selectCategory, selectEconomicsType, selectAgMgt, selectWater, selectCostRevenueType, selectLanduse, selectYear], () => {
      // Reset values if they're no longer valid options
      if (selectCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = getOptionsForLevel('agMgt');
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectAgMgt.value)) {
          selectAgMgt.value = validAgMgtOptions[0];
        }
      }

      if (selectCategory.value === 'Ag') {
        const validCostRevenueTypeOptions = getOptionsForLevel('costRevenueType');
        if (validCostRevenueTypeOptions.length > 0 && !validCostRevenueTypeOptions.includes(selectCostRevenueType.value)) {
          selectCostRevenueType.value = validCostRevenueTypeOptions[0];
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

      // Update map configuration
      updateMapConfiguration();
    });

    // Refresh chart when category/level/region change
    watch([selectCategory, selectChartLevel, selectRegion, selectEconomicsType], () => {
      updateChartSeries();
    });

    // Functions to get chart data and options
    const getChartData = () => {
      // Map the visible "Chart level" label to the loaded dataset key
      const chartKeyMap = {
        'Ag': {
          'Overview': 'Economics_overview',
          'Ranking': 'Economics_ranking',
          'Land-use': 'Economics_split_Ag_1_Land-use',
          'Type': 'Economics_split_Ag_2_Type',
          'Water supply': 'Economics_split_Ag_3_Water_supply',
        },
        'Ag Mgt': {
          'Overview': 'Economics_overview',
          'Ranking': 'Economics_ranking',
          'Management Type': 'Economics_split_AM_1_Management_Type',
          'Water supply': 'Economics_split_AM_2_Water_supply',
          'Land-use': 'Economics_split_AM_3_Land-use',
        },
        'Non-Ag': {
          'Overview': 'Economics_overview',
          'Ranking': 'Economics_ranking',
          'Land-use': 'Economics_split_NonAg_1_Land-use',
        },
      };
      return chartKeyMap[selectCategory.value]?.[selectChartLevel.value] || 'Economics_overview';
    };

    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        // Ag options only available in 'Ag' category
        if (selectCategory.value !== 'Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Economics']['Ag']);
      }
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectCategory.value !== 'Ag Mgt' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Economics']['Ag Mgt']);
      }
      if (level === 'nonag') {
        // Non-Ag options only available in 'Non-Ag' category
        if (selectCategory.value !== 'Non-Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Economics']['Non-Ag']);
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
            text: "AUD",
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
      availableEconomicsTypes,
      availableAgMgt,
      availableWater,
      availableLanduse,
      availableCostRevenueType,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,
      selectChartLevel,

      selectRegion,
      selectDataset,

      selectCategory,
      selectEconomicsType,
      selectAgMgt,
      selectWater,
      selectCostRevenueType,
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


      <!-- Data selection controls container - Categories, Economics Type, AgMgt, Water, Landuse selections -->
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

        <!-- Economics Type buttons (only visible when drawer is closed) -->
        <div class="flex items-center border-t border-white/10 pt-1" v-if="!isDrawerOpen">
          <div class="flex space-x-1">
            <span class="text-[0.8rem] mr-1 font-medium">Type:</span>
            <button v-for="(val, key) in availableEconomicsTypes" :key="key"
              @click="selectEconomicsType = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
              :class="{'bg-sky-500 text-white': selectEconomicsType === val}">
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

        <!-- Cost/Revenue Type options (only for Ag category when drawer is closed) -->
        <div 
          v-if="dataLoaded && !isDrawerOpen && selectCategory === 'Ag' && availableCostRevenueType.length > 0" 
          class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">{{ selectEconomicsType }} Type:</span>
            <button v-for="(val, key) in availableCostRevenueType" :key="key"
              @click="selectCostRevenueType = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectCostRevenueType === val}">
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