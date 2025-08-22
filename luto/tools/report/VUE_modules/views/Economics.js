window.EconomicsView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    const selectRegion = inject('globalSelectedRegion');
    const isDrawerOpen = ref(false);
    const yearIndex = ref(0);

    // Data selection and visualization state
    const selectDataset = ref({});
    const selectChartLevel = ref('Overview');

    // Base map selection state
    const mapVarName = ref('');
    const mapVarPath = ref([]);

    // Data|Map service
    const MapRegister = window.MapService.mapCategories['Economics'];     // MapService was registered in the index.html
    const dataConstructor = new window.DataConstructor();

    // Map selection state
    const selectMapCategory = ref('Ag');
    const selectMapEconomicsType = ref('Cost'); // New: Cost or Revenue selection
    const selectMapAgMgt = ref('Asparagopsis taxiformis');
    const selectMapWater = ref('dry');
    const selectMapLanduse = ref('Beef - modified land');
    const selectMapCostRevenueType = ref('Water cost'); // Default for cost, will be updated based on selection
    const selectYear = ref(2020);

    const availableYears = ref([]);
    const availableCategories = ref([]);
    const availableEconomicsTypes = ref(['Cost', 'Revenue']); // New: Available economics types

    const dataLoaded = ref(false);

    // Function to load data for current category and economics type
    const loadDataForCategory = (category, economicsType) => {
      const mapData = window[MapRegister[economicsType][category]['name']];
      if (!mapData) return;
      dataConstructor.loadData(mapData);
    };

    // Computed properties using DataConstructor
    const availableMapAgMgt = computed(() => {
      if (selectMapCategory.value !== 'Ag Mgt') return [];
      return dataConstructor.getAvailableKeysAtNextLevel({});
    });

    const availableMapWater = computed(() => {
      // Non-Ag category has no water options
      if (selectMapCategory.value === 'Non-Ag') {
        return [];
      }

      const fixedLevels = {};

      if (selectMapCategory.value === 'Ag') {
        const economicsTypeKey = selectMapEconomicsType.value.toLowerCase();
        if (economicsTypeKey === 'cost') {
          // For cost: Water cost > dry/irr
          if (selectMapCostRevenueType.value) {
            fixedLevels.level_1 = selectMapCostRevenueType.value;
          }
        } else {
          // For revenue: Wool > landuse > dry/irr
          if (selectMapCostRevenueType.value) {
            fixedLevels.level_1 = selectMapCostRevenueType.value;
            if (selectMapLanduse.value) {
              fixedLevels.level_2 = selectMapLanduse.value;
            }
          }
        }
      } else if (selectMapCategory.value === 'Ag Mgt') {
        // Both cost and revenue: AgMgt > landuse > water
        if (selectMapAgMgt.value) {
          fixedLevels.level_1 = selectMapAgMgt.value;
          if (selectMapLanduse.value) {
            fixedLevels.level_2 = selectMapLanduse.value;
          }
        }
      }

      return dataConstructor.getAvailableKeysAtNextLevel(fixedLevels);
    });

    const availableMapLanduse = computed(() => {
      const fixedLevels = {};

      if (selectMapCategory.value === 'Ag') {
        const economicsTypeKey = selectMapEconomicsType.value.toLowerCase();
        if (economicsTypeKey === 'cost') {
          // For cost: Water cost > dry/irr > landuse
          if (selectMapCostRevenueType.value) {
            fixedLevels.level_1 = selectMapCostRevenueType.value;
            if (selectMapWater.value) {
              fixedLevels.level_2 = selectMapWater.value;
            }
          }
        } else {
          // For revenue: Wool > landuse
          if (selectMapCostRevenueType.value) {
            fixedLevels.level_1 = selectMapCostRevenueType.value;
          }
        }
      } else if (selectMapCategory.value === 'Ag Mgt') {
        // Both cost and revenue: AgMgt > landuse
        if (selectMapAgMgt.value) {
          fixedLevels.level_1 = selectMapAgMgt.value;
        }
      }

      return dataConstructor.getAvailableKeysAtNextLevel(fixedLevels);
    });

    // Cost/Revenue Type options (different for each economics type and category)
    const availableCostRevenueType = computed(() => {
      if (selectMapCategory.value !== 'Ag') return [];
      return dataConstructor.getAvailableKeysAtNextLevel({});
    });

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

      // Load map data for all categories using MapService structure
      await loadScript(MapRegister['Cost']['Ag']['path'], MapRegister['Cost']['Ag']['name']);
      await loadScript(MapRegister['Cost']['Ag Mgt']['path'], MapRegister['Cost']['Ag Mgt']['name']);
      await loadScript(MapRegister['Cost']['Non-Ag']['path'], MapRegister['Cost']['Non-Ag']['name']);
      await loadScript(MapRegister['Revenue']['Ag']['path'], MapRegister['Revenue']['Ag']['name']);
      await loadScript(MapRegister['Revenue']['Ag Mgt']['path'], MapRegister['Revenue']['Ag Mgt']['name']);
      await loadScript(MapRegister['Revenue']['Non-Ag']['path'], MapRegister['Revenue']['Non-Ag']['name']);

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(MapRegister['Cost']);

      // Load initial data for the default category and economics type
      loadDataForCategory(selectMapCategory.value, selectMapEconomicsType.value);

      // Set map configuration based on category
      updateMapOverlay();

      // Update chart with initial data
      updateChartSeries();

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        // Set dataLoaded to true after all data has been processed and the DOM has updated
        dataLoaded.value = true;
      });
    });

    // Watch for category and economics type changes to reload data
    watch([selectMapCategory, selectMapEconomicsType], ([newCategory, newEconomicsType]) => {
      loadDataForCategory(newCategory, newEconomicsType);
    });

    // Watch for changes to validate selections using DataConstructor
    watch([selectMapCategory, selectMapEconomicsType, selectMapAgMgt, selectMapCostRevenueType, selectMapWater, selectMapLanduse, selectYear], () => {
      // Reset values if they're no longer valid options using DataConstructor
      if (selectMapCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = dataConstructor.getAvailableKeysAtNextLevel({});
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectMapAgMgt.value)) {
          selectMapAgMgt.value = validAgMgtOptions[0];
        }
      }

      if (selectMapCategory.value === 'Ag') {
        const validCostRevenueTypeOptions = dataConstructor.getAvailableKeysAtNextLevel({});
        if (validCostRevenueTypeOptions.length > 0 && !validCostRevenueTypeOptions.includes(selectMapCostRevenueType.value)) {
          selectMapCostRevenueType.value = validCostRevenueTypeOptions[0];
        }
      }

      // Validate water options
      if (selectMapCategory.value !== 'Non-Ag') {
        const waterFixedLevels = {};
        if (selectMapCategory.value === 'Ag') {
          const economicsTypeKey = selectMapEconomicsType.value.toLowerCase();
          if (economicsTypeKey === 'cost' && selectMapCostRevenueType.value) {
            waterFixedLevels.level_1 = selectMapCostRevenueType.value;
          } else if (economicsTypeKey !== 'cost' && selectMapCostRevenueType.value && selectMapLanduse.value) {
            waterFixedLevels.level_1 = selectMapCostRevenueType.value;
            waterFixedLevels.level_2 = selectMapLanduse.value;
          }
        } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value && selectMapLanduse.value) {
          waterFixedLevels.level_1 = selectMapAgMgt.value;
          waterFixedLevels.level_2 = selectMapLanduse.value;
        }
        const validWaterOptions = dataConstructor.getAvailableKeysAtNextLevel(waterFixedLevels);
        if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectMapWater.value)) {
          selectMapWater.value = validWaterOptions[0];
        }
      }

      // Validate landuse options
      const landuseFixedLevels = {};
      if (selectMapCategory.value === 'Ag') {
        const economicsTypeKey = selectMapEconomicsType.value.toLowerCase();
        if (economicsTypeKey === 'cost' && selectMapCostRevenueType.value && selectMapWater.value) {
          landuseFixedLevels.level_1 = selectMapCostRevenueType.value;
          landuseFixedLevels.level_2 = selectMapWater.value;
        } else if (economicsTypeKey !== 'cost' && selectMapCostRevenueType.value) {
          landuseFixedLevels.level_1 = selectMapCostRevenueType.value;
        }
      } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value) {
        landuseFixedLevels.level_1 = selectMapAgMgt.value;
      }
      const validLanduseOptions = dataConstructor.getAvailableKeysAtNextLevel(landuseFixedLevels);
      if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
        selectMapLanduse.value = validLanduseOptions[0];
      }

      // Set map configuration based on category
      updateMapOverlay();
    });

    // Refresh chart when category/level/region change
    watch([selectMapCategory, selectChartLevel, selectRegion], () => {
      updateChartSeries();
    });

    const updateMapOverlay = () => {
      const economicsTypeKey = selectMapEconomicsType.value.toLowerCase();

      if (selectMapCategory.value === 'Ag') {
        mapVarName.value = MapRegister[selectMapEconomicsType.value]["Ag"]["name"];

        if (economicsTypeKey === 'cost') {
          // Cost structure: Water cost > dry/irr > landuse > year
          mapVarPath.value = [selectMapCostRevenueType.value, selectMapWater.value, selectMapLanduse.value, selectYear.value];
        } else {
          // Revenue structure: Wool > landuse > dry/irr > year
          mapVarPath.value = [selectMapCostRevenueType.value, selectMapLanduse.value, selectMapWater.value, selectYear.value];
        }
      } else if (selectMapCategory.value === 'Ag Mgt') {
        mapVarName.value = MapRegister[selectMapEconomicsType.value]["Ag Mgt"]["name"];
        // Both cost and revenue: AgMgt > landuse > water > year
        mapVarPath.value = [selectMapAgMgt.value, selectMapLanduse.value, selectMapWater.value, selectYear.value];
      } else if (selectMapCategory.value === 'Non-Ag') {
        mapVarName.value = MapRegister[selectMapEconomicsType.value]["Non-Ag"]["name"];
        // Both cost and revenue: landuse > year
        mapVarPath.value = [selectMapLanduse.value, selectYear.value];
      }
    };

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

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
      return chartKeyMap[selectMapCategory.value]?.[selectChartLevel.value] || 'Economics_overview';
    };

    const getChartOptionsForLevel = (level) => {
      try {
        if (level === 'ag') {
          // Ag options only available in 'Ag' category
          if (selectMapCategory.value !== 'Ag' || !window.DataService) return [];
          return Object.keys(window.DataService.ChartPaths['Economics']['Ag']) || [];
        }
        if (level === 'agMgt') {
          // Ag Mgt options only available in 'Ag Mgt' category
          if (selectMapCategory.value !== 'Ag Mgt' || !window.DataService) return [];
          return Object.keys(window.DataService.ChartPaths['Economics']['Ag Mgt']) || [];
        }
        if (level === 'nonag') {
          // Non-Ag options only available in 'Non-Ag' category
          if (selectMapCategory.value !== 'Non-Ag' || !window.DataService) return [];
          return Object.keys(window.DataService.ChartPaths['Economics']['Non-Ag']) || [];
        }
        return [];
      } catch (e) {
        console.warn('Error in getChartOptionsForLevel:', e);
        return [];
      }
    };

    const availableChartAg = computed(() => getChartOptionsForLevel('ag'));
    const availableChartAgMgt = computed(() => getChartOptionsForLevel('agMgt'));
    const availableChartNonAg = computed(() => getChartOptionsForLevel('nonag'));

    const updateChartSeries = () => {
      try {
        const dsKey = getChartData();
        if (!window[dsKey] || !window[dsKey][selectRegion.value]) {
          console.warn('Chart data not available for:', dsKey, selectRegion.value);
          return;
        }
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
      } catch (e) {
        console.warn('Error in updateChartSeries:', e);
      }
    };

    return {
      yearIndex,
      isDrawerOpen,
      toggleDrawer,
      dataLoaded,

      availableYears,
      availableCategories,
      availableEconomicsTypes,
      availableMapAgMgt,
      availableMapWater,
      availableMapLanduse,
      availableCostRevenueType,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,
      selectChartLevel,

      selectRegion,
      selectDataset,

      selectMapCategory,
      selectMapEconomicsType,
      selectMapAgMgt,
      selectMapWater,
      selectMapCostRevenueType,
      selectMapLanduse,
      selectYear,

      mapVarPath,
      mapVarName,
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
              @click="selectMapCategory = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
              :class="{'bg-sky-500 text-white': selectMapCategory === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Economics Type buttons (only visible when drawer is closed) -->
        <div class="flex items-center border-t border-white/10 pt-1" v-if="!isDrawerOpen">
          <div class="flex space-x-1">
            <span class="text-[0.8rem] mr-1 font-medium">Type:</span>
            <button v-for="(val, key) in availableEconomicsTypes" :key="key"
              @click="selectMapEconomicsType = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
              :class="{'bg-sky-500 text-white': selectMapEconomicsType === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Cost/Revenue Type options (only for Ag category - first level) -->
        <div 
          v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && availableCostRevenueType && availableCostRevenueType.length > 0" 
          class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">{{ selectMapEconomicsType }} Type:</span>
            <button v-for="(val, key) in availableCostRevenueType" :key="key"
              @click="selectMapCostRevenueType = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapCostRevenueType === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category - first level) -->
        <div v-if="!isDrawerOpen && selectMapCategory === 'Ag Mgt' && availableMapAgMgt && availableMapAgMgt.length > 0" class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableMapAgMgt" :key="key"
              @click="selectMapAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapAgMgt === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Landuse options - different positions based on category and economics type -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <!-- For Revenue Ag: Landuse comes after Cost/Revenue Type -->
          <div v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && selectMapEconomicsType === 'Revenue' && availableMapLanduse && availableMapLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableMapLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
              {{ val }}
            </button>
          </div>
          <!-- For Ag Mgt: Landuse comes after Ag Mgt selection -->
          <div v-else-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag Mgt' && availableMapLanduse && availableMapLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableMapLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
              {{ val }}
            </button>
          </div>
          <!-- For Non-Ag: Landuse is the first option -->
          <div v-else-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Non-Ag' && availableMapLanduse && availableMapLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableMapLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
              {{ val }}
            </button>
          </div>
          <!-- Chart options for when drawer is open -->
          <div v-else-if="isDrawerOpen && availableChartAgMgt && availableChartAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAgMgt" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Water options - comes at different positions based on structure -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <!-- For Cost Ag: Water comes after Cost Type -->
          <div v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && selectMapEconomicsType === 'Cost' && availableMapWater && availableMapWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableMapWater" :key="key"
              @click="selectMapWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapWater === val}">
              {{ val }}
            </button>
          </div>
          <!-- For Revenue Ag: Water comes after Landuse -->
          <div v-else-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && selectMapEconomicsType === 'Revenue' && availableMapWater && availableMapWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableMapWater" :key="key"
              @click="selectMapWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapWater === val}">
              {{ val }}
            </button>
          </div>
          <!-- For Ag Mgt: Water comes after Landuse -->
          <div v-else-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag Mgt' && availableMapWater && availableMapWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableMapWater" :key="key"
              @click="selectMapWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapWater === val}">
              {{ val }}
            </button>
          </div>
          <!-- Chart options for when drawer is open -->
          <div v-else-if="dataLoaded && isDrawerOpen && availableChartAg && availableChartAg.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Landuse options for Cost Ag - comes last in the hierarchy -->
        <div v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && selectMapEconomicsType === 'Cost' && availableMapLanduse && availableMapLanduse.length > 0" class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableMapLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Chart options for Non-Ag when drawer is open -->
        <div v-if="dataLoaded && isDrawerOpen && availableChartNonAg && availableChartNonAg.length > 0" class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
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
          :mapName="mapVarName"
          :mapPath="mapVarPath"
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