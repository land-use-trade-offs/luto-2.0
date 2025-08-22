window.GHGView = {
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
    const MapRegister = window.MapService.mapCategories['GHG'];     // MapService was registered in the index.html
    const dataConstructor = new window.DataConstructor();

    // Map selection state
    const selectMapCategory = ref('Ag');
    const selectMapGHGSource = ref('TCO2E_CHEM_APPL'); // GHG source (TCO2E_CHEM_APPL, TCO2E_CULTIV, etc.)
    const selectMapAgMgt = ref('Precision Agriculture');
    const selectMapWater = ref('dry');
    const selectMapLanduse = ref('Beef - modified land');
    const selectYear = ref(2020);

    const availableYears = ref([]);
    const availableCategories = ref([]);

    const dataLoaded = ref(false);

    // Function to load data for current category
    const loadDataForCategory = (category) => {
      if (!window[MapRegister[category]['name']]) return;
      dataConstructor.loadData(window[MapRegister[category]['name']]);
    };

    // Computed properties using DataConstructor
    const availableGHGSource = computed(() => {
      if (selectMapCategory.value !== 'Ag') return [];
      return dataConstructor.getAvailableKeysAtNextLevel({});
    });

    const availableAgMgt = computed(() => {
      if (selectMapCategory.value !== 'Ag Mgt') return [];
      return dataConstructor.getAvailableKeysAtNextLevel({});
    });

    const availableWater = computed(() => {
      // Non-Ag category has no water options
      if (selectMapCategory.value === 'Non-Ag') {
        return [];
      }

      const fixedLevels = {};

      if (selectMapCategory.value === 'Ag' && selectMapGHGSource.value && selectMapLanduse.value) {
        fixedLevels.level_1 = selectMapGHGSource.value;
        fixedLevels.level_2 = selectMapLanduse.value;
      } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value && selectMapLanduse.value) {
        fixedLevels.level_1 = selectMapAgMgt.value;
        fixedLevels.level_2 = selectMapLanduse.value;
      }

      return dataConstructor.getAvailableKeysAtNextLevel(fixedLevels);
    });

    const availableLanduse = computed(() => {
      const fixedLevels = {};

      if (selectMapCategory.value === 'Ag' && selectMapGHGSource.value) {
        fixedLevels.level_1 = selectMapGHGSource.value;
      } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value) {
        fixedLevels.level_1 = selectMapAgMgt.value;
      }

      return dataConstructor.getAvailableKeysAtNextLevel(fixedLevels);
    });
    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');

      // Load GHG chart data files
      await loadScript("./data/GHG_overview.js", 'GHG_overview');
      await loadScript("./data/GHG_ranking.js", 'GHG_ranking');

      // Load GHG Ag split data
      await loadScript("./data/GHG_split_Ag_1_GHG_Category.js", 'GHG_split_Ag_1_GHG_Category');
      await loadScript("./data/GHG_split_Ag_2_Land-use.js", 'GHG_split_Ag_2_Land-use');
      await loadScript("./data/GHG_split_Ag_3_Land-use_type.js", 'GHG_split_Ag_3_Land-use_type');
      await loadScript("./data/GHG_split_Ag_4_Source.js", 'GHG_split_Ag_4_Source');
      await loadScript("./data/GHG_split_Ag_5_Water_supply.js", 'GHG_split_Ag_5_Water_supply');

      // Load GHG Ag Mgt split data
      await loadScript("./data/GHG_split_Am_1_Land-use.js", 'GHG_split_Am_1_Land-use');
      await loadScript("./data/GHG_split_Am_2_Land-use_type.js", 'GHG_split_Am_2_Land-use_type');
      await loadScript("./data/GHG_split_Am_3_Agricultural_Management_Type.js", 'GHG_split_Am_3_Agricultural_Management_Type');
      await loadScript("./data/GHG_split_Am_4_Water_supply.js", 'GHG_split_Am_4_Water_supply');

      // Load GHG Non-Ag split data
      await loadScript("./data/GHG_split_NonAg_1_Land-use.js", 'GHG_split_NonAg_1_Land-use');

      // Load map data for all categories
      await loadScript(MapRegister['Ag']['path'], MapRegister['Ag']['name']);
      await loadScript(MapRegister['Ag Mgt']['path'], MapRegister['Ag Mgt']['name']);
      await loadScript(MapRegister['Non-Ag']['path'], MapRegister['Non-Ag']['name']);

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['GHG']);

      // Load initial data for the default category
      loadDataForCategory(selectMapCategory.value);

      // Update chart and map after initial data load
      updateChartSeries();
      updateMapOverlay();

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        dataLoaded.value = true;
      });
    });

    // Watch for category changes to reload data
    watch(selectMapCategory, (newCategory) => {
      loadDataForCategory(newCategory);
    });

    // Watch for changes to validate selections using DataConstructor
    watch([selectMapCategory, selectMapGHGSource, selectMapAgMgt, selectMapWater, selectMapLanduse, selectYear], () => {
      // Reset values if they're no longer valid options using DataConstructor
      if (selectMapCategory.value === 'Ag') {
        const validGHGSourceOptions = dataConstructor.getAvailableKeysAtNextLevel({});
        if (validGHGSourceOptions.length > 0 && !validGHGSourceOptions.includes(selectMapGHGSource.value)) {
          selectMapGHGSource.value = validGHGSourceOptions[0];
        }
      }

      if (selectMapCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = dataConstructor.getAvailableKeysAtNextLevel({});
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectMapAgMgt.value)) {
          selectMapAgMgt.value = validAgMgtOptions[0];
        }
      }

      // Validate landuse options
      const landuseFixedLevels = {};
      if (selectMapCategory.value === 'Ag' && selectMapGHGSource.value) {
        landuseFixedLevels.level_1 = selectMapGHGSource.value;
      } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value) {
        landuseFixedLevels.level_1 = selectMapAgMgt.value;
      }
      const validLanduseOptions = dataConstructor.getAvailableKeysAtNextLevel(landuseFixedLevels);
      if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
        selectMapLanduse.value = validLanduseOptions[0];
      }

      // Validate water options (only for Ag and Ag Mgt categories)
      if (selectMapCategory.value !== 'Non-Ag') {
        const waterFixedLevels = {};
        if (selectMapCategory.value === 'Ag' && selectMapGHGSource.value && selectMapLanduse.value) {
          waterFixedLevels.level_1 = selectMapGHGSource.value;
          waterFixedLevels.level_2 = selectMapLanduse.value;
        } else if (selectMapCategory.value === 'Ag Mgt' && selectMapAgMgt.value && selectMapLanduse.value) {
          waterFixedLevels.level_1 = selectMapAgMgt.value;
          waterFixedLevels.level_2 = selectMapLanduse.value;
        }
        const validWaterOptions = dataConstructor.getAvailableKeysAtNextLevel(waterFixedLevels);
        if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectMapWater.value)) {
          selectMapWater.value = validWaterOptions[0];
        }
      }

      // Set map configuration based on category
      updateMapOverlay();
    });

    // Refresh chart when category/level/region change
    watch([selectMapCategory, selectChartLevel, selectRegion], () => {
      updateChartSeries();
    });

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };



    // Functions to get chart data and options
    const getChartData = () => {
      // Map the visible "Chart level" label to the loaded dataset key
      const chartKeyMap = {
        'Ag': {
          'Overview': 'GHG_overview',
          'Ranking': 'GHG_ranking',
          'GHG Category': 'GHG_split_Ag_1_GHG_Category',
          'Land-use': 'GHG_split_Ag_2_Land-use',
          'Land-use type': 'GHG_split_Ag_3_Land-use_type',
          'Source': 'GHG_split_Ag_4_Source',
          'Water supply': 'GHG_split_Ag_5_Water_supply',
        },
        'Ag Mgt': {
          'Overview': 'GHG_overview',
          'Ranking': 'GHG_ranking',
          'Land-use': 'GHG_split_Am_1_Land-use',
          'Land-use type': 'GHG_split_Am_2_Land-use_type',
          'Agricultural Management Type': 'GHG_split_Am_3_Agricultural_Management_Type',
          'Water supply': 'GHG_split_Am_4_Water_supply',
        },
        'Non-Ag': {
          'Overview': 'GHG_overview',
          'Ranking': 'GHG_ranking',
          'Land-use': 'GHG_split_NonAg_1_Land-use',
        },
      };
      return chartKeyMap[selectMapCategory.value]?.[selectChartLevel.value] || 'GHG_overview';
    };

    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        // Ag options only available in 'Ag' category
        if (selectMapCategory.value !== 'Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['GHG']['Ag']);
      }
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectMapCategory.value !== 'Ag Mgt' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['GHG']['Ag Mgt']);
      }
      if (level === 'nonag') {
        // Non-Ag options only available in 'Non-Ag' category
        if (selectMapCategory.value !== 'Non-Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['GHG']['Non-Ag']);
      }
      return [];
    };

    const availableChartAg = computed(() => getChartOptionsForLevel('ag'));
    const availableChartAgMgt = computed(() => getChartOptionsForLevel('agMgt'));
    const availableChartNonAg = computed(() => getChartOptionsForLevel('nonag'));

    const updateChartSeries = () => {
      try {
        const dsKey = getChartData();

        // Check if the required data exists
        if (!window[dsKey] || !window[dsKey][selectRegion.value]) {
          console.warn(`Chart data not found for key: ${dsKey}, region: ${selectRegion.value}`);
          return;
        }

        selectDataset.value = {
          ...window.Chart_default_options,
          chart: { height: 500 },
          yAxis: {
            title: {
              text: "tCO2e",
            },
          },
          series: window[dsKey][selectRegion.value],
        };
      } catch (error) {
        console.error('Error updating chart series:', error);
      }
    };

    const updateMapOverlay = () => {
      // Set map configuration based on category
      if (selectMapCategory.value === 'Ag') {
        mapVarName.value = MapRegister["Ag"]["name"];
        mapVarPath.value = [selectMapGHGSource.value, selectMapLanduse.value, selectMapWater.value, selectYear.value];
      } else if (selectMapCategory.value === 'Ag Mgt') {
        mapVarName.value = MapRegister["Ag Mgt"]["name"];
        mapVarPath.value = [selectMapAgMgt.value, selectMapLanduse.value, selectMapWater.value, selectYear.value];
      } else if (selectMapCategory.value === 'Non-Ag') {
        mapVarName.value = MapRegister["Non-Ag"]["name"];
        mapVarPath.value = [selectMapLanduse.value, selectYear.value];
      }
    };

    return {
      yearIndex,
      isDrawerOpen,
      toggleDrawer,
      dataLoaded,

      availableYears,
      availableCategories,
      availableGHGSource,
      availableAgMgt,
      availableWater,
      availableLanduse,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,
      selectChartLevel,

      selectRegion,
      selectDataset,

      selectMapCategory,
      selectMapGHGSource,
      selectMapAgMgt,
      selectMapWater,
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


      <!-- Data selection controls container - Categories, AgMgt, Water, Landuse selections -->
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

        <!-- GHG Source options (only for Ag category when drawer is closed) -->
        <div 
          v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag' && availableGHGSource.length > 0" 
          class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">GHG Source:</span>
            <button v-for="(val, key) in availableGHGSource" :key="key"
              @click="selectMapGHGSource = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapGHGSource === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category when drawer is closed) -->
        <div v-if="dataLoaded && !isDrawerOpen && selectMapCategory === 'Ag Mgt' && availableAgMgt.length > 0" 
             class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableAgMgt" :key="key"
              @click="selectMapAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapAgMgt === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Water options (for map categories when drawer is closed) -->
        <div v-if="dataLoaded && !isDrawerOpen && availableWater.length > 0" 
             class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableWater" :key="key"
              @click="selectMapWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapWater === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Landuse options (for map categories when drawer is closed) -->
        <div v-if="dataLoaded && !isDrawerOpen && availableLanduse.length > 0" 
             class="flex items-start border-t border-white/10 pt-1">
          <div class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Chart level options (when drawer is open) -->
        <div v-if="dataLoaded && isDrawerOpen" class="flex items-start border-t border-white/10 pt-1">
          <div v-if="selectMapCategory === 'Ag' && availableChartAg.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="selectMapCategory === 'Ag Mgt' && availableChartAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-for="(val, key) in availableChartAgMgt" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="selectMapCategory === 'Non-Ag' && availableChartNonAg.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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