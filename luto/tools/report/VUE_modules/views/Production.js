window.ProductionView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    // Global selection state
    const selectYear = ref(2020);
    const selectRegion = inject('globalSelectedRegion');
    const yearIndex = ref(0);

    // Global variables
    const availableYears = ref([]);
    const availableCategories = ref(['Ag', 'Ag Mgt', 'Non-Ag']);
    const dataLoaded = ref(false);
    const isDrawerOpen = ref(false);

    // Data selection and visualization state
    const selectChartDataset = ref({});

    // Base map selection state
    const mapVarName = ref('');
    const mapVarPath = ref([]);

    // Data|Map service
    const MapRegister = window.MapService.mapCategories['Production'];     // MapService was registered in the index.html
    const dataConstructor = new window.DataConstructor();

    // Map selection state
    const selectMapCategory = ref('Ag');
    const selectMapAgMgt = ref('Asparagopsis taxiformis');
    const selectMapCommodity = ref('apples');

    // Chart selection state
    const selectChartLevel = ref('Agricultural');

    // Function to load data for current category
    const loadDataForCategory = (category) => {
      if (!window[MapRegister[category]['name']]) return;
      dataConstructor.loadData(window[MapRegister[category]['name']]);
    };

    // Computed properties using DataConstructor
    const availableMapCommodities = computed(() => {
      // Load and return commodities specific to the current category
      if (selectMapCategory.value === 'Ag') {
        if (window[MapRegister['Ag']['name']]) {
          dataConstructor.loadData(window[MapRegister['Ag']['name']]);
          return dataConstructor.getAvailableKeysAtNextLevel({});
        }
      } else if (selectMapCategory.value === 'Ag Mgt') {
        if (window[MapRegister['Ag Mgt']['name']]) {
          dataConstructor.loadData(window[MapRegister['Ag Mgt']['name']]);
          return dataConstructor.getAvailableKeysAtNextLevel({});
        }
      } else if (selectMapCategory.value === 'Non-Ag') {
        if (window[MapRegister['Non-Ag']['name']]) {
          dataConstructor.loadData(window[MapRegister['Non-Ag']['name']]);
          return dataConstructor.getAvailableKeysAtNextLevel({});
        }
      }
      return [];
    });

    const availableMapAgMgt = computed(() => {
      if (selectMapCategory.value !== 'Ag Mgt') return [];

      const fixedLevels = {};
      if (selectMapCommodity.value) {
        fixedLevels.level_1 = selectMapCommodity.value;
      }
      return dataConstructor.getAvailableKeysAtNextLevel(fixedLevels);
    });

    const availableChartAg = computed(() => getChartOptionsForLevel('ag'));
    const availableChartAgMgt = computed(() => getChartOptionsForLevel('agMgt'));
    const availableChartNonAg = computed(() => getChartOptionsForLevel('nonag'));



    // Centralized function to navigate nested data structure based on current selections
    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        if (selectMapCategory.value !== 'Ag') return [];
        return ['Agricultural', 'Overview', 'Commodity', 'Type'];
      }
      if (level === 'agMgt') {
        if (selectMapCategory.value !== 'Ag Mgt') return [];
        return ['Agricultural Management', 'Overview', 'Commodity', 'Type'];
      }
      if (level === 'nonag') {
        if (selectMapCategory.value !== 'Non-Ag') return [];
        return ['Non-Agricultural', 'Overview', 'Commodity', 'Type'];
      }
      return [];
    };

    const getChartData = () => {
      // Map the visible "Chart level" label to the loaded dataset key
      const chartKeyMap = {
        'Ag': {
          'Agricultural': 'Production_LUTO_1_Agricultural',
          'Overview': 'Production_achive_percent',
          'Commodity': 'Production_sum_1_Commodity',
          'Type': 'Production_sum_2_Type',
        },
        'Ag Mgt': {
          'Agricultural Management': 'Production_LUTO_3_Agricultural_Management',
          'Overview': 'Production_achive_percent',
          'Commodity': 'Production_sum_1_Commodity',
          'Type': 'Production_sum_2_Type',
        },
        'Non-Ag': {
          'Non-Agricultural': 'Production_LUTO_2_Non-Agricultural',
          'Overview': 'Production_achive_percent',
          'Commodity': 'Production_sum_1_Commodity',
          'Type': 'Production_sum_2_Type',
        },
      };
      return chartKeyMap[selectMapCategory.value]?.[selectChartLevel.value] || 'Production_LUTO_1_Agricultural';
    };


    onMounted(async () => {

      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');

      // Load map data for all categories
      await loadScript(MapRegister['Ag']['path'], MapRegister['Ag']['name']);
      await loadScript(MapRegister['Ag Mgt']['path'], MapRegister['Ag Mgt']['name']);
      await loadScript(MapRegister['Non-Ag']['path'], MapRegister['Non-Ag']['name']);

      // Chart data
      await loadScript("./data/Production_LUTO_1_Agricultural.js", 'Production_LUTO_1_Agricultural');
      await loadScript("./data/Production_LUTO_2_Non-Agricultural.js", 'Production_LUTO_2_Non-Agricultural');
      await loadScript("./data/Production_LUTO_3_Agricultural_Management.js", 'Production_LUTO_3_Agricultural_Management');
      await loadScript("./data/Production_achive_percent.js", 'Production_achive_percent');
      await loadScript("./data/Production_sum_1_Commodity.js", 'Production_sum_1_Commodity');
      await loadScript("./data/Production_sum_2_Type.js", 'Production_sum_2_Type');

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['Production']);

      // Load initial data for the default category
      loadDataForCategory(selectMapCategory.value);

      updateMapOverlay();
      updateChartSeries();

      nextTick(() => { dataLoaded.value = true; });
    });

    // Watch for category changes to reload data
    watch(selectMapCategory, (newCategory) => {
      loadDataForCategory(newCategory);
    });

    // Watch for changes to validate selections using DataConstructor
    watch([selectMapCategory, selectMapCommodity, selectMapAgMgt, selectYear], () => {
      // Reset values if they're no longer valid options using DataConstructor
      const validCommodityOptions = dataConstructor.getAvailableKeysAtNextLevel({});
      if (validCommodityOptions.length > 0 && !validCommodityOptions.includes(selectMapCommodity.value)) {
        selectMapCommodity.value = validCommodityOptions[0];
      }

      if (selectMapCategory.value === 'Ag Mgt') {
        const agMgtFixedLevels = {};
        if (selectMapCommodity.value) {
          agMgtFixedLevels.level_1 = selectMapCommodity.value;
        }
        const validAgMgtOptions = dataConstructor.getAvailableKeysAtNextLevel(agMgtFixedLevels);
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectMapAgMgt.value)) {
          selectMapAgMgt.value = validAgMgtOptions[0];
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

    const updateChartSeries = () => {
      const dsKey = getChartData();
      selectChartDataset.value = {
        ...window.Chart_default_options,
        chart: { height: 500 },
        yAxis: {
          title: {
            text: "tonnes/kilolitre",
          },
        },
        series: window[dsKey][selectRegion.value],
      };
    };

    const updateMapOverlay = () => {
      if (selectMapCategory.value === 'Ag') {
        mapVarName.value = MapRegister["Ag"]["name"];
        mapVarPath.value = [selectMapCommodity.value, selectYear.value] || [];
      } else if (selectMapCategory.value === 'Ag Mgt') {
        mapVarName.value = MapRegister["Ag Mgt"]["name"];
        mapVarPath.value = [selectMapCommodity.value, selectMapAgMgt.value, selectYear.value] || [];
      } else if (selectMapCategory.value === 'Non-Ag') {
        mapVarName.value = MapRegister["Non-Ag"]["name"];
        mapVarPath.value = [selectMapCommodity.value, selectYear.value] || [];
      }
    };

    return {
      yearIndex,
      isDrawerOpen,
      toggleDrawer,
      dataLoaded,

      availableYears,
      availableCategories,

      availableMapAgMgt,
      availableMapCommodities,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,

      selectChartLevel,

      selectRegion,
      selectChartDataset,

      selectMapCategory,
      selectMapAgMgt,
      selectMapCommodity,
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


      <!-- Data selection controls container - Categories, AgMgt, Commodity selections -->
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

        <!-- Commodity options -->
        <div v-if="dataLoaded" class="flex items-start border-t border-white/10 pt-1">
          <div v-if="!isDrawerOpen && availableMapCommodities.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Commodity:</span>
            <button v-for="(val, key) in availableMapCommodities" :key="key"
              @click="selectMapCommodity = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapCommodity === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="dataLoaded && isDrawerOpen && (selectMapCategory === 'Ag' || selectMapCategory === 'Non-Ag')" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-if="selectMapCategory === 'Ag'" v-for="(val, key) in availableChartAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
            <button v-if="selectMapCategory === 'Non-Ag'" v-for="(val, key) in availableChartNonAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category when drawer is closed) -->
        <div v-if="dataLoaded && selectMapCategory === 'Ag Mgt'" class="flex items-start border-t border-white/10 pt-1">
          <div v-if="!isDrawerOpen && availableMapAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableMapAgMgt" :key="key"
              @click="selectMapAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapAgMgt === val}">
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
            :chartData="selectChartDataset" 
            :draggable="true"
            :zoomable="true"
            style="width: 100%; height: 200px;">
          </chart-container>
        </div>
      </div>

      

    </div>
  `
};