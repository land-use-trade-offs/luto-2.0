window.ProductionView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    const selectRegion = inject('globalSelectedRegion');
    const isDrawerOpen = ref(false);
    const yearIndex = ref(0);

    // Data selection and visualization state
    const selectDataset = ref({});
    const selectChartLevel = ref('Agricultural');
    const mapPathName = ref({});
    const mapSelectKey = ref([]);

    // Category selection state
    const selectCategory = ref('Ag');
    const selectAgMgt = ref('Sparagopsis taxiformis');
    const selectCommodity = ref('apple');
    const selectYear = ref(2020);

    const availableYears = ref([]);
    const availableCategories = ref([]);

    const dataLoaded = ref(false);

    // Centralized function to navigate nested data structure based on current selections
    const getNestedData = (path = []) => {
      // Start with the appropriate data object based on category
      let dataSource;
      if (selectCategory.value === 'Ag') {
        dataSource = window.map_quantities_Ag;
      } else if (selectCategory.value === 'Ag Mgt') {
        dataSource = window.map_quantities_Am;
      } else if (selectCategory.value === 'Non-Ag') {
        dataSource = window.map_quantities_NonAg;
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
        if (selectCategory.value !== 'Ag Mgt' || !window.map_quantities_Am) return [];
        return Object.keys(window.map_quantities_Am);
      }

      if (level === 'commodity') {
        // Commodity options depend on category and possibly ag mgt selection
        if (selectCategory.value === 'Ag') {
          return Object.keys(window.map_quantities_Ag || {});
        } else if (selectCategory.value === 'Ag Mgt') {
          const agMgtData = getNestedData([selectAgMgt.value]);
          return agMgtData ? Object.keys(agMgtData) : [];
        } else if (selectCategory.value === 'Non-Ag') {
          return Object.keys(window.map_quantities_NonAg || {});
        }
        return [];
      }

      return [];
    };

    // Computed properties using the centralized functions
    const availableAgMgt = computed(() => getOptionsForLevel('agMgt'));
    const availableCommodities = computed(() => getOptionsForLevel('commodity'));

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');

      // Load Production chart data files
      await loadScript("./data/Production_LUTO_1_Agricultural.js", 'Production_LUTO_1_Agricultural');
      await loadScript("./data/Production_LUTO_2_Non-Agricultural.js", 'Production_LUTO_2_Non-Agricultural');
      await loadScript("./data/Production_LUTO_3_Agricultural_Management.js", 'Production_LUTO_3_Agricultural_Management');
      await loadScript("./data/Production_achive_percent.js", 'Production_achive_percent');
      await loadScript("./data/Production_sum_1_Commodity.js", 'Production_sum_1_Commodity');
      await loadScript("./data/Production_sum_2_Type.js", 'Production_sum_2_Type');

      // Load map data for all categories
      await loadScript(`${window.MapService.mapCategories['Production']['Ag']}`, 'map_quantities_Ag');
      await loadScript(`${window.MapService.mapCategories['Production']['Ag Mgt']}`, 'map_quantities_Am');
      await loadScript(`${window.MapService.mapCategories['Production']['Non-Ag']}`, 'map_quantities_NonAg');

      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['Production']);

      if (selectCategory.value === 'Ag') {
        mapPathName.value = 'window.map_quantities_Ag';
        mapSelectKey.value = [selectCommodity.value, selectYear.value];
      } else if (selectCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_quantities_Am';
        mapSelectKey.value = [selectAgMgt.value, selectCommodity.value, selectYear.value];
      } else if (selectCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_quantities_NonAg';
        mapSelectKey.value = [selectCommodity.value, selectYear.value];
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

    watch([selectCategory, selectAgMgt, selectCommodity, selectYear], () => {
      // Reset values if they're no longer valid options
      if (selectCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = getOptionsForLevel('agMgt');
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectAgMgt.value)) {
          selectAgMgt.value = validAgMgtOptions[0];
        }
      }

      const validCommodityOptions = getOptionsForLevel('commodity');
      if (validCommodityOptions.length > 0 && !validCommodityOptions.includes(selectCommodity.value)) {
        selectCommodity.value = validCommodityOptions[0];
      }

      // Set map configuration based on category
      if (selectCategory.value === 'Ag') {
        mapPathName.value = 'window.map_quantities_Ag';
        mapSelectKey.value = [selectCommodity.value, selectYear.value];
      } else if (selectCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_quantities_Am';
        mapSelectKey.value = [selectAgMgt.value, selectCommodity.value, selectYear.value];
      } else if (selectCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_quantities_NonAg';
        mapSelectKey.value = [selectCommodity.value, selectYear.value];
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
      return chartKeyMap[selectCategory.value]?.[selectChartLevel.value] || 'Production_LUTO_1_Agricultural';
    };

    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        return ['Agricultural', 'Overview', 'Commodity', 'Type'];
      }
      if (level === 'agMgt') {
        return ['Agricultural Management', 'Overview', 'Commodity', 'Type'];
      }
      if (level === 'nonag') {
        return ['Non-Agricultural', 'Overview', 'Commodity', 'Type'];
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
            text: "tonnes/kilolitre",
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
      availableCommodities,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,
      selectChartLevel,

      selectRegion,
      selectDataset,

      selectCategory,
      selectAgMgt,
      selectCommodity,
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


      <!-- Data selection controls container - Categories, AgMgt, Commodity selections -->
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

        <!-- Commodity options -->
        <div class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && !isDrawerOpen && availableCommodities.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Commodity:</span>
            <button v-for="(val, key) in availableCommodities" :key="key"
              @click="selectCommodity = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectCommodity === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="dataLoaded && isDrawerOpen && (selectCategory.value === 'Ag' || selectCategory.value === 'Non-Ag')" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Chart level:</span>
            <button v-if="selectCategory.value === 'Ag'" v-for="(val, key) in availableChartAg" :key="key"
              @click="selectChartLevel = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectChartLevel === val}">
              {{ val }}
            </button>
            <button v-if="selectCategory.value === 'Non-Ag'" v-for="(val, key) in availableChartNonAg" :key="key"
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