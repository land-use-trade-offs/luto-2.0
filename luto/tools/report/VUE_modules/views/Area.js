window.AreaView = {
  setup() {
    const { ref, onMounted, inject, computed, watch, nextTick } = Vue;

    const selectRegion = inject('globalSelectedRegion');
    const isDrawerOpen = ref(false);
    const yearIndex = ref(0);

    // Data selection and visualization state
    const selectChartData = ref([]);
    const mapPathName = ref({});
    const mapSelectKey = ref([]);

    // Global selection state
    const selectYear = ref(2020);

    // Category selection state
    const selectMapCategory = ref('Ag');
    const selectMapAgMgt = ref('Environmental Plantings');
    const selectMapWater = ref('dry');
    const selectMapLanduse = ref('Beef - modified land');


    // Chart selection state
    const selectChartLevel = ref('Landuse');
    const selectChartItem = ref('Beef - modified land');

    const availableYears = ref([]);
    const availableCategories = ref(['Ag', 'Ag Mgt', 'Non-Ag']);


    // Chart data
    const dataLoaded = ref(false);

    // Centralized function to navigate nested data structure based on current selections
    const selectMapData = (path = []) => {
      // Start with the appropriate data object based on category
      let dataSource;
      if (selectMapCategory.value === 'Ag') {
        dataSource = window.map_area_Ag;
      } else if (selectMapCategory.value === 'Ag Mgt') {
        dataSource = window.map_area_Am;
      } else if (selectMapCategory.value === 'Non-Ag') {
        dataSource = window.map_area_NonAg;
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
    const getMapOptionsForLevel = (level) => {
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectMapCategory.value !== 'Ag Mgt' || !window.map_area_Am) return [];
        return Object.keys(window.map_area_Am);
      }
      if (level === 'water') {
        // Water options depend on category and possibly ag mgt selection
        if (selectMapCategory.value === 'Ag') {
          return Object.keys(window.map_area_Ag || {});
        } else if (selectMapCategory.value === 'Ag Mgt') {
          const agMgtData = selectMapData([selectMapAgMgt.value]);
          return agMgtData ? Object.keys(agMgtData) : [];
        }
        return [];
      }
      if (level === 'landuse') {
        // Landuse options depend on category and previous selections
        if (selectMapCategory.value === 'Ag') {
          const waterData = selectMapData([selectMapWater.value]);
          return waterData ? Object.keys(waterData) : [];
        } else if (selectMapCategory.value === 'Ag Mgt') {
          const waterData = selectMapData([selectMapAgMgt.value, selectMapWater.value]);
          return waterData ? Object.keys(waterData) : [];
        } else if (selectMapCategory.value === 'Non-Ag') {
          return Object.keys(window.map_area_NonAg || {});
        }
        return [];
      }

      return [];
    };

    const getChartOptionsForLevel = (level) => {
      if (level === 'ag') {
        // Ag options only available in 'Ag' category
        if (selectMapCategory.value !== 'Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Area']['Ag']);
      }
      if (level === 'agMgt') {
        // Ag Mgt options only available in 'Ag Mgt' category
        if (selectMapCategory.value !== 'Ag Mgt' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Area']['Ag Mgt']);
      }
      if (level === 'nonag') {
        // Non-Ag options only available in 'Non-Ag' category
        if (selectMapCategory.value !== 'Non-Ag' || !window.DataService) return [];
        return Object.keys(window.DataService.ChartPaths['Area']['Non-Ag']);
      }
      return [];
    };

    const getChartData = () => {
      // Map the visible "Chart level" label to the loaded dataset key
      const chartKeyMap = {
        'Ag': {
          'Landuse': 'Area_Ag_1_Land-use',
          'Water': 'Area_Ag_2_Water_supply',
        },
        'Ag Mgt': {
          'Mgt Type': 'Area_Am_1_Type',
          'Water': 'Area_Am_2_Water_supply',
          'Landuse': 'Area_Am_3_Land-use',
        },
        'Non-Ag': {
          // Only this file is loaded in onMounted
          'Landuse': 'Area_NonAg_1_Land-use',
        },
      };
      return chartKeyMap[selectMapCategory.value]?.[selectChartLevel.value] || null;
    };

    const updateChartSeries = () => {
      const dsKey = getChartData();
      selectChartData.value = {
        ...window.Chart_default_options,
        chart: { height: 500 },
        yAxis: {
          title: {
            text: "Area (ha)",
          },
        },
        series: window[dsKey][selectRegion.value],
      };
    };

    // Computed properties using the centralized functions
    const availableMapAgMgt = computed(() => getMapOptionsForLevel('agMgt'));
    const availableMapWater = computed(() => getMapOptionsForLevel('water'));
    const availableMapLanduse = computed(() => getMapOptionsForLevel('landuse'));

    const availableChartAg = computed(() => getChartOptionsForLevel('ag'));
    const availableChartAgMgt = computed(() => getChartOptionsForLevel('agMgt'));
    const availableChartNonAg = computed(() => getChartOptionsForLevel('nonag'));


    onMounted(async () => {

      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');
      await loadScript("./data/Area_overview_2_Category.js", 'Area_overview_2_Category');
      await loadScript("./data/Area_overview_2_Category.js", 'Area_overview_2_Category');
      await loadScript(`${window.MapService.mapCategories['Area']['Ag']}`, 'map_area_Ag');
      await loadScript(`${window.MapService.mapCategories['Area']['Ag Mgt']}`, 'map_area_Am');
      await loadScript(`${window.MapService.mapCategories['Area']['Non-Ag']}`, 'map_area_NonAg');

      // Chart data
      await loadScript("./data/Area_Ag_1_Land-use.js", 'Area_Ag_1_Land-use');
      await loadScript("./data/Area_Ag_2_Water_supply.js", 'Area_Ag_2_Water_supply');
      await loadScript("./data/Area_Am_1_Type.js", 'Area_Am_1_Type');
      await loadScript("./data/Area_Am_2_Water_supply.js", 'Area_Am_2_Water_supply');
      await loadScript("./data/Area_Am_3_Land-use.js", 'Area_Am_3_Land-use');
      await loadScript("./data/Area_NonAg_1_Land-use.js", 'Area_NonAg_1_Land-use');


      availableYears.value = window.Supporting_info.years;
      availableCategories.value = Object.keys(window.MapService.mapCategories['Area']);

      if (selectMapCategory.value === 'Ag') {
        mapPathName.value = 'window.map_area_Ag';
        mapSelectKey.value = [selectMapWater.value, selectMapLanduse.value, selectYear.value];
      } else if (selectMapCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_area_Am';
        mapSelectKey.value = [selectMapAgMgt.value, selectMapWater.value, selectMapLanduse.value, selectYear.value];
      } else if (selectMapCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_area_NonAg';
        mapSelectKey.value = [selectMapLanduse.value, selectYear.value];
      }

      // Set the initial chart series based on current selections
      updateChartSeries();

      console.log(getChartData());

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        // Set dataLoaded to true after all data has been processed and the DOM has updated
        dataLoaded.value = true;
      });


    });

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch([selectRegion, selectMapCategory, selectMapAgMgt, selectMapWater, selectMapLanduse, selectYear], () => {
      // Reset values if they're no longer valid options
      if (selectMapCategory.value === 'Ag Mgt') {
        const validAgMgtOptions = getMapOptionsForLevel('agMgt');
        if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectMapAgMgt.value)) {
          selectMapAgMgt.value = validAgMgtOptions[0];
        }
      }

      const validWaterOptions = getMapOptionsForLevel('water');
      if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectMapWater.value)) {
        selectMapWater.value = validWaterOptions[0];
      }

      const validLanduseOptions = getMapOptionsForLevel('landuse');
      if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
        selectMapLanduse.value = validLanduseOptions[0];
      }

      // Set map configuration based on category
      if (selectMapCategory.value === 'Ag') {
        mapPathName.value = 'window.map_area_Ag';
        mapSelectKey.value = [selectMapWater.value, selectMapLanduse.value, selectYear.value];
      } else if (selectMapCategory.value === 'Ag Mgt') {
        mapPathName.value = 'window.map_area_Am';
        mapSelectKey.value = [selectMapAgMgt.value, selectMapWater.value, selectMapLanduse.value, selectYear.value];
      } else if (selectMapCategory.value === 'Non-Ag') {
        mapPathName.value = 'window.map_area_NonAg';
        mapSelectKey.value = [selectMapLanduse.value, selectYear.value];
      }


    });

    // Refresh chart when category/level/region change
    watch([selectMapCategory, selectChartLevel, selectRegion], () => {
      updateChartSeries();
    });



    return {
      yearIndex,
      isDrawerOpen,
      toggleDrawer,
      dataLoaded,

      availableYears,
      availableCategories,

      availableMapAgMgt,
      availableMapWater,
      availableMapLanduse,

      availableChartAg,
      availableChartAgMgt,
      availableChartNonAg,

      selectChartLevel,
      selectChartItem,

      selectRegion,
      selectChartData,

      selectMapCategory,
      selectMapAgMgt,
      selectMapWater,
      selectMapLanduse,
      selectYear,

      mapSelectKey,
      mapPathName,
    };
  },
  template: `
    <div class="relative w-full h-screen">


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

        <!-- Ag Mgt options (only for Ag Mgt category) -->
        <div 
          class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && !isDrawerOpen && availableMapAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
            <button v-for="(val, key) in availableMapAgMgt" :key="key"
              @click="selectMapAgMgt = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapAgMgt === val}">
              {{ val }}
            </button>
          </div>
          <div v-else-if="dataLoaded && isDrawerOpen && availableChartAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
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
        <div 
          class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && !isDrawerOpen && availableMapWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
            <button v-for="(val, key) in availableMapWater" :key="key"
              @click="selectMapWater = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapWater === val}">
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
        <div 
          class="flex items-start border-t border-white/10 pt-1">
          <div v-if="dataLoaded && ! isDrawerOpen" class="flex flex-wrap gap-1 max-w-[300px]">
            <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
            <button v-for="(val, key) in availableMapLanduse" :key="key"
              @click="selectMapLanduse = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
              :class="{'bg-sky-500 text-white': selectMapLanduse === val}">
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

      

      

      <!-- Map container with slide-out chart drawer -->
      <div style="position: relative; width: 100%; height: 100%; overflow: hidden;">
        <!-- Map component takes full space -->
        <regions-map 
          :mapPathName="mapPathName"
          :mapKey="mapSelectKey"
          style="width: 100%; height: 100%;">
        </regions-map>

        <!-- Drawer toggle button - Controls visibility of the chart panel -->
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
  `
};
