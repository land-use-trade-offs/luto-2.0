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
    const getMapOptionsForLevel = window.MapService.getMapOptionsForLevel;

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
    const isUpdating = ref(false); // Flag to prevent circular updates

    // We need special handlers for GHG data since it has a unique structure
    const availableGHGSource = computed(() => {
      if (selectMapCategory.value !== 'Ag' || !window.map_GHG_Ag) return [];
      return Object.keys(window.map_GHG_Ag || {});
    });

    const availableAgMgt = computed(() => {
      if (selectMapCategory.value !== 'Ag Mgt' || !window.map_GHG_Am) return [];
      return Object.keys(window.map_GHG_Am || {});
    });

    const availableWater = computed(() => {
      if (selectMapCategory.value === 'Ag' && window.map_GHG_Ag) {
        const ghgData = window.map_GHG_Ag[selectMapGHGSource.value];
        if (ghgData && ghgData[selectMapLanduse.value]) {
          return Object.keys(ghgData[selectMapLanduse.value]);
        }
      } else if (selectMapCategory.value === 'Ag Mgt' && window.map_GHG_Am) {
        const amData = window.map_GHG_Am[selectMapAgMgt.value];
        if (amData && amData[selectMapLanduse.value]) {
          return Object.keys(amData[selectMapLanduse.value]);
        }
      }
      return [];
    });

    const availableLanduse = computed(() => {
      if (selectMapCategory.value === 'Ag' && window.map_GHG_Ag) {
        const ghgData = window.map_GHG_Ag[selectMapGHGSource.value];
        if (ghgData) {
          return Object.keys(ghgData);
        }
      } else if (selectMapCategory.value === 'Ag Mgt' && window.map_GHG_Am) {
        const amData = window.map_GHG_Am[selectMapAgMgt.value];
        if (amData) {
          return Object.keys(amData);
        }
      } else if (selectMapCategory.value === 'Non-Ag' && window.map_GHG_NonAg) {
        return Object.keys(window.map_GHG_NonAg);
      }
      return [];
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

      // Initialize with first available options for each category
      if (selectMapCategory.value === 'Ag' && window.map_GHG_Ag) {
        const ghgSourceOptions = Object.keys(window.map_GHG_Ag);
        if (ghgSourceOptions.length > 0 && !ghgSourceOptions.includes(selectMapGHGSource.value)) {
          selectMapGHGSource.value = ghgSourceOptions[0];
        }

        // Set landuse to first available for the current GHG source
        const ghgData = window.map_GHG_Ag[selectMapGHGSource.value];
        if (ghgData) {
          const landuseOptions = Object.keys(ghgData);
          if (landuseOptions.length > 0 && !landuseOptions.includes(selectMapLanduse.value)) {
            selectMapLanduse.value = landuseOptions[0];
          }

          // Set water to first available for current GHG source and landuse
          const waterData = ghgData[selectMapLanduse.value];
          if (waterData) {
            const waterOptions = Object.keys(waterData);
            if (waterOptions.length > 0 && !waterOptions.includes(selectMapWater.value)) {
              selectMapWater.value = waterOptions[0];
            }
          }
        }
      } else if (selectMapCategory.value === 'Ag Mgt' && window.map_GHG_Am) {
        const agMgtOptions = Object.keys(window.map_GHG_Am);
        if (agMgtOptions.length > 0 && !agMgtOptions.includes(selectMapAgMgt.value)) {
          selectMapAgMgt.value = agMgtOptions[0];
        }

        // Set landuse to first available for the current Ag Mgt
        const amData = window.map_GHG_Am[selectMapAgMgt.value];
        if (amData) {
          const landuseOptions = Object.keys(amData);
          if (landuseOptions.length > 0 && !landuseOptions.includes(selectMapLanduse.value)) {
            selectMapLanduse.value = landuseOptions[0];
          }

          // Set water to first available for current Ag Mgt and landuse
          const waterData = amData[selectMapLanduse.value];
          if (waterData) {
            const waterOptions = Object.keys(waterData);
            if (waterOptions.length > 0 && !waterOptions.includes(selectMapWater.value)) {
              selectMapWater.value = waterOptions[0];
            }
          }
        }
      } else if (selectMapCategory.value === 'Non-Ag' && window.map_GHG_NonAg) {
        const landuseOptions = Object.keys(window.map_GHG_NonAg);
        if (landuseOptions.length > 0 && !landuseOptions.includes(selectMapLanduse.value)) {
          selectMapLanduse.value = landuseOptions[0];
        }
      }

      // Use nextTick to ensure the data is processed before rendering the UI components
      nextTick(() => {
        // Set dataLoaded to true after all data has been processed and the DOM has updated
        nextTick(() => {
          isUpdating.value = true; // Prevent watch triggers during initial setup

          dataLoaded.value = true;

          // Update chart and map after initial data load
          try {
            updateChartSeries();
            updateMapOverlay();
          } catch (error) {
            console.error('Error during initial data setup:', error);
          }

          isUpdating.value = false; // Allow normal watch operations
        });
      });


    });

    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    // Watch for drawer opening to update chart
    watch(isDrawerOpen, (newValue) => {
      if (isUpdating.value) return; // Prevent updates during state changes

      if (newValue && dataLoaded.value) {
        // Use nextTick to ensure DOM is updated before chart operations
        nextTick(() => {
          try {
            updateChartSeries();
          } catch (error) {
            console.error('Error updating chart when drawer opens:', error);
          }
        });
      }
    });

    watch([selectMapCategory, selectMapGHGSource, selectMapAgMgt, selectMapWater, selectMapLanduse, selectYear], () => {
      try {
        if (isUpdating.value || !dataLoaded.value) return; // Prevent circular updates

        // Use nextTick to ensure all reactive updates are processed
        nextTick(() => {
          try {
            isUpdating.value = true;

            // Reset values if they're no longer valid options based on actual data structure
            if (selectMapCategory.value === 'Ag' && window.map_GHG_Ag) {
              const validGHGSourceOptions = Object.keys(window.map_GHG_Ag);
              if (validGHGSourceOptions.length > 0 && !validGHGSourceOptions.includes(selectMapGHGSource.value)) {
                selectMapGHGSource.value = validGHGSourceOptions[0];
              }

              const ghgData = window.map_GHG_Ag[selectMapGHGSource.value];
              if (ghgData) {
                const validLanduseOptions = Object.keys(ghgData);
                if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
                  selectMapLanduse.value = validLanduseOptions[0];
                }

                const waterData = ghgData[selectMapLanduse.value];
                if (waterData) {
                  const validWaterOptions = Object.keys(waterData);
                  if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectMapWater.value)) {
                    selectMapWater.value = validWaterOptions[0];
                  }
                }
              }
            } else if (selectMapCategory.value === 'Ag Mgt' && window.map_GHG_Am) {
              const validAgMgtOptions = Object.keys(window.map_GHG_Am);
              if (validAgMgtOptions.length > 0 && !validAgMgtOptions.includes(selectMapAgMgt.value)) {
                selectMapAgMgt.value = validAgMgtOptions[0];
              }

              const amData = window.map_GHG_Am[selectMapAgMgt.value];
              if (amData) {
                const validLanduseOptions = Object.keys(amData);
                if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
                  selectMapLanduse.value = validLanduseOptions[0];
                }

                const waterData = amData[selectMapLanduse.value];
                if (waterData) {
                  const validWaterOptions = Object.keys(waterData);
                  if (validWaterOptions.length > 0 && !validWaterOptions.includes(selectMapWater.value)) {
                    selectMapWater.value = validWaterOptions[0];
                  }
                }
              }
            } else if (selectMapCategory.value === 'Non-Ag' && window.map_GHG_NonAg) {
              const validLanduseOptions = Object.keys(window.map_GHG_NonAg);
              if (validLanduseOptions.length > 0 && !validLanduseOptions.includes(selectMapLanduse.value)) {
                selectMapLanduse.value = validLanduseOptions[0];
              }
            }

            // Update map configuration
            updateMapOverlay();

            isUpdating.value = false;
          } catch (e) {
            console.warn('Error in nextTick watch function:', e);
            isUpdating.value = false;
          }
        });
      } catch (e) {
        console.warn('Error in watch function:', e);
        isUpdating.value = false;
      }
    });

    // Refresh chart when category/level/region change
    watch([selectMapCategory, selectChartLevel, selectRegion], () => {
      if (isUpdating.value) return; // Prevent updates during state changes
      if (dataLoaded.value) {
        updateChartSeries();
      }
    });



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