window.HomeView = {

  setup(props, { emit }) {

    const { ref, onMounted, watch, computed, inject, nextTick } = Vue;
    const loadScript = window.loadScript;


    // Define reactive variables
    const dataLoaded = ref(false);
    const selectRegion = inject('globalSelectedRegion');
    const selectDataType = inject('globalSelectedDataType');
    const selectYear = ref(2020);
    const yearIndex = ref(0);
    const availableYears = ref([]);
    const runScenario = ref({})

    //  Reactive data
    const chartOverview = ref({});
    const colorsRanking = ref({});
    const availableDatasets = ref({
      'Economics_overview': { 'type': 'Economics', 'unit': 'AUD' },
      'Area_overview_2_Category': { 'type': 'Area', 'unit': 'Hectares' },
      'GHG_overview': { 'type': 'GHG', 'unit': 'Mt CO2e' },
      'Water_overview_NRM_region_2_Type': { 'type': 'Water', 'unit': 'ML' },
      'BIO_quality_overview_1_Type': { 'type': 'Biodiversity', 'unit': 'Weighted score (ha)' },
    });
    const selectDataset = ref('Area_overview_2_Category');
    const DtypeSubCategories = computed(() => {
      return window.DataService.getSubcategories(selectDataType.value);
    });
    const selectSubcategory = ref('');


    // Functions
    const changeDataset = async (datasetName) => {
      try {
        // Load the selected dataset script
        await loadScript(`./data/${datasetName}.js`, datasetName);

        // Directly update the chartOverview with the new dataset
        chartOverview.value = {
          ...window['Chart_default_options'],
          chart: {
            height: 550,
          },
          yAxis: {
            title: {
              text: availableDatasets.value[datasetName]['unit']
            }
          },
          series: window[datasetName][selectRegion.value],
          colors: window['Supporting_info'].colors,
        };
      } catch (error) {
        console.error(`Error loading dataset ${datasetName}:`, error);
      }
    };


    // Load scripts and data when the component is mounted
    onMounted(async () => {
      try {

        // Load required data
        await loadScript("./data/Supporting_info.js", 'Supporting_info');
        await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');
        await loadScript("./data/Biodiversity_ranking.js", 'Biodiversity_ranking');
        await loadScript("./data/GHG_ranking.js", 'GHG_ranking');
        await loadScript("./data/Water_ranking.js", 'Water_ranking');
        await loadScript("./data/Area_ranking.js", 'Area_ranking');
        await loadScript("./data/Economics_ranking.js", 'Economics_ranking');
        await loadScript("./services/DataService.js", 'DataService');

        // Set initial year to first available year
        availableYears.value = window.Supporting_info.years;
        selectYear.value = availableYears.value[0];
        selectSubcategory.value = DtypeSubCategories.value[0];
        colorsRanking.value = window.Supporting_info.colors_ranking;
        runScenario.value = {
          'SSP': window.Supporting_info.model_run_settings.filter(item => item.parameter === "SSP")[0]['val'],
          'GHG': window.Supporting_info.model_run_settings.filter(item => item.parameter === "GHG_EMISSIONS_LIMITS")[0]['val'],
          'BIO_CUT': window.Supporting_info.model_run_settings.filter(item => item.parameter === "GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT")[0]['val'],
          'BIO_GBF2': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_2")[0]['val'],
          'BIO_GBF3': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_3")[0]['val'],
          'BIO_GBF4_SNES': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_4_SNES")[0]['val'],
          'BIO_GBF4_ECNES': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_4_ECNES")[0]['val'],
          'BIODIVERSITY_TARGET_GBF_8': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_8")[0]['val'],
        }


        // RankingData component is now included in index.html

        // Initialize the overview chart with the selected dataset
        await changeDataset(selectDataset.value);

        // Mark data as loaded and use nextTick to ensure UI updates
        dataLoaded.value = true;
        await nextTick();

      } catch (error) {
        console.error("Error loading dependencies:", error);
      }
    });



    // Watch for changes and then make reactive updates
    watch(
      selectDataset,
      (newDataset) => {
        selectDataType.value = availableDatasets.value[newDataset].type;
        selectSubcategory.value = DtypeSubCategories.value[0];
      }
    );
    watch(
      [selectRegion, selectDataType],
      (newValue, oldValue) => {
        changeDataset(selectDataset.value);
      }
    );
    watch(
      yearIndex,
      (newIndex) => {
        selectYear.value = availableYears.value[newIndex];
      }
    );


    return {
      dataLoaded,
      availableYears,
      availableDatasets,
      DtypeSubCategories,
      yearIndex,
      runScenario,

      selectRegion,
      selectDataset,
      selectDataType,
      selectYear,
      selectSubcategory,

      chartOverview,
      colorsRanking,
      changeDataset,
    };
  },

  // This template is a fallback that will be replaced by the loaded template
  template: `
    <div>

      <div class="flex flex-col">

        <!-- Rank cards -->
        <p class="text-[#505051] font-bold p-1 pt-8"> SSP - {{ runScenario.SSP }} | GHG - {{ runScenario.GHG }} | Biodiversity - {{ runScenario.BIO_GBF2 }}</p>
        <div class="mb-4 mr-4">
          <ranking-cards 
            v-if="dataLoaded"
            :selectRegion="selectRegion"
            :selectYear="selectYear">
          </ranking-cards>
        </div>

        <!-- Title for map and chart -->
        <div class="flex items-center justify-between">
          <p class="text-[#505051] w-[500px] text font-bold p-1 pt-8">Map and Statistics</p>
          <p class="flex-1 text-[#505051] font-bold ml-4 p-1 pt-8">{{ selectDataType }} overview for {{ selectRegion }}</p>
        </div>

        <div class="flex mr-4 gap-4 mb-4">

          <div class="flex flex-col rounded-[10px] bg-white shadow-md w-[500px]">

            <!-- Buttons -->
            <div class="flex items-center justify-between w-full">
              <div class="text-[0.8rem] ml-2">
                <p>Region: <strong>{{ selectRegion }}</strong></p>
              </div>
              <div class="flex items-center justify-end p-2">
                <div class="flex space-x-1">
                  <button v-for="(data, key) in availableDatasets" :key="key"
                    @click="selectDataset = key"
                    class="bg-[#e8eaed] text-[#1f1f1f] text-[0.8rem] px-1 py-1 rounded"
                    :class="{'bg-sky-500 text-white': selectDataset === key}">
                    {{ data.type }}
                  </button>
                </div>
              </div>
            </div>

            <hr class="border-gray-300">

            <!-- Map -->
            <div class="relative">
              <div class="absolute flex-col w-full top-1 left-2 right-2 pr-4 justify-between items-center z-10">
                
              <div class="flex flex-col">
                <div class="flex items-center justify-between">
                  <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
                  <div class="flex space-x-1 mr-4">
                    <button v-for="cat in DtypeSubCategories" :key="cat"
                      @click="selectSubcategory = cat"
                      class="bg-[#e8eaed] text-[#1f1f1f] text-[0.6rem] px-1 rounded"
                      :class="{'bg-sky-500 text-white': selectSubcategory === cat}">
                      {{ cat }}
                    </button>
                  </div>
                </div>

                <el-slider 
                  v-if="availableYears.length > 0"
                  class="flex-1 max-w-[150px] pt-2 pl-2" 
                  v-model="yearIndex"
                  size="small"
                  :show-tooltip="false"
                  :min="0" 
                  :max="availableYears.length - 1"
                  :step="1"
                  :format-tooltip="index => availableYears[index]"
                  :marks="availableYears.reduce((acc, year, index) => ({ ...acc, [index]: year }), {})"
                  @input="(index) => { yearIndex = index; }"
                />
        
                </div>
              </div>
              <map-geojson 
                v-if="dataLoaded"
                :height="'530px'" 
                :selectDataType="selectDataType" 
                :selectYear="selectYear" 
                :selectSubcategory="selectSubcategory"
                :legendObj="colorsRanking"
              />
            </div>

          </div>

          <!-- Statistics Chart -->
          <chart-container 
          v-if="dataLoaded"
          class="flex-1 rounded-[10px] bg-white shadow-md"
          :chartData="chartOverview"></chart-container>

        </div>
        
      </div>
    </div>
  `,
};
