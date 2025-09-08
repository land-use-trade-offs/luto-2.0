window.HomeView = {

  setup() {

    const { ref, onMounted, watch, computed, inject, nextTick } = Vue;

    // Data service
    const chartRegister = window.DataService.chartCategories;   // DataService has been registered in index.html      [DataService.js]
    const loadScript = window.loadScript;                       // DataConstructor has been registered in index.html  [helpers.js]

    // Global variables
    const selectRegion = inject('globalSelectedRegion');
    const ChartData = ref({});
    const rankingData = ref({});
    const colorsRanking = ref({});

    // Available selections
    const availableYears = ref([]);
    const availableChartCategories = ref([]);
    const availableChartSubCategories = ref([]);
    const availableUnit = {
      'Area': 'Hectares',
      'Economics': 'AUD',
      'GHG': 'Mt CO2e',
      'Water': 'ML',
      'Biodiversity': 'Relative Percentage (Pre-1750 = 100%)',
    };
    const RankSubcategoriesRename = {
      'Agricultural Landuse': 'Ag',
      'Agricultural Management': 'Ag Mgt',
      'Non-Agricultural Landuse': 'Non-Ag',
      'Non-Agricultural land-use': 'Non-Ag',
    };
    const availableRankSubcategories = ref([]);

    // Default selections
    const yearIndex = ref(0);
    const selectYear = ref(null);
    const selectChartCategory = ref('');
    const selectChartSubCategory = ref('');
    const selectRankingSubCategory = ref('');
    const selectRankingColors = ref({});


    //  Reactive data
    const selectChartData = computed(() => {
      const originalData = ChartData.value?.[selectChartCategory.value]?.[selectChartSubCategory.value]?.[selectRegion.value];
      const seriesData = originalData ? JSON.parse(JSON.stringify(originalData)) : [];
      const seriesColors = seriesData.map(serie => serie.color).filter(color => color);
      const colors = seriesColors.length > 0 ? seriesColors : window['Supporting_info']?.colors || [];
      return {
        ...window['Chart_default_options'],
        chart: {
          height: 440,
        },
        yAxis: {
          title: {
            text: availableUnit[selectChartCategory.value]
          }
        },
        series: seriesData,
        colors: colors,
      };
    });


    const selectRanking = computed(() => {
      const currentRegion = selectRegion.value;
      const currentYear = selectYear.value;
      return {
        economicTotal: rankingData.value['Economics']?.[currentRegion]?.['Total']?.['value']?.[currentYear] || 'N/A',
        economicCost: rankingData.value['Economics']?.[currentRegion]?.['Cost']?.['value']?.[currentYear] || 'N/A',
        economicRevenue: rankingData.value['Economics']?.[currentRegion]?.['Revenue']?.['value']?.[currentYear] || 'N/A',
        areaTotal: rankingData.value['Area']?.[currentRegion]?.['Total']?.['value']?.[currentYear] || 'N/A',
        areaAgLand: rankingData.value['Area']?.[currentRegion]?.['Agricultural Landuse']?.['value']?.[currentYear] || 'N/A',
        areaAgMgt: rankingData.value['Area']?.[currentRegion]?.['Agricultural Management']?.['value']?.[currentYear] || 'N/A',
        areaNonAg: rankingData.value['Area']?.[currentRegion]?.['Non-Agricultural Landuse']?.['value']?.[currentYear] || 'N/A',
        ghgTotal: rankingData.value['GHG']?.[currentRegion]?.['Total']?.['value']?.[currentYear] || 'N/A',
        ghgEmissions: rankingData.value['GHG']?.[currentRegion]?.['GHG emissions']?.['value']?.[currentYear] || 'N/A',
        ghgReduction: rankingData.value['GHG']?.[currentRegion]?.['GHG sequestrations']?.['value']?.[currentYear] || 'N/A',
        waterTotal: rankingData.value['Water']?.[currentRegion]?.['Total']?.['value']?.[currentYear] || 'N/A',
        waterAgLand: rankingData.value['Water']?.[currentRegion]?.['Agricultural Landuse']?.['value']?.[currentYear] || 'N/A',
        waterAgMgt: rankingData.value['Water']?.[currentRegion]?.['Agricultural Management']?.['value']?.[currentYear] || 'N/A',
        waterNonAg: rankingData.value['Water']?.[currentRegion]?.['Non-Agricultural Landuse']?.['value']?.[currentYear] || 'N/A',
        biodiversityTotal: rankingData.value['Biodiversity']?.[currentRegion]?.['Total']?.['value']?.[currentYear] || 'N/A',
        biodiversityAgLand: rankingData.value['Biodiversity']?.[currentRegion]?.['Agricultural Landuse']?.['value']?.[currentYear] || 'N/A',
        biodiversityAgMgt: rankingData.value['Biodiversity']?.[currentRegion]?.['Agricultural Management']?.['value']?.[currentYear] || 'N/A',
        biodiversityNonAg: rankingData.value['Biodiversity']?.[currentRegion]?.['Non-Agricultural land-use']?.['value']?.[currentYear] || 'N/A'
      };
    });


    const runScenario = computed(() => {
      return !window.Supporting_info ? {} : {
        'SSP': window.Supporting_info.model_run_settings.filter(item => item.parameter === "SSP")[0]['val'],
        'GHG': window.Supporting_info.model_run_settings.filter(item => item.parameter === "GHG_EMISSIONS_LIMITS")[0]['val'],
        'BIO_CUT': window.Supporting_info.model_run_settings.filter(item => item.parameter === "GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT")[0]['val'],
        'BIO_GBF2': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_2")[0]['val'],
        'BIO_GBF3': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_3")[0]['val'],
        'BIO_GBF4_SNES': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_4_SNES")[0]['val'],
        'BIO_GBF4_ECNES': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_4_ECNES")[0]['val'],
        'BIODIVERSITY_TARGET_GBF_8': window.Supporting_info.model_run_settings.filter(item => item.parameter === "BIODIVERSITY_TARGET_GBF_8")[0]['val'],
      };
    });


    // Data loaded flag
    const dataLoaded = ref(false);


    onMounted(async () => {

      // Load required data
      await loadScript("./data/Supporting_info.js", 'Supporting_info');
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');
      await loadScript("./data/geo/NRM_AUS.js", 'NRM_AUS');

      // Overview chart data
      const chartOverview_area = chartRegister['Area']['overview'];
      const chartOverview_economics = chartRegister['Economics']['overview'];
      const chartOverview_economics_ag = chartRegister['Economics']['Ag'];
      const chartOverview_economics_agMgt = chartRegister['Economics']['Ag Mgt'];
      const chartOverview_economics_Nonag = chartRegister['Economics']['Non-Ag'];
      const chartOverview_ghg = chartRegister['GHG']['overview'];
      const chartOverview_ghg_ag = chartRegister['GHG']['Ag'];
      const chartOverview_ghg_agMgt = chartRegister['GHG']['Ag Mgt'];
      const chartOverview_ghg_Nonag = chartRegister['GHG']['Non-Ag'];
      const chartOverview_water = chartRegister['Water']['NRM']['overview'];
      const chartOverview_bio_GBF2 = chartRegister['Biodiversity']['GBF2']['overview'];
      const rankingArea = chartRegister['Area']['ranking'];
      const rankingEconomics = chartRegister['Economics']['ranking'];
      const rankingGHG = chartRegister['GHG']['ranking'];
      const rankingWater = chartRegister['Water']['NRM']['ranking'];
      const rankingBiodiversity = chartRegister['Biodiversity']['ranking'];

      await loadScript(chartOverview_area['Source']['path'], chartOverview_area['Source']['name']);
      await loadScript(chartOverview_area['Category']['path'], chartOverview_area['Category']['name']);
      await loadScript(chartOverview_area['Land-use']['path'], chartOverview_area['Land-use']['name']);
      await loadScript(chartOverview_economics['sum']['path'], chartOverview_economics['sum']['name']);
      await loadScript(chartOverview_economics_ag['path'], chartOverview_economics_ag['name']);
      await loadScript(chartOverview_economics_agMgt['path'], chartOverview_economics_agMgt['name']);
      await loadScript(chartOverview_economics_Nonag['path'], chartOverview_economics_Nonag['name']);
      await loadScript(chartOverview_ghg['path'], chartOverview_ghg['name']);
      await loadScript(chartOverview_ghg_ag['path'], chartOverview_ghg_ag['name']);
      await loadScript(chartOverview_ghg_agMgt['path'], chartOverview_ghg_agMgt['name']);
      await loadScript(chartOverview_ghg_Nonag['path'], chartOverview_ghg_Nonag['name']);
      await loadScript(chartOverview_water['Type']['path'], chartOverview_water['Type']['name']);
      await loadScript(chartOverview_bio_GBF2['path'], chartOverview_bio_GBF2['name']);
      await loadScript(rankingArea['path'], rankingArea['name']);
      await loadScript(rankingEconomics['path'], rankingEconomics['name']);
      await loadScript(rankingGHG['path'], rankingGHG['name']);
      await loadScript(rankingWater['path'], rankingWater['name']);
      await loadScript(rankingBiodiversity['path'], rankingBiodiversity['name']);


      ChartData.value = {
        'Area': {
          'Source': window[chartOverview_area['Source']['name']],
          'Category': window[chartOverview_area['Category']['name']],
          'Land-use': window[chartOverview_area['Land-use']['name']],
        },
        'Economics': {
          'Overview': window[chartOverview_economics['sum']['name']],
          'Ag': window[chartOverview_economics_ag['name']],
          'Ag Mgt': window[chartOverview_economics_agMgt['name']],
          'Non-Ag': window[chartOverview_economics_Nonag['name']],
        },
        'GHG': {
          'Overview': window[chartOverview_ghg['name']],
          'Ag': window[chartOverview_ghg_ag['name']],
          'Ag Mgt': window[chartOverview_ghg_agMgt['name']],
          'Non-Ag': window[chartOverview_ghg_Nonag['name']],
        },
        'Water': {
          'Type': window[chartOverview_water['Type']['name']],
        },
        'Biodiversity': {
          'GBF2': window[chartOverview_bio_GBF2['name']],
        },
      };

      rankingData.value = {
        'Area': window[rankingArea['name']],
        'Economics': window[rankingEconomics['name']],
        'GHG': window[rankingGHG['name']],
        'Water': window[rankingWater['name']],
        'Biodiversity': window[rankingBiodiversity['name']],
      };



      //  Set initial values
      availableYears.value = window['Supporting_info']['years'];
      selectYear.value = availableYears.value[0];

      availableChartCategories.value = Object.keys(ChartData.value);
      selectChartCategory.value = availableChartCategories.value[0];

      selectChartSubCategory.value = Object.keys(ChartData.value[selectChartCategory.value])[0];
      const rankingKeys = Object.keys(rankingData.value?.[selectChartCategory.value]?.[selectRegion.value] || {}).filter(key => key !== "Total");
      selectRankingSubCategory.value = rankingKeys[0] || 'N/A';
      colorsRanking.value = window.Supporting_info.colors_ranking;

      await nextTick(() => { dataLoaded.value = true; });


    });

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    watch(selectChartCategory, (newCategory) => {
      availableChartSubCategories.value = Object.keys(ChartData.value[selectChartCategory.value])
      availableRankSubcategories.value = Object.keys(rankingData.value?.[selectChartCategory.value]?.[selectRegion.value] || {}).filter(key => key !== "Total");
      selectChartSubCategory.value = availableChartSubCategories.value[0];
      selectRankingSubCategory.value = availableRankSubcategories.value[0] || 'N/A';
    });

    watch([selectYear, selectRankingSubCategory], (newValues, oldValues) => {
      const [newYear, newSubCategory] = newValues;
      const categoryData = rankingData.value?.[selectChartCategory.value];
      selectRankingColors.value = Object.fromEntries(
        Object.entries(categoryData).map(([region, values]) => [
          region,
          values?.[newSubCategory]?.['color']?.[newYear] || {}
        ])
      );
    });

    return {
      yearIndex,
      runScenario,
      dataLoaded,

      ChartData,
      rankingData,
      RankSubcategoriesRename,
      colorsRanking,

      availableYears,
      availableChartCategories,
      availableChartSubCategories,
      availableRankSubcategories,

      selectYear,
      selectRegion,
      selectChartCategory,
      selectChartSubCategory,
      selectRankingSubCategory,
      selectChartData,
      selectRankingColors,
      selectRanking,
    };
  },

  // This template is a fallback that will be replaced by the loaded template
  template: `
    <div v-if="dataLoaded">

      <div class="flex flex-col">

        <!-- Rank cards -->
        <p class="text-[#505051] font-bold p-1 pt-8"> SSP - {{ runScenario.SSP }} | GHG - {{ runScenario.GHG }} | Biodiversity - {{ runScenario.BIO_GBF2 }}</p>
        <div class="mb-4 mr-4">
          <ranking-cards 
            :selectRankingData="selectRanking">
          </ranking-cards>
        </div>


        <!-- Title for map and chart -->
        <div class="flex items-center justify-between">
          <p class="text-[#505051] w-[500px] text font-bold p-1 pt-8">Map and Statistics</p>
          <p class="flex-1 text-[#505051] font-bold ml-4 p-1 pt-8">{{ selectChartCategory }} overview for {{ selectRegion }}</p>
        </div>

        <!-- Container for Map and Chart -->
        <div class="flex mr-4 gap-4 mb-4 flex-row">


          <!-- Map, chart buttons, and year scroll -->
          <div class="flex flex-col rounded-[10px] bg-white shadow-md w-[500px] h-[500px] relative">

            <!-- Chart Primary Category Buttons -->
            <div class="flex items-center justify-between w-full">
              <p class="text-[0.8rem] ml-2">Region: <strong>{{ selectRegion }}</strong></p>
              <div class="flex items-center space-x-1 justify-end p-2">
                <button v-for="(data, key) in availableChartCategories" :key="key"
                  @click="selectChartCategory = data"
                  class="bg-[#e8eaed] text-[#1f1f1f] text-[0.8rem] px-1 py-1 rounded"
                  :class="{'bg-sky-500 text-white': selectChartCategory === data}">
                  {{ data }}
                </button>
              </div>
            </div>

            <!-- Horizontal Divider -->
            <hr class="border-gray-300 z-[100]">

            <!-- Ranking Subcategory Buttons -->
            <div class="flex items-center space-x-1 justify-end absolute top-[55px] left-[220px] z-[100]">
              <button v-for="(data, key) in availableRankSubcategories" :key="key"
                @click="selectRankingSubCategory = data"
                class="bg-[#e8eaed] text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
                :class="{'bg-sky-500 text-white': selectRankingSubCategory === data}">
                {{ RankSubcategoriesRename[data] || data }}
              </button>
            </div>

            <!-- Year scroll -->
            <div class="flex flex-col absolute top-[50px] left-[10px] w-[200px] z-[100]">
              <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
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

            <!-- Map -->
            <map-geojson 
              class="absolute top-[50px] left-0 w-full z-[10]"
              :height="'430px'"
              :selectRankingColors="selectRankingColors">
            </map-geojson>

            <!-- Legend -->
            <div v-if="colorsRanking" class="absolute bottom-[20px] left-[35px] z-[100]">
              <div class="font-bold text-sm mb-2 text-gray-600">Ranking</div>
              <div class="flex flex-row items-center">
                <div v-for="(color, label) in colorsRanking" :key="label" class="flex items-center mr-4 mb-1">
                    <span class="inline-block w-[12px] h-[12px] mr-[3px]" :style="{ backgroundColor: color }"></span>
                    <span class="text-sm text-gray-600">{{ label }}</span>
                </div>
              </div>
            </div>

          </div>


          <div class="relative flex flex-1 rounded-[10px] bg-white shadow-md h-[500px]">

            <!-- Chart subcategory buttons -->
            <div class="absolute flex flex-row space-x-1 mr-4 top-[9px] left-[10px] z-10">
              <button v-for="cat in availableChartSubCategories" :key="cat"
                @click="selectChartSubCategory = cat"
                class="bg-[#e8eaed] text-[#1f1f1f] text-[0.8rem] px-1 py-1 rounded"
                :class="{'bg-sky-500 text-white': selectChartSubCategory === cat}">
                {{ cat }}
              </button>
            </div>

            <chart-container
              class="w-full h-full pt-[50px]"
              :chartData="selectChartData">
            </chart-container>

          </div>


        </div>

        
      </div>
    </div>
  `,
};