window.HomeView = {

  setup() {

    const { ref, onMounted, onUnmounted, watch, computed, inject, nextTick } = Vue;

    // Data service
    const chartRegister = window.ChartService.chartCategories;   // ChartService has been registered in index.html      [ChartService.js]
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Home";

    // Global variables
    const selectRegion = inject('globalSelectedRegion');
    const ChartData = ref({});
    const rankingData = ref({});
    const rankingColors = ref({});
    const runScenario = ref({});
    const dataLoaded = ref(false);

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
      'Production': 'Tonnes or KL',
    };
    const RankSubcategoriesRename = {
      'Agricultural Landuse': 'Ag',
      'Agricultural Management': 'Ag Mgt',
      'Non-Agricultural Land-use': 'Non-Ag',
    };
    const availableRankSubcategories = ref([]);

    // Default selections
    const yearIndex = ref(0);
    const selectYear = ref(null);
    const selectChartCategory = ref('');
    const selectChartSubCategory = ref('');
    const selectRankingSubCategory = ref('');

    //  Reactive data
    const selectChartData = computed(() => {

      let seriesData, yAxisTitle = null;
      const originalData = ChartData.value?.[selectChartCategory.value]?.[selectChartSubCategory.value]?.[selectRegion.value] || [];
      seriesData = JSON.parse(JSON.stringify(originalData));

      if (selectChartSubCategory.value === 'Off-target achievement') {
        yAxisTitle = 'Achievement (%)';
      } else if (['GBF3 (NVIS)', 'GBF3 (IBRA)', 'GBF4 (SNES)', 'GBF4 (ECNES)', 'GBF8 (SPECIES)', 'GBF8 (GROUP)'].includes(selectChartSubCategory.value)) {
        yAxisTitle = 'Priority Weighted Hectares';
      }

      const seriesColors = seriesData.map(serie => serie.color).filter(color => color) || [];
      const colors = seriesColors.length > 0 ? seriesColors : window['Supporting_info']?.colors || [];
      return {
        ...window['Chart_default_options'],
        chart: {
          height: 440,
        },
        yAxis: {
          title: {
            text: yAxisTitle || availableUnit[selectChartCategory.value]
          }
        },
        series: seriesData,
        colors: colors,
      };
    });


    const selectRanking = computed(() => {
      return {
        economicTotal: rankingData.value['Economics']?.[selectRegion.value]?.['Total']?.['value']?.[selectYear.value] || 'N/A',
        economicCost: rankingData.value['Economics']?.[selectRegion.value]?.['Cost']?.['value']?.[selectYear.value] || 'N/A',
        economicRevenue: rankingData.value['Economics']?.[selectRegion.value]?.['Revenue']?.['value']?.[selectYear.value] || 'N/A',
        areaTotal: rankingData.value['Area']?.[selectRegion.value]?.['Total']?.['value']?.[selectYear.value] || 'N/A',
        areaAgLand: rankingData.value['Area']?.[selectRegion.value]?.['Agricultural Landuse']?.['value']?.[selectYear.value] || 'N/A',
        areaAgMgt: rankingData.value['Area']?.[selectRegion.value]?.['Agricultural Management']?.['value']?.[selectYear.value] || 'N/A',
        areaNonAg: rankingData.value['Area']?.[selectRegion.value]?.['Non-Agricultural Land-use']?.['value']?.[selectYear.value] || 'N/A',
        ghgTotal: rankingData.value['GHG']?.[selectRegion.value]?.['Total']?.['value']?.[selectYear.value] || 'N/A',
        ghgEmissions: rankingData.value['GHG']?.[selectRegion.value]?.['GHG emissions']?.['value']?.[selectYear.value] || 'N/A',
        ghgReduction: rankingData.value['GHG']?.[selectRegion.value]?.['GHG sequestrations']?.['value']?.[selectYear.value] || 'N/A',
        waterTotal: rankingData.value['Water']?.[selectRegion.value]?.['Total']?.['value']?.[selectYear.value] || 'N/A',
        waterAgLand: rankingData.value['Water']?.[selectRegion.value]?.['Agricultural Landuse']?.['value']?.[selectYear.value] || 'N/A',
        waterAgMgt: rankingData.value['Water']?.[selectRegion.value]?.['Agricultural Management']?.['value']?.[selectYear.value] || 'N/A',
        waterNonAg: rankingData.value['Water']?.[selectRegion.value]?.['Non-Agricultural Land-use']?.['value']?.[selectYear.value] || 'N/A',
        biodiversityTotal: rankingData.value['Biodiversity (Quality)']?.[selectRegion.value]?.['Total']?.['value']?.[selectYear.value] || 'N/A',
        biodiversityAgLand: rankingData.value['Biodiversity (Quality)']?.[selectRegion.value]?.['Agricultural Landuse']?.['value']?.[selectYear.value] || 'N/A',
        biodiversityAgMgt: rankingData.value['Biodiversity (Quality)']?.[selectRegion.value]?.['Agricultural Management']?.['value']?.[selectYear.value] || 'N/A',
        biodiversityNonAg: rankingData.value['Biodiversity (Quality)']?.[selectRegion.value]?.['Non-Agricultural Land-use']?.['value']?.[selectYear.value] || 'N/A'
      };
    });

    const selectRankingColors = computed(() => {
      if (!dataLoaded.value) { return {} }
      return Object.fromEntries(
        Object.entries(rankingData.value[selectChartCategory.value]).map(([region, values]) => [
          region,
          values?.[selectRankingSubCategory.value]?.['color']?.[selectYear.value] || {}
        ])
      );
    });


    onMounted(async () => {

      // Load required data
      await loadScript("./data/Supporting_info.js", 'Supporting_info', VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options', VIEW_NAME);
      await loadScript("./data/geo/NRM_AUS.js", 'NRM_AUS', VIEW_NAME);

      runScenario.value = Object.fromEntries(window['Supporting_info']['model_run_settings'].map(item => [item.parameter, item.val]));

      const chartOverview_area_source = chartRegister['Area']['overview']['Source'];
      const chartOverview_area_category = chartRegister['Area']['overview']['Category'];
      const chartOverview_area_landuse = chartRegister['Area']['overview']['Land-use'];
      const chartOverview_bio_quality = chartRegister['Biodiversity']['quality']['overview']['sum'];
      const chartOverview_bio_GBF2 = chartRegister['Biodiversity']['GBF2']['overview']['sum'];
      const chartOverview_bio_GBF3_NVIS = chartRegister['Biodiversity']['GBF3_NVIS']['overview']['sum'];
      const chartOverview_bio_GBF3_IBRA = chartRegister['Biodiversity']['GBF3_IBRA']['overview']['sum'];
      const chartOverview_bio_GBF4_SNES = chartRegister['Biodiversity']['GBF4_SNES']['overview']['sum'];
      const chartOverview_bio_GBF4_ECNES = chartRegister['Biodiversity']['GBF4_ECNES']['overview']['sum'];
      const chartOverview_bio_GBF8_SPECIES = chartRegister['Biodiversity']['GBF8_SPECIES']['overview']['sum'];
      const chartOverview_bio_GBF8_GROUP = chartRegister['Biodiversity']['GBF8_GROUP']['overview']['sum'];
      const chartOverview_economics_sum = chartRegister['Economics']['overview']['sum'];
      const chartOverview_economics_ag = chartRegister['Economics']['overview']['Ag'];
      const chartOverview_economics_agMgt = chartRegister['Economics']['overview']['Ag Mgt'];
      const chartOverview_economics_Nonag = chartRegister['Economics']['overview']['Non-Ag'];
      const chartOverview_ghg_sum = chartRegister['GHG']['overview']['sum'];
      const chartOverview_ghg_ag = chartRegister['GHG']['overview']['Ag'];
      const chartOverview_ghg_agMgt = chartRegister['GHG']['overview']['Ag Mgt'];
      const chartOverview_ghg_Nonag = chartRegister['GHG']['overview']['Non-Ag'];
      const chartOverview_prod_achieve = chartRegister['Production']['overview']['achieve'];
      const chartOverview_prod_overview = chartRegister['Production']['overview']['sum'];
      const chartOverview_prod_domestic = chartRegister['Production']['overview']['Domestic'];
      const chartOverview_prod_export = chartRegister['Production']['overview']['Exports'];
      const chartOverview_prod_import = chartRegister['Production']['overview']['Imports'];
      const chartOverview_prod_feed = chartRegister['Production']['overview']['Feed'];
      const chartOverview_water_sum = chartRegister['Water']['NRM']['overview']['sum'];
      const chartOverview_water_ag = chartRegister['Water']['NRM']['overview']['Ag'];
      const chartOverview_water_agMgt = chartRegister['Water']['NRM']['overview']['Ag Mgt'];
      const chartOverview_water_Nonag = chartRegister['Water']['NRM']['overview']['Non-Ag'];

      const rankingArea = chartRegister['Area']['ranking'];
      const rankingEconomics = chartRegister['Economics']['ranking'];
      const rankingGHG = chartRegister['GHG']['ranking'];
      const rankingProduction = chartRegister['Production']['ranking'];
      const rankingWater = chartRegister['Water']['NRM']['ranking'];
      const rankingBiodiversityQuality = chartRegister['Biodiversity']['quality']['ranking'];
      const rankingBiodiversityAll = chartRegister['Biodiversity']['ranking'];

      await loadScript(chartOverview_area_source['path'], chartOverview_area_source['name'], VIEW_NAME);
      await loadScript(chartOverview_area_category['path'], chartOverview_area_category['name'], VIEW_NAME);
      await loadScript(chartOverview_area_landuse['path'], chartOverview_area_landuse['name'], VIEW_NAME);
      await loadScript(chartOverview_bio_quality['path'], chartOverview_bio_quality['name'], VIEW_NAME);
      // Conditional biodiversity script loading based on runScenario
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_2'] !== 'off') {
        await loadScript(chartOverview_bio_GBF2['path'], chartOverview_bio_GBF2['name'], VIEW_NAME);
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_3_NVIS'] !== 'off') {
        await loadScript(chartOverview_bio_GBF3_NVIS['path'], chartOverview_bio_GBF3_NVIS['name'], VIEW_NAME);
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_3_IBRA'] !== 'off') {
        await loadScript(chartOverview_bio_GBF3_IBRA['path'], chartOverview_bio_GBF3_IBRA['name'], VIEW_NAME);
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_4_SNES'] !== 'off') {
        await loadScript(chartOverview_bio_GBF4_SNES['path'], chartOverview_bio_GBF4_SNES['name'], VIEW_NAME);
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_4_ECNES'] !== 'off') {
        await loadScript(chartOverview_bio_GBF4_ECNES['path'], chartOverview_bio_GBF4_ECNES['name'], VIEW_NAME);
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_8'] !== 'off') {
        await loadScript(chartOverview_bio_GBF8_SPECIES['path'], chartOverview_bio_GBF8_SPECIES['name'], VIEW_NAME);
        await loadScript(chartOverview_bio_GBF8_GROUP['path'], chartOverview_bio_GBF8_GROUP['name'], VIEW_NAME);
      }
      await loadScript(chartOverview_economics_sum['path'], chartOverview_economics_sum['name'], VIEW_NAME);
      await loadScript(chartOverview_economics_ag['path'], chartOverview_economics_ag['name'], VIEW_NAME);
      await loadScript(chartOverview_economics_agMgt['path'], chartOverview_economics_agMgt['name'], VIEW_NAME);
      await loadScript(chartOverview_economics_Nonag['path'], chartOverview_economics_Nonag['name'], VIEW_NAME);
      await loadScript(chartOverview_ghg_sum['path'], chartOverview_ghg_sum['name'], VIEW_NAME);
      await loadScript(chartOverview_ghg_ag['path'], chartOverview_ghg_ag['name'], VIEW_NAME);
      await loadScript(chartOverview_ghg_agMgt['path'], chartOverview_ghg_agMgt['name'], VIEW_NAME);
      await loadScript(chartOverview_ghg_Nonag['path'], chartOverview_ghg_Nonag['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_achieve['path'], chartOverview_prod_achieve['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_overview['path'], chartOverview_prod_overview['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_domestic['path'], chartOverview_prod_domestic['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_export['path'], chartOverview_prod_export['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_import['path'], chartOverview_prod_import['name'], VIEW_NAME);
      await loadScript(chartOverview_prod_feed['path'], chartOverview_prod_feed['name'], VIEW_NAME);
      await loadScript(chartOverview_water_sum['path'], chartOverview_water_sum['name'], VIEW_NAME);
      await loadScript(chartOverview_water_ag['path'], chartOverview_water_ag['name'], VIEW_NAME);
      await loadScript(chartOverview_water_agMgt['path'], chartOverview_water_agMgt['name'], VIEW_NAME);
      await loadScript(chartOverview_water_Nonag['path'], chartOverview_water_Nonag['name'], VIEW_NAME);

      await loadScript(rankingArea['path'], rankingArea['name'], VIEW_NAME);
      await loadScript(rankingEconomics['path'], rankingEconomics['name'], VIEW_NAME);
      await loadScript(rankingGHG['path'], rankingGHG['name'], VIEW_NAME);
      await loadScript(rankingProduction['path'], rankingProduction['name'], VIEW_NAME);
      await loadScript(rankingWater['path'], rankingWater['name'], VIEW_NAME);
      await loadScript(rankingBiodiversityQuality['path'], rankingBiodiversityQuality['name'], VIEW_NAME);
      await loadScript(rankingBiodiversityAll['path'], rankingBiodiversityAll['name'], VIEW_NAME);



      rankingData.value = {
        'Area': window[rankingArea['name']],
        'Economics': window[rankingEconomics['name']],
        'GHG': window[rankingGHG['name']],
        'Production': window[rankingProduction['name']],
        'Water': window[rankingWater['name']],
        'Biodiversity (Quality)': window[rankingBiodiversityQuality['name']],
        'Biodiversity': window[rankingBiodiversityAll['name']],
      };



      // Create base ChartData structure without GBF data
      ChartData.value = {
        'Area': {
          'Overview': window[chartOverview_area_source['name']],
          'Category': window[chartOverview_area_category['name']],
          'Land-use': window[chartOverview_area_landuse['name']],
        },
        'Biodiversity': {
          'Quality': window[chartOverview_bio_quality['name']],
        },
        'Economics': {
          'Overview': window[chartOverview_economics_sum['name']],
          'Ag': window[chartOverview_economics_ag['name']],
          'Ag Mgt': window[chartOverview_economics_agMgt['name']],
          'Non-Ag': window[chartOverview_economics_Nonag['name']],
        },
        'GHG': {
          'Overview': window[chartOverview_ghg_sum['name']],
          'Ag': window[chartOverview_ghg_ag['name']],
          'Ag Mgt': window[chartOverview_ghg_agMgt['name']],
          'Non-Ag': window[chartOverview_ghg_Nonag['name']],
        },
        'Production': {
          'Off-target achievement': window[chartOverview_prod_achieve['name']],
          'Overview': window[chartOverview_prod_overview['name']],
          'Domestic': window[chartOverview_prod_domestic['name']],
          'Exports': window[chartOverview_prod_export['name']],
          'Imports': window[chartOverview_prod_import['name']],
          'Feed': window[chartOverview_prod_feed['name']],
        },
        'Water': {
          'Overview': window[chartOverview_water_sum['name']],
          'Ag': window[chartOverview_water_ag['name']],
          'Ag Mgt': window[chartOverview_water_agMgt['name']],
          'Non-Ag': window[chartOverview_water_Nonag['name']],
        },
      };

      // Dynamically add GBF data based on what was loaded
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_2'] !== 'off') {
        ChartData.value['Biodiversity']['GBF2'] = window[chartOverview_bio_GBF2['name']];
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_3_NVIS'] !== 'off') {
        ChartData.value['Biodiversity']['GBF3 (NVIS)'] = window[chartOverview_bio_GBF3_NVIS['name']];
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_3_IBRA'] !== 'off') {
        ChartData.value['Biodiversity']['GBF3 (IBRA)'] = window[chartOverview_bio_GBF3_IBRA['name']];
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_4_SNES'] !== 'off') {
        ChartData.value['Biodiversity']['GBF4 (SNES)'] = window[chartOverview_bio_GBF4_SNES['name']];
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_4_ECNES'] !== 'off') {
        ChartData.value['Biodiversity']['GBF4 (ECNES)'] = window[chartOverview_bio_GBF4_ECNES['name']];
      }
      if (runScenario.value['BIODIVERSITY_TARGET_GBF_8'] !== 'off') {
        ChartData.value['Biodiversity']['GBF8 (SPECIES)'] = window[chartOverview_bio_GBF8_SPECIES['name']];
        ChartData.value['Biodiversity']['GBF8 (GROUP)'] = window[chartOverview_bio_GBF8_GROUP['name']];
      }

      //  Set initial values
      availableYears.value = window['Supporting_info']['years'];
      selectYear.value = availableYears.value[0];

      availableChartCategories.value = Object.keys(ChartData.value);
      selectChartCategory.value = availableChartCategories.value[0];

      selectChartSubCategory.value = Object.keys(ChartData.value[selectChartCategory.value])[0];
      const rankingKeys = Object.keys(rankingData.value?.[selectChartCategory.value]?.[selectRegion.value] || {}).filter(key => key !== "Total");
      selectRankingSubCategory.value = rankingKeys[0] || 'N/A';
      rankingColors.value = window.Supporting_info.colors_ranking;

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

    // Memory cleanup on component unmount
    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    return {
      yearIndex,
      runScenario,
      dataLoaded,

      ChartData,
      rankingData,
      RankSubcategoriesRename,
      rankingColors,

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

        <!-- Scenario Information Header -->
        <p class="text-[#505051] font-bold p-1 pt-8">
          SSP - {{ runScenario.SSP }} |
          GHG - {{ runScenario.GHG_EMISSIONS_LIMITS }} |
          Biodiversity - {{ runScenario.BIODIVERSITY_TARGET_GBF_2 }}
        </p>

        <!-- Ranking Cards Section -->
        <div class="mb-4 mr-4">
          <ranking-cards :selectRankingData="selectRanking" />
        </div>

        <!-- Section Headers -->
        <div class="flex items-center justify-between">
          <p class="text-[#505051] w-[500px] font-bold p-1 pt-8">
            Map and Statistics
          </p>
          <p class="flex-1 text-[#505051] font-bold ml-4 p-1 pt-8">
            {{ selectChartCategory }} overview for {{ selectRegion }}
          </p>
        </div>

        <!-- Main Content Container -->
        <div class="flex mr-4 gap-4 mb-4 flex-row">

          <!-- Left Panel: Map Section -->
          <div class="flex flex-col rounded-[10px] bg-white shadow-md w-[500px] h-[500px] relative">

            <!-- Chart Primary Category Buttons -->
            <div class="flex items-center justify-between w-full">
              <!-- Region Display -->
              <p class="text-[0.8rem] ml-2">
                Region: <strong>{{ selectRegion }}</strong>
              </p>

              <!-- Chart Category Buttons -->
              <div class="flex items-center space-x-1 justify-end p-2">
                <button
                  v-for="(data, key) in availableChartCategories"
                  :key="key"
                  @click="selectChartCategory = data"
                  class="bg-[#e8eaed] text-[#1f1f1f] text-[0.7rem] px-1 py-1 rounded"
                  :class="{'bg-sky-500 text-white': selectChartCategory === data}"
                >
                  {{ data }}
                </button>
              </div>
            </div>

            <!-- Divider -->
            <hr class="border-gray-300 z-[100]">

            <!-- Ranking Subcategory Buttons (Absolute Positioned) -->
            <div class="flex items-center space-x-1 justify-end absolute top-[55px] left-[180px] z-[101]">
              <button
                v-for="(data, key) in availableRankSubcategories"
                :key="key"
                @click="selectRankingSubCategory = data"
                class="bg-[#e8eaed] text-[#1f1f1f] text-[0.57rem] px-1 py-1 rounded"
                :class="{'bg-sky-500 text-white': selectRankingSubCategory === data}"
              >
                {{ RankSubcategoriesRename[data] || data }}
              </button>
            </div>

            <!-- Year Slider (Absolute Positioned) -->
            <div class="flex flex-col absolute top-[50px] left-[10px] w-[200px] z-[100]">
              <p class="text-[0.8rem]">
                Year: <strong>{{ selectYear }}</strong>
              </p>

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

            <!-- Map Component -->
            <map-geojson
              class="absolute top-[50px] left-0 w-full z-[10]"
              :height="'430px'"
              :selectRankingColors="selectRankingColors"
            />

            <!-- Map Legend -->
            <div
              v-if="rankingColors"
              class="absolute bottom-[20px] left-[35px] z-[100]"
            >
              <div class="font-bold text-sm mb-2 text-gray-600">
                Ranking
              </div>
              <div class="flex flex-row items-center">
                <div
                  v-for="(color, label) in rankingColors"
                  :key="label"
                  class="flex items-center mr-4 mb-1"
                >
                  <span
                    class="inline-block w-[12px] h-[12px] mr-[3px]"
                    :style="{ backgroundColor: color }"
                  ></span>
                  <span class="text-sm text-gray-600">{{ label }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Right Panel: Chart Section -->
          <div class="relative flex flex-1 rounded-[10px] bg-white shadow-md h-[500px]">

            <!-- Chart Subcategory Buttons -->
            <div class="absolute flex flex-row space-x-1 mr-4 top-[9px] left-[10px] z-10">
              <button
                v-for="cat in availableChartSubCategories"
                :key="cat"
                @click="selectChartSubCategory = cat"
                class="bg-[#e8eaed] text-[#1f1f1f] text-[0.7rem] px-1 py-1 rounded"
                :class="{'bg-sky-500 text-white': selectChartSubCategory === cat}"
              >
                {{ cat }}
              </button>
            </div>

            <!-- Chart Component -->
            <chart-container
              class="w-full h-full pt-[50px]"
              :chartData="selectChartData"
            />
          </div>
        </div>
      </div>
    </div>
  `,
};