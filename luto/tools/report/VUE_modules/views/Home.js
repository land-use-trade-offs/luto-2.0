window.HomeView = {
  name: 'HomeView',

  setup() {

    const { ref, onMounted, onUnmounted, watch, computed, inject, nextTick } = Vue;

    // Data service
    const chartRegister = window.ChartService.chartCategories;   // ChartService has been registered in index.html      [ChartService.js]
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Home";

    // Global variables
    const selectRegion = inject('globalSelectedRegion');
    const ChartData = ref({});

    // Navigate to region-level data handling different structures across categories.
    // After the region_level change, data is wrapped as: { region_NRM: { region: ... }, region_state: ... }
    // Water uses 'NRM' as its level key; BIO Quality has an extra 'Suitability' backend wrapper;
    // Production overview still uses direct region keys (national only).
    function getChartRegionData(cat, sub, region) {
      const dataObj = ChartData.value?.[cat]?.[sub];
      if (!dataObj) return undefined;
      if (cat === 'Biodiversity' && sub === 'Quality') {
        return dataObj?.['Suitability']?.['region_NRM']?.[region];
      }
      if (cat === 'Water') {
        return dataObj?.['NRM']?.[region];
      }
      if (cat === 'Production') {
        // Production is reported nationally only (the demand rows are national)
        return dataObj?.['AUSTRALIA'];
      }
      return dataObj?.['region_NRM']?.[region];
    }

    function getRankingLevel(cat) {
      return cat === 'Water' ? 'NRM' : 'region_NRM';
    }

    // Species selection state (for GBF3/4/8 charts that have region → species → [series] structure)
    const selectSpecies = ref('');
    const availableSpecies = computed(() => {
      if (!dataLoaded.value) return [];
      const data = getChartRegionData(selectChartCategory.value, selectChartSubCategory.value, selectRegion.value);
      if (!data || Array.isArray(data)) return [];
      return Object.keys(data);
    });
    const hasSpeciesData = computed(() => availableSpecies.value.length > 0);

    // Transition heatmap state
    const transitionData = ref(null);
    const transitionFromWater = ref('ALL');
    const transitionToWater = ref('ALL');
    const WATER_OPTIONS = ['ALL', 'Dryland', 'Irrigated'];
    const rankingData = ref({});
    const rankingColors = ref({});
    const runScenario = ref({});
    const dataLoaded = ref(false);
    const lazyBioEntries = {};               // GBF tab -> ChartService entry, fetched on first pick
    const isLoadingChart = ref(false);

    // Available selections
    const availableYears = ref([]);
    const availableChartCategories = ref([]);
    const availableChartSubCategories = ref([]);
    const availableUnit = window.ReportLabels.units;
    const RankSubcategoriesRename = {
      'Agricultural Land-use': 'Ag',
      'Agricultural Management': 'Ag Mgt',
      'Non-Agricultural Land-use': 'Non-Ag',
    };
    const rankSubcategoryOrder = ['Agricultural Land-use', 'Agricultural Management', 'Non-Agricultural Land-use'];
    const availableRankSubcategories = ref([]);

    // Default selections
    const yearIndex = ref(0);
    const selectYear = ref(null);
    const selectChartCategory = ref('');
    const selectChartSubCategory = ref('');
    const selectRankingSubCategory = ref('');

    // Computed leaf for the transition heatmap
    const transitionHeatmapLeaf = computed(() => {
      if (!transitionData.value) return null;
      const region = selectRegion.value || 'AUSTRALIA';
      // Fall back to AUSTRALIA if the region doesn't exist in the data
      const regionData = transitionData.value[region] || transitionData.value['AUSTRALIA'];
      if (!regionData) return null;
      const fromData = regionData[transitionFromWater.value];
      if (!fromData) return null;
      return fromData[transitionToWater.value] || null;
    });

    const transitionFormattedMax = computed(() => {
      const v = transitionHeatmapLeaf.value?.max_val;
      if (!v && v !== 0) return '0';
      return v >= 1e6 ? (v / 1e6).toFixed(1) + 'M'
        : v >= 1e3 ? (v / 1e3).toFixed(0) + 'k'
          : String(Math.round(v));
    });

    // 'SSP245' -> 'SSP2-4.5'
    const scenarioLabel = computed(() => {
      const ssp = String(runScenario.value.SSP || '');
      return ssp.length === 3 ? `SSP${ssp[0]}-${ssp[1]}.${ssp[2]}` : `SSP${ssp}`;
    });

    // "<quantity> — <view>, <region>"; Production is national only
    const chartHeading = computed(() => {
      const cat = selectChartCategory.value;
      const unit = availableUnit[cat] || cat;
      const quantity = cat === 'Biodiversity' ? 'Biodiversity' : unit.replace(/\s*\(.*\)\s*$/, '');
      const region = cat === 'Production' && selectRegion.value !== 'AUSTRALIA'
        ? 'AUSTRALIA (reported nationally only)'
        : selectRegion.value;
      return `${quantity} — ${selectChartSubCategory.value}, ${region}`;
    });

    //  Reactive data
    const selectChartData = computed(() => {
      let seriesData, yAxisTitle = null;
      let originalData = getChartRegionData(selectChartCategory.value, selectChartSubCategory.value, selectRegion.value) || [];
      // If species-level data (region → species → [series]), pick the selected species
      if (hasSpeciesData.value && selectSpecies.value) {
        originalData = originalData[selectSpecies.value] || [];
      }
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
        legend: { ...window['Chart_default_options'].legend },
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
      const r = selectRegion.value;
      const y = selectYear.value;
      const rnm = 'region_NRM';
      const E = rankingData.value['Economics']?.[rnm]?.[r];
      const A = rankingData.value['Area']?.[rnm]?.[r];
      const G = rankingData.value['GHG']?.[rnm]?.[r];
      const W = rankingData.value['Water']?.['NRM']?.[r];
      const B = rankingData.value['Biodiversity (Quality)']?.[rnm]?.[r];
      return {
        economicTotal: E?.['Profit']?.['value']?.[y] || 'N/A',
        economicCost: E?.['Cost']?.['value']?.[y] || 'N/A',
        economicRevenue: E?.['Revenue']?.['value']?.[y] || 'N/A',
        areaTotal: A?.['Total']?.['value']?.[y] || 'N/A',
        areaAgLand: A?.['Agricultural Land-use']?.['value']?.[y] || 'N/A',
        areaAgMgt: A?.['Agricultural Management']?.['value']?.[y] || 'N/A',
        areaNonAg: A?.['Non-Agricultural Land-use']?.['value']?.[y] || 'N/A',
        ghgTotal: G?.['Total']?.['value']?.[y] || 'N/A',
        ghgEmissions: G?.['GHG emissions']?.['value']?.[y] || 'N/A',
        ghgReduction: G?.['GHG sequestrations']?.['value']?.[y] || 'N/A',
        waterTotal: W?.['Total']?.['value']?.[y] || 'N/A',
        waterAgLand: W?.['Agricultural Land-use']?.['value']?.[y] || 'N/A',
        waterAgMgt: W?.['Agricultural Management']?.['value']?.[y] || 'N/A',
        waterNonAg: W?.['Non-Agricultural Land-use']?.['value']?.[y] || 'N/A',
        biodiversityTotal: B?.['Total']?.['value']?.[y] || 'N/A',
        biodiversityAgLand: B?.['Agricultural Land-use']?.['value']?.[y] || 'N/A',
        biodiversityAgMgt: B?.['Agricultural Management']?.['value']?.[y] || 'N/A',
        biodiversityNonAg: B?.['Non-Agricultural Land-use']?.['value']?.[y] || 'N/A'
      };
    });

    const selectRankingColors = computed(() => {
      if (!dataLoaded.value) { return {} }
      const rl = getRankingLevel(selectChartCategory.value);
      return Object.fromEntries(
        Object.entries(rankingData.value[selectChartCategory.value]?.[rl] || {}).map(([region, values]) => [
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
      const chartOverview_bio_GBF4_SNES = chartRegister['Biodiversity']['GBF4_SNES']['overview']['sum'];
      const chartOverview_bio_GBF4_ECNES = chartRegister['Biodiversity']['GBF4_ECNES']['overview']['sum'];
      const chartOverview_bio_GBF8_SPECIES = chartRegister['Biodiversity']['GBF8_SPECIES']['overview']['sum'];
      const chartOverview_bio_GBF8_GROUP = chartRegister['Biodiversity']['GBF8_GROUP']['overview']['sum'];
      const chartOverview_economics_sum = chartRegister['Economics']['overview']['sum'];
      const chartOverview_economics_ag = chartRegister['Economics']['overview']['Ag'];
      const chartOverview_economics_agMgt = chartRegister['Economics']['overview']['Ag Mgt'];
      const chartOverview_economics_Nonag = chartRegister['Economics']['Non-Ag']['Profit'];
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

      // Every overview and ranking file is independent: fetch them in parallel. The GBF overviews
      // (the SNES one alone can be > 100 MB) are fetched only when their tab is picked.
      const eagerEntries = [
        chartOverview_area_source, chartOverview_area_category, chartOverview_area_landuse,
        chartOverview_bio_quality,
        chartOverview_economics_sum, chartOverview_economics_ag, chartOverview_economics_agMgt, chartOverview_economics_Nonag,
        chartOverview_ghg_sum, chartOverview_ghg_ag, chartOverview_ghg_agMgt, chartOverview_ghg_Nonag,
        chartOverview_prod_achieve, chartOverview_prod_overview, chartOverview_prod_domestic,
        chartOverview_prod_export, chartOverview_prod_import, chartOverview_prod_feed,
        chartOverview_water_sum, chartOverview_water_ag, chartOverview_water_agMgt, chartOverview_water_Nonag,
        rankingArea, rankingEconomics, rankingGHG, rankingProduction, rankingWater,
        rankingBiodiversityQuality, rankingBiodiversityAll,
      ];
      await Promise.all(eagerEntries.map(entry => loadScript(entry['path'], entry['name'], VIEW_NAME)));

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
          'Overview': window[chartOverview_prod_overview['name']],
          'Domestic': window[chartOverview_prod_domestic['name']],
          'Exports': window[chartOverview_prod_export['name']],
          'Imports': window[chartOverview_prod_import['name']],
          'Feed': window[chartOverview_prod_feed['name']],
          'Off-target achievement': window[chartOverview_prod_achieve['name']],
        },
        'Water': {
          'Overview': window[chartOverview_water_sum['name']],
          'Ag': window[chartOverview_water_ag['name']],
          'Ag Mgt': window[chartOverview_water_agMgt['name']],
          'Non-Ag': window[chartOverview_water_Nonag['name']],
        },
      };

      // GBF tabs: listed now, their data fetched on first pick (lazyBioEntries, see the watcher)
      const gbfTabs = [
        ['GBF2', 'GBF2_TARGET', chartOverview_bio_GBF2],
        ['GBF3 (NVIS)', 'WRITE_GBF3_NVIS', chartOverview_bio_GBF3_NVIS],
        ['GBF4 (SNES)', 'WRITE_GBF4_SNES', chartOverview_bio_GBF4_SNES],
        ['GBF4 (ECNES)', 'WRITE_GBF4_ECNES', chartOverview_bio_GBF4_ECNES],
        ['GBF8 (SPECIES)', 'GBF8_TARGET', chartOverview_bio_GBF8_SPECIES],
        ['GBF8 (GROUP)', 'GBF8_TARGET', chartOverview_bio_GBF8_GROUP],
      ];
      for (const [tab, settingKey, entry] of gbfTabs) {
        if (runScenario.value[settingKey] === 'off') continue;
        ChartData.value['Biodiversity'][tab] = null;
        lazyBioEntries[tab] = entry;
      }

      //  Set initial values
      availableYears.value = window['Supporting_info']['years'];
      selectYear.value = availableYears.value[0];

      availableChartCategories.value = Object.keys(ChartData.value);
      selectChartCategory.value = availableChartCategories.value[0];

      selectChartSubCategory.value = Object.keys(ChartData.value[selectChartCategory.value])[0];
      const rl0 = getRankingLevel(selectChartCategory.value);
      const rankingKeys = Object.keys(rankingData.value?.[selectChartCategory.value]?.[rl0]?.[selectRegion.value] || {}).filter(key => key !== "Total");
      selectRankingSubCategory.value = rankingKeys[0] || 'N/A';
      rankingColors.value = window.Supporting_info.colors_ranking;

      // Load transition start-end heatmap data
      const transitionStartEnd = chartRegister['Transition']['Area']['start_end'];
      await loadScript(transitionStartEnd['path'], transitionStartEnd['name'], VIEW_NAME);
      transitionData.value = window[transitionStartEnd['name']];

      await nextTick(() => { dataLoaded.value = true; });

    });

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    watch(selectChartCategory, (newCategory) => {
      availableChartSubCategories.value = Object.keys(ChartData.value[selectChartCategory.value])
      const rl = getRankingLevel(newCategory);
      const rankKeys = Object.keys(rankingData.value?.[newCategory]?.[rl]?.[selectRegion.value] || {}).filter(key => key !== "Total");
      availableRankSubcategories.value = rankKeys.sort((a, b) => rankSubcategoryOrder.indexOf(a) - rankSubcategoryOrder.indexOf(b));
      selectChartSubCategory.value = availableChartSubCategories.value[0];
      selectRankingSubCategory.value = availableRankSubcategories.value[0] || 'N/A';
    });

    // Fetch a GBF overview the first time its tab is picked
    watch(selectChartSubCategory, async (sub) => {
      if (selectChartCategory.value !== 'Biodiversity') return;
      const entry = lazyBioEntries[sub];
      if (!entry || ChartData.value['Biodiversity'][sub]) return;
      isLoadingChart.value = true;
      try {
        await loadScript(entry['path'], entry['name'], VIEW_NAME);
        ChartData.value['Biodiversity'][sub] = window[entry['name']];
      } finally {
        isLoadingChart.value = false;
      }
    });

    // Reset species selection when subcategory or category changes
    watch(availableSpecies, (newSpecies) => {
      if (!newSpecies.includes(selectSpecies.value)) {
        selectSpecies.value = newSpecies[0] || '';
      }
    });

    // Memory cleanup on component unmount
    onUnmounted(() => { window.MemoryService.cleanupViewData(VIEW_NAME); });

    const _state = {
      yearIndex,
      runScenario,
      scenarioLabel,
      chartHeading,
      isLoadingChart,
      dataLoaded,

      ChartData,
      rankingData,
      RankSubcategoriesRename,
      rankingColors,

      availableYears,
      availableChartCategories,
      availableChartSubCategories,
      availableRankSubcategories,
      availableSpecies,

      selectYear,
      selectRegion,
      selectChartCategory,
      selectChartSubCategory,
      selectRankingSubCategory,
      selectSpecies,
      selectChartData,
      selectRankingColors,
      selectRanking,
      hasSpeciesData,

      // Transition heatmap
      transitionData,
      transitionFromWater,
      transitionToWater,
      transitionHeatmapLeaf,
      transitionFormattedMax,
      WATER_OPTIONS,
    };
    window._debug[VIEW_NAME] = _state;
    return _state;
  },

  // This template is a fallback that will be replaced by the loaded template
  template: /*html*/`
    <div v-if="dataLoaded">
      <div class="flex flex-col">

        <!-- Scenario Information Header -->
        <p class="text-[#505051] font-bold p-1 pt-8">
          Scenario: {{ scenarioLabel }} ·
          GHG target: {{ runScenario.GHG_EMISSIONS_LIMITS }} ·
          GBF2 target: {{ runScenario.GBF2_TARGET }}
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
            {{ chartHeading }}
            <span v-if="isLoadingChart" class="ml-2 text-[0.75rem] font-normal text-gray-500">loading…</span>
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
                class="bg-[#e8eaed] text-[#1f1f1f] text-[0.7rem] px-2 py-1 rounded"
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
                :min="0"
                :max="availableYears.length - 1"
                :step="1"
                :format-tooltip="index => availableYears[index]"
                :show-stops="true"
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
                NRM rank by {{ RankSubcategoriesRename[selectRankingSubCategory] || selectRankingSubCategory }}, {{ selectYear }}
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

            <!-- Species Selector (for GBF3/4/8 charts with region → species → [series] data) -->
            <div v-if="hasSpeciesData" class="absolute top-[0px] right-[0px] w-[270px] z-10">
              <filterable-dropdown
                :items="availableSpecies"
                :selectedValue="selectSpecies"
                :useSearch="true"
                placeholder="Select species..."
                searchPlaceholder="Search species..."
                @change="selectSpecies = $event"
              />
            </div>

            <!-- Chart Component -->
            <chart-container
              class="w-full h-full pt-[50px]"
              :chartData="selectChartData"
            />
          </div>
        </div>

        <!-- ── Transition Heatmap Section ─────────────────────────────── -->
        <div class="flex flex-col mb-8 mr-4 mt-8">

          <p class="text-[#505051] font-bold p-1 mb-2">
            Land-use Transition Area: {{ selectRegion }} (2010 → {{ availableYears[availableYears.length - 1] }})
          </p>

          <!-- Card: controls overlaid at top-left, heatmap fills the card -->
          <div class="relative rounded-[10px] bg-white shadow-md overflow-x-auto">

            <!-- Controls overlay top-left -->
            <div class="absolute top-2 left-3 z-10 flex items-center gap-4 bg-white/80 rounded px-2 py-1">
              <!-- From-Water buttons -->
              <div class="flex items-center gap-1">
                <span class="text-[0.75rem] text-[#505051] mr-1">From:</span>
                <button
                  v-for="opt in WATER_OPTIONS" :key="'fw-' + opt"
                  @click="transitionFromWater = opt"
                  class="bg-[#e8eaed] text-[#1f1f1f] text-[0.7rem] px-2 py-1 rounded"
                  :class="{'bg-sky-500 text-white': transitionFromWater === opt}"
                >{{ opt }}</button>
              </div>
              <!-- To-Water buttons -->
              <div class="flex items-center gap-1">
                <span class="text-[0.75rem] text-[#505051] mr-1">To:</span>
                <button
                  v-for="opt in WATER_OPTIONS" :key="'tw-' + opt"
                  @click="transitionToWater = opt"
                  class="bg-[#e8eaed] text-[#1f1f1f] text-[0.7rem] px-2 py-1 rounded"
                  :class="{'bg-sky-500 text-white': transitionToWater === opt}"
                >{{ opt }}</button>
              </div>
              <!-- Color legend -->
              <div v-if="transitionHeatmapLeaf" class="flex items-center gap-1 ml-2">
                <span style="font-size:0.72rem;color:#666;">0</span>
                <div style="width:120px;height:10px;border-radius:3px;border:1px solid #ddd;background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026);"></div>
                <span style="font-size:0.72rem;color:#666;">{{ transitionFormattedMax }} ha</span>
              </div>
            </div>

            <!-- Heatmap or placeholder -->
            <heatmap-container
              v-if="transitionHeatmapLeaf"
              :xCats="transitionHeatmapLeaf.x_categories"
              :yCats="transitionHeatmapLeaf.y_categories"
              :data="transitionHeatmapLeaf.data"
              :maxVal="transitionHeatmapLeaf.max_val"
              :exportable="true"
            />
            <div
              v-else
              class="flex items-center justify-center h-[200px] text-gray-400 text-sm"
            >
              No transition data for the selected combination.
            </div>
          </div>
        </div>

      </div>
    </div>
  `,
};