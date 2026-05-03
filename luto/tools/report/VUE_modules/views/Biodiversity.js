window.BiodiversityView = {
  name: 'BiodiversityView',
  setup() {
    const { ref, onMounted, onUnmounted, inject, computed, watch, nextTick } = Vue;

    // Data|Map service — top-level keys are metric names
    const chartRegister = window.ChartService.chartCategories["Biodiversity"];
    const mapRegister = window.MapService.mapCategories["Biodiversity"];
    const loadScript = window.loadScriptWithTracking;

    // View identification for memory management
    const VIEW_NAME = "Biodiversity";

    // Global selection state
    const yearIndex = ref(0);
    const selectYear = ref(2020);
    const selectRegion = inject("globalSelectedRegion");

    // Available variables
    const availableYears = ref([]);
    const availableUnit = {
      Biodiversity: "Relative Percentage (Pre-1750 = 100%)",
    };

    // Metric display labels (internal keys must match MapService/ChartService)
    const METRIC_LABELS = {
      'quality': 'Quality',
      'GBF2': 'GBF2',
      'GBF3_NVIS': 'GBF3 NVIS',
      'GBF4_SNES': 'GBF4 SNES',
      'GBF4_ECNES': 'GBF4 ECNES',
      'GBF8_GROUP': 'GBF8 Group',
      'GBF8_SPECIES': 'GBF8 Species',
    };

    // Metric → Supporting_info setting key (null = always available)
    const METRIC_TO_SETTING = {
      'quality': null,
      'GBF2': 'BIODIVERSITY_TARGET_GBF_2',
      'GBF3_NVIS': 'BIODIVERSITY_TARGET_GBF_3_NVIS',
      'GBF4_SNES': 'BIODIVERSITY_TARGET_GBF_4_SNES',
      'GBF4_ECNES': 'BIODIVERSITY_TARGET_GBF_4_ECNES',
      'GBF8_GROUP': 'BIODIVERSITY_TARGET_GBF_8',
      'GBF8_SPECIES': 'BIODIVERSITY_TARGET_GBF_8',
    };

    // Available selections
    const availableMetrics = ref(['quality']);
    const ALL_CATEGORIES = ["Sum", "Ag", "Ag Mgt", "Non-Ag"];
    // Per-metric available categories (some metrics like GBF3_NVIS / GBF4_* have no "Sum" layer)
    const availableCategories = computed(() => {
      const mr = mapRegister[selectMetric.value] || {};
      return ALL_CATEGORIES.filter(c => mr[c]);
    });
    const availableAgMgt = ref([]);
    const availableWater = ref([]);
    const availableSpecies = ref([]);
    const availableLanduse = ref([]);

    // Map selection state
    const selectMetric = ref("quality");
    const selectCategory = ref("");
    const selectAgMgt = ref("");
    const selectWater = ref("");
    const selectSpecies = ref("");
    const selectLanduse = ref("");

    // Metrics that have an extra Species/VegGroup dimension nested between
    // (water | agMgt+water | top) and landuse.
    //   GBF3 NVIS Ag:   water -> species -> landuse -> year
    //   GBF3 NVIS Am:   agMgt -> water -> species -> landuse -> year
    //   GBF3 NVIS NonAg: species -> landuse -> year
    // (GBF4 SNES / ECNES use the same shape.) GBF2 + quality have no species dim.
    const METRICS_WITH_SPECIES = ['GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8_GROUP', 'GBF8_SPECIES'];
    const hasSpecies = computed(() => METRICS_WITH_SPECIES.includes(selectMetric.value));
    const speciesLabel = computed(() => {
      const m = selectMetric.value;
      if (m === 'GBF3_NVIS') return 'Veg group:';
      if (m === 'GBF4_SNES') return 'Species:';
      if (m === 'GBF4_ECNES') return 'Community:';
      if (m === 'GBF8_GROUP') return 'Group:';
      if (m === 'GBF8_SPECIES') return 'Species:';
      return 'Species:';
    });

    // Previous selections memory (per category)
    const previousSelections = ref({
      "Sum": { species: "", landuse: "" },
      "Ag": { water: "", landuse: "" },
      "Ag Mgt": { agMgt: "", water: "", landuse: "" },
      "Non-Ag": { landuse: "" }
    });

    // Display labels for the "Sum" category's Type dimension
    const SUM_TYPE_LABELS = { 'ALL': 'ALL', 'ag': 'Ag', 'non-ag': 'Non-Ag', 'ag-man': 'Ag Mgt' };
    function formatLanduse(val) {
      return (selectCategory.value === 'Sum') ? (SUM_TYPE_LABELS[val] || val) : val;
    }

    // UI state
    const dataLoaded = ref(false);
    const isLoadingData = ref(false);
    const triggerVersion = ref(0);
    const isDrawerOpen = ref(false);

    // GBF2 mask overlay — only active when GBF2 metric is selected
    const gbf2MaskOverlay = computed(() =>
      (dataLoaded.value && selectMetric.value === 'GBF2') ? (window.BIO_GBF2_MASK || null) : null
    );

    // Cascade helper — reads from mapRegister[selectMetric]
    function doCascade(category) {
      const mr = mapRegister[selectMetric.value];
      const sumData = window[mr?.["Sum"]?.["name"]];
      const agData = window[mr?.["Ag"]?.["name"]];
      const amData = window[mr?.["Ag Mgt"]?.["name"]];
      const nonAgData = window[mr?.["Non-Ag"]?.["name"]];

      // Remember current selections for cross-category restore
      const curWater = selectWater.value;
      const curLanduse = selectLanduse.value;
      const curAgMgt = selectAgMgt.value;
      const curSpecies = selectSpecies.value;
      const withSpecies = hasSpecies.value;

      // Reset species when not used
      if (!withSpecies) {
        availableSpecies.value = [];
        selectSpecies.value = '';
      }

      if (category === "Sum") {
        availableAgMgt.value = [];
        availableWater.value = [];
        // For species-aware metrics the Sum map has a species/group dim (group/species → Type → year)
        if (withSpecies && sumData) {
          availableSpecies.value = Object.keys(sumData);
          const prevSp = previousSelections.value["Sum"].species || curSpecies;
          selectSpecies.value = (prevSp && availableSpecies.value.includes(prevSp)) ? prevSp : (availableSpecies.value[0] || '');
          availableLanduse.value = Object.keys(sumData[selectSpecies.value] || {});
        } else {
          availableSpecies.value = [];
          selectSpecies.value = '';
          availableLanduse.value = Object.keys(sumData || {});
        }
        const prevLU = previousSelections.value["Sum"].landuse || curLanduse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Ag") {
        availableWater.value = Object.keys(agData || {});
        const prevWater = previousSelections.value["Ag"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        let baseNode = agData?.[selectWater.value];
        if (withSpecies) {
          availableSpecies.value = Object.keys(baseNode || {});
          selectSpecies.value = (curSpecies && availableSpecies.value.includes(curSpecies)) ? curSpecies : (availableSpecies.value[0] || '');
          baseNode = baseNode?.[selectSpecies.value];
        }
        availableLanduse.value = Object.keys(baseNode || {});
        const prevLU = previousSelections.value["Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Ag Mgt") {
        availableAgMgt.value = Object.keys(amData || {});
        const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt || curAgMgt;
        selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');

        availableWater.value = Object.keys(amData?.[selectAgMgt.value] || {});
        const prevWater = previousSelections.value["Ag Mgt"].water || curWater;
        selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

        let baseNode = amData?.[selectAgMgt.value]?.[selectWater.value];
        if (withSpecies) {
          availableSpecies.value = Object.keys(baseNode || {});
          selectSpecies.value = (curSpecies && availableSpecies.value.includes(curSpecies)) ? curSpecies : (availableSpecies.value[0] || '');
          baseNode = baseNode?.[selectSpecies.value];
        }
        availableLanduse.value = Object.keys(baseNode || {});
        const prevLU = previousSelections.value["Ag Mgt"].landuse || curLanduse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (category === "Non-Ag") {
        let baseNode = nonAgData;
        if (withSpecies) {
          availableSpecies.value = Object.keys(baseNode || {});
          selectSpecies.value = (curSpecies && availableSpecies.value.includes(curSpecies)) ? curSpecies : (availableSpecies.value[0] || '');
          baseNode = baseNode?.[selectSpecies.value];
        }
        availableLanduse.value = Object.keys(baseNode || {});
        const prevLU = previousSelections.value["Non-Ag"].landuse || curLanduse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
      }
    }

    // Reactive data
    // map_bio_*_Sum (non-species metrics): Type → Year
    // map_bio_*_Sum (species-aware metrics): group/species → Type → Year
    // map_bio_*_Ag:    Water → LU → Year  (Water → species → LU → Year for species-aware)
    // map_bio_*_Am:    AgMgt → Water → LU → Year  (... → species → LU → Year)
    // map_bio_*_NonAg: LU → Year  (species → LU → Year for species-aware)
    const selectMapData = computed(() => {
      const metric = selectMetric.value;
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const species = selectSpecies.value;
      const landuse = selectLanduse.value;
      const year = selectYear.value;
      void triggerVersion.value;
      if (!dataLoaded.value) return {};
      const mr = mapRegister[metric];
      const mapData = window[mr?.[cat]?.["name"]];
      const withSpecies = METRICS_WITH_SPECIES.includes(metric);
      if (cat === "Sum") {
        // species-aware: group/species → Type → year; non-species: Type → year
        return withSpecies
          ? (mapData?.[species]?.[landuse]?.[year] || {})
          : (mapData?.[landuse]?.[year] || {});
      } else if (cat === "Ag") {
        return withSpecies
          ? (mapData?.[water]?.[species]?.[landuse]?.[year] || {})
          : (mapData?.[water]?.[landuse]?.[year] || {});
      } else if (cat === "Ag Mgt") {
        return withSpecies
          ? (mapData?.[agMgt]?.[water]?.[species]?.[landuse]?.[year] || {})
          : (mapData?.[agMgt]?.[water]?.[landuse]?.[year] || {});
      } else if (cat === "Non-Ag") {
        return withSpecies
          ? (mapData?.[species]?.[landuse]?.[year] || {})
          : (mapData?.[landuse]?.[year] || {});
      }
      return {};
    });

    // BIO_*_Ag chart:    Region → Water → [series(name=LU)]
    // BIO_*_Am chart:    Region → AgMgt → Water → [series(name=LU)]
    // BIO_*_NonAg chart: Region → [series(name=LU)]
    // For metrics with a species dim the hierarchy gains a Species level right
    // after Region (mirrors the map):
    //   BIO_*_Ag:    Region → Species → Water → [series(name=LU)]
    //   BIO_*_Am:    Region → Species → AgMgt → Water → [series(name=LU)]
    //   BIO_*_NonAg: Region → Species → [series(name=LU)]
    // Sum: no chart data available
    const selectChartData = computed(() => {
      const metric = selectMetric.value;
      const cat = selectCategory.value;
      const agMgt = selectAgMgt.value;
      const water = selectWater.value;
      const landuse = selectLanduse.value;
      const region = selectRegion.value;
      const species = selectSpecies.value;
      const withSpecies = hasSpecies.value;
      void triggerVersion.value;
      if (!dataLoaded.value) return {};
      const cr = chartRegister[metric];
      const regionNode = window[cr?.[cat]?.["name"]]?.[region];
      // For species-aware metrics, descend into the species level first.
      const chartData = withSpecies ? regionNode?.[species] : regionNode;
      let seriesData;

      if (cat === "Sum") {
        // Sum chart always shows all series (ag + non-ag + ag-man + outside) stacked.
        // The landuse/Type selector controls only the map; chart always shows full breakdown.
        const sumEntry = cr?.['Sum'] ?? cr?.['overview']?.['sum'];
        const rawSumData = window[sumEntry?.['name']]?.[region];
        seriesData = (withSpecies ? rawSumData?.[species] : rawSumData) || [];
      } else if (cat === "Ag") {
        seriesData = chartData?.[water] || [];
        seriesData = seriesData.filter(s => landuse === "ALL" || s.name === landuse);
      } else if (cat === "Ag Mgt") {
        seriesData = chartData?.[agMgt]?.[water] || [];
        seriesData = seriesData.filter(s => landuse === "ALL" || s.name === landuse);
      } else if (cat === "Non-Ag") {
        seriesData = (chartData || []).filter(s => landuse === "ALL" || s.name === landuse);
      }

      return {
        ...window["Chart_default_options"],
        chart: { height: 440 },
        plotOptions: { column: { stacking: 'normal' } },
        yAxis: { title: { text: availableUnit["Biodiversity"] } },
        series: seriesData || [],
      };
    });

    // Memory cleanup on component unmount
    onUnmounted(() => {
      window.MemoryService.cleanupViewData(VIEW_NAME);
    });

    // ── Lazy loader (maps only) ──────────────────────────────────────────────
    async function ensureDataLoaded(metric, cat) {
      const mapEntry = mapRegister[metric]?.[cat];
      if (mapEntry && !window[mapEntry.name]) {
        isLoadingData.value = true;
        await loadScript(mapEntry.path, mapEntry.name, VIEW_NAME);
        isLoadingData.value = false;
      }
    }

    // Pre-load chart files for enabled metrics only
    async function loadAllCharts() {
      const pending = [];
      for (const metric of availableMetrics.value) {
        const metricCr = chartRegister[metric];
        for (const [key, val] of Object.entries(metricCr || {})) {
          if (key === 'overview') {
            for (const entry of Object.values(val || {})) {
              if (entry?.name && !window[entry.name])
                pending.push(loadScript(entry.path, entry.name, VIEW_NAME));
            }
          } else if (val?.name && !window[val.name]) {
            pending.push(loadScript(val.path, val.name, VIEW_NAME));
          }
        }
      }
      if (pending.length > 0) await Promise.allSettled(pending);
    }

    onMounted(async () => {
      await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
      await loadScript("./data/chart_option/Chart_default_options.js", "Chart_default_options", VIEW_NAME);

      // Determine enabled metrics from run scenario settings
      const runScenario = {};
      (window.Supporting_info.model_run_settings || []).forEach(s => {
        runScenario[s.parameter] = s.val;
      });

      // Update GBF3_NVIS label to show region mode (NRM or IBRA)
      const gbf3Mode = window.Supporting_info.GBF3_NVIS_REGION_MODE || 'NRM';
      METRIC_LABELS['GBF3_NVIS'] = `GBF3 NVIS (${gbf3Mode})`;

      const enabledMetrics = ['quality'];
      for (const [metric, settingKey] of Object.entries(METRIC_TO_SETTING)) {
        if (metric === 'quality') continue;
        if (settingKey && runScenario[settingKey] !== 'off' && mapRegister[metric]) {
          enabledMetrics.push(metric);
        }
      }
      availableMetrics.value = enabledMetrics;

      // Load GBF2 mask overlay upfront if enabled (small file)
      if (enabledMetrics.includes('GBF2')) {
        const mask = mapRegister['GBF2']['mask'];
        await loadScript(mask.path, mask.name, VIEW_NAME);
      }

      availableYears.value = window.Supporting_info.years;
      selectYear.value = availableYears.value[0] || 2020;

      // Load initial map + ALL chart files in parallel
      const initMetric = enabledMetrics[0];
      const initMr = mapRegister[initMetric] || {};
      const initCat = ALL_CATEGORIES.find(c => initMr[c]) || "Ag";
      await Promise.all([ensureDataLoaded(initMetric, initCat), loadAllCharts()]);

      // Cascade initial selections synchronously so computed has all values when dataLoaded=true
      selectMetric.value = initMetric; // doCascade reads mapRegister[selectMetric.value]
      doCascade(initCat);
      selectCategory.value = initCat;
      dataLoaded.value = true;
    });

    // Watchers and methods
    const toggleDrawer = () => {
      isDrawerOpen.value = !isDrawerOpen.value;
    };

    watch(yearIndex, (newIndex) => {
      selectYear.value = availableYears.value[newIndex];
    });

    // Metric change: re-cascade based on current category
    watch(selectMetric, async (newMetric) => {
      // If current category isn't available for the new metric, fall back to the first that is
      const mr = mapRegister[newMetric] || {};
      let cat = selectCategory.value;
      if (!mr[cat]) {
        cat = ALL_CATEGORIES.find(c => mr[c]) || cat;
        selectCategory.value = cat; // triggers selectCategory watcher which loads + cascades
        return;
      }
      const _me = mr[cat];
      if (_me && !window[_me.name]) {
        await ensureDataLoaded(newMetric, cat);
      }
      doCascade(cat);
      triggerVersion.value++;
    });

    // Progressive selection chain watchers
    watch(selectCategory, async (newCategory, oldCategory) => {
      // Save previous selections before switching
      if (oldCategory === "Sum") {
        previousSelections.value["Sum"] = { landuse: selectLanduse.value };
      } else if (oldCategory === "Ag") {
        previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Ag Mgt") {
        previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
      } else if (oldCategory === "Non-Ag") {
        previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
      }
      const _me = mapRegister[selectMetric.value]?.[newCategory];
      if (_me && !window[_me.name]) {
        await ensureDataLoaded(selectMetric.value, newCategory);
      }
      doCascade(newCategory);
      triggerVersion.value++;
    });

    watch(selectAgMgt, (newAgMgt) => {
      if (selectCategory.value !== "Ag Mgt") return;
      previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
      const amData = window[mapRegister[selectMetric.value]?.["Ag Mgt"]?.["name"]];

      availableWater.value = Object.keys(amData?.[newAgMgt] || {});
      const prevWater = previousSelections.value["Ag Mgt"].water;
      selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');

      let baseNode = amData?.[newAgMgt]?.[selectWater.value];
      if (hasSpecies.value) {
        availableSpecies.value = Object.keys(baseNode || {});
        if (!availableSpecies.value.includes(selectSpecies.value)) {
          selectSpecies.value = availableSpecies.value[0] || '';
        }
        baseNode = baseNode?.[selectSpecies.value];
      }
      availableLanduse.value = Object.keys(baseNode || {});
      const prevLU = previousSelections.value["Ag Mgt"].landuse;
      selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
    });

    watch(selectWater, (newWater) => {
      if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].water = newWater;
        const agData = window[mapRegister[selectMetric.value]?.["Ag"]?.["name"]];

        let baseNode = agData?.[newWater];
        if (hasSpecies.value) {
          availableSpecies.value = Object.keys(baseNode || {});
          if (!availableSpecies.value.includes(selectSpecies.value)) {
            selectSpecies.value = availableSpecies.value[0] || '';
          }
          baseNode = baseNode?.[selectSpecies.value];
        }
        availableLanduse.value = Object.keys(baseNode || {});
        const prevLU = previousSelections.value["Ag"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');

      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].water = newWater;
        const amData = window[mapRegister[selectMetric.value]?.["Ag Mgt"]?.["name"]];

        let baseNode = amData?.[selectAgMgt.value]?.[newWater];
        if (hasSpecies.value) {
          availableSpecies.value = Object.keys(baseNode || {});
          if (!availableSpecies.value.includes(selectSpecies.value)) {
            selectSpecies.value = availableSpecies.value[0] || '';
          }
          baseNode = baseNode?.[selectSpecies.value];
        }
        availableLanduse.value = Object.keys(baseNode || {});
        const prevLU = previousSelections.value["Ag Mgt"].landuse;
        selectLanduse.value = (prevLU && availableLanduse.value.includes(prevLU)) ? prevLU : (availableLanduse.value[0] || '');
      }
    });

    // Species change: re-derive landuse list (year stays). Does nothing for metrics without species.
    watch(selectSpecies, (newSpecies) => {
      if (!hasSpecies.value || !newSpecies) return;
      const cat = selectCategory.value;
      const mr = mapRegister[selectMetric.value] || {};
      let baseNode;
      if (cat === "Sum") {
        // Sum: group/species → Type → year; newSpecies changes the Type list
        const sumData = window[mr["Sum"]?.name];
        availableLanduse.value = Object.keys(sumData?.[newSpecies] || {});
        if (!availableLanduse.value.includes(selectLanduse.value)) {
          selectLanduse.value = availableLanduse.value[0] || '';
        }
        previousSelections.value["Sum"].species = newSpecies;
      } else if (cat === "Ag") {
        baseNode = window[mr["Ag"]?.name]?.[selectWater.value]?.[newSpecies];
        availableLanduse.value = Object.keys(baseNode || {});
        if (!availableLanduse.value.includes(selectLanduse.value)) {
          selectLanduse.value = availableLanduse.value[0] || '';
        }
      } else if (cat === "Ag Mgt") {
        baseNode = window[mr["Ag Mgt"]?.name]?.[selectAgMgt.value]?.[selectWater.value]?.[newSpecies];
        availableLanduse.value = Object.keys(baseNode || {});
        if (!availableLanduse.value.includes(selectLanduse.value)) {
          selectLanduse.value = availableLanduse.value[0] || '';
        }
      } else if (cat === "Non-Ag") {
        baseNode = window[mr["Non-Ag"]?.name]?.[newSpecies];
        availableLanduse.value = Object.keys(baseNode || {});
        if (!availableLanduse.value.includes(selectLanduse.value)) {
          selectLanduse.value = availableLanduse.value[0] || '';
        }
      } else {
        return;
      }
      triggerVersion.value++;
    });

    watch(selectLanduse, (newLanduse) => {
      if (selectCategory.value === "Sum") {
        previousSelections.value["Sum"].landuse = newLanduse;
        previousSelections.value["Sum"].species = selectSpecies.value;
      } else if (selectCategory.value === "Ag") {
        previousSelections.value["Ag"].landuse = newLanduse;
      } else if (selectCategory.value === "Ag Mgt") {
        previousSelections.value["Ag Mgt"].landuse = newLanduse;
      } else if (selectCategory.value === "Non-Ag") {
        previousSelections.value["Non-Ag"].landuse = newLanduse;
      }
    });

    const _state = {
      yearIndex,
      selectYear,
      selectRegion,

      METRIC_LABELS,
      availableYears,
      availableMetrics,
      availableCategories,
      availableAgMgt,
      availableWater,
      availableSpecies,
      availableLanduse,

      selectMetric,
      selectCategory,
      selectAgMgt,
      selectWater,
      selectSpecies,
      selectLanduse,

      hasSpecies,
      speciesLabel,
      formatLanduse,
      selectMapData,
      selectChartData,
      gbf2MaskOverlay,

      dataLoaded, isLoadingData,
      isDrawerOpen,
      toggleDrawer,
    };
    window._debug[VIEW_NAME] = _state;
    return _state;
  },
  template: /*html*/`
    <div class="relative w-full h-screen">

      <!-- Region selection dropdown -->
      <div class="absolute w-[262px] top-32 left-[20px] z-50 bg-white/70 rounded-lg shadow-lg max-w-xs z-[9999]">
        <filterable-dropdown></filterable-dropdown>
      </div>

      <!-- Year slider -->
      <div class="absolute top-[200px] left-[20px] z-[1001] w-[262px] bg-white/70 p-2 rounded-lg items-center">
        <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
        <el-slider
          v-if="availableYears && availableYears.length > 0"
          v-model="yearIndex"
          size="small"
          :min="0"
          :max="availableYears.length - 1"
          :step="1"
          :format-tooltip="index => availableYears[index]"
          :show-stops="true"
          @input="(index) => { yearIndex = index; selectYear = availableYears[index]; }"
        />
      </div>

      <!-- Data selection controls container -->
      <div class="absolute top-[285px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

        <!-- Metric buttons (always visible) -->
        <div class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Metric:</span>
          <button v-for="(val, key) in availableMetrics" :key="key"
            @click="selectMetric = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectMetric === val}">
            {{ METRIC_LABELS[val] || val }}
          </button>
        </div>

        <!-- Category buttons -->
        <div class="flex space-x-1">
          <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
          <button v-for="(val, key) in availableCategories" :key="key"
            @click="selectCategory = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
            :class="{'bg-sky-500 text-white': selectCategory === val}">
            {{ val }}
          </button>
        </div>

        <!-- Ag Mgt options (only for Ag Mgt category) -->
        <div v-if="dataLoaded && selectCategory === 'Ag Mgt' && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
          <button v-for="(val, key) in availableAgMgt" :key="key"
            @click="selectAgMgt = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectAgMgt === val}">
            {{ val }}
          </button>
        </div>

        <!-- Water options (Ag and Ag Mgt only) -->
        <div v-if="selectCategory !== 'Non-Ag' && selectCategory !== 'Sum' && dataLoaded && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
          <button v-for="(val, key) in availableWater" :key="key"
            @click="selectWater = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectWater === val}">
            {{ val }}
          </button>
        </div>

        <!-- Landuse options (for Sum: shows Type dimension values) -->
        <div v-if="dataLoaded" class="flex flex-wrap gap-1 max-w-[300px]">
          <span class="text-[0.8rem] mr-1 font-medium">{{ selectCategory === 'Sum' ? 'Type:' : 'Landuse:' }}</span>
          <button v-for="(val, key) in availableLanduse" :key="key"
            @click="selectLanduse = val"
            class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
            :class="{'bg-sky-500 text-white': selectLanduse === val}">
            {{ formatLanduse(val) }}
          </button>
        </div>
      </div>

      <!-- Map container with slide-out chart drawer -->
      <div style="position: relative; width: 100%; height: 100%; overflow: hidden;">

        <!-- Species / VegGroup / Community options (GBF3/GBF4/GBF8 only, including Sum category) — floating bottom-right scroll panel -->
        <div v-if="dataLoaded && availableSpecies.length > 0"
          class="absolute bottom-[20px] right-[20px] z-[1001] w-[280px] max-h-[260px] bg-white/85 rounded-lg shadow-md p-2 flex flex-col"
          :class="{ 'right-[440px]': isDrawerOpen }"
          style="transition: right 0.3s ease;">
          <div class="text-[0.8rem] font-medium mb-1 flex-shrink-0">{{ speciesLabel }}</div>
          <div class="flex flex-wrap gap-1 overflow-y-auto pr-1">
            <button v-for="(val, key) in availableSpecies" :key="key"
              @click="selectSpecies = val"
              class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1 text-left"
              :class="{'bg-sky-500 text-white': selectSpecies === val}">
              {{ val }}
            </button>
          </div>
        </div>

        <!-- Loading overlay shown while lazy-loading a new map file -->
        <div v-if="isLoadingData"
          class="absolute inset-0 z-[2000] flex items-center justify-center bg-white/60 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-2 text-gray-600 text-sm font-medium">
            <svg class="animate-spin h-8 w-8 text-sky-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            Loading map data…
          </div>
        </div>

        <!-- Map component takes full space -->
        <regions-map
          :mapData="selectMapData"
          :overlayGeoJSON="gbf2MaskOverlay"
          :show-legend="!isDrawerOpen"
          style="width: 100%; height: 100%;">
        </regions-map>

        <!-- Drawer toggle button -->
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
  `,
};
