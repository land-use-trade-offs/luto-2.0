window.SettingsView = {
  name: 'SettingsView',
  setup(props, { emit }) {
    const { ref, onMounted, watch, computed } = Vue;
    const loadScript = window.loadScript;
    const VIEW_NAME = "Settings";

    // Tab management
    const activeTab = ref('settings');

    // Settings related reactive variables
    const searchTerm = ref('');
    const activeFilter = ref('all');
    const modelRunSettings = ref([]);
    const chartMemLogData = ref({});

    // The groups of the settings, in the order of the buttons. A setting belongs to every group that names it
    // (`names`, exact) or whose pattern it matches (`patterns`), so one that drives several parts of the model
    // (SSP: demand, water, GBF8) shows under each; what no group takes falls under 'Other'.
    const GROUPS = [
      { key: 'Demand', names: ['SSP', 'SCENARIO', 'DIET_DOM', 'DIET_GLOB', 'CONVERGENCE', 'IMPORT_TREND', 'WASTE', 'FEED_EFFICIENCY', 'APPLY_DEMAND_MULTIPLIERS', 'DEMAND_BOUNDS', 'OFF_LAND_COMMODITIES', 'EGGS_AVG_WEIGHT', 'GHG_EMISSIONS_LIMITS', 'GHG_TARGETS_DICT'] },
      { key: 'GHG', names: ['CARBON_EFFECTS_WINDOW', 'RISK_OF_REVERSAL', 'FIRE_RISK', 'HIR_EXISTING_PROJECTS', 'HIR_CARBON_REVENUE', 'HIR_SHARE_OF_REGISTER_RATE', 'OFF_LAND_COMMODITIES', 'CARBON_PRICES_FIELD', 'CARBON_PRICE_COSTANT'], patterns: [/^GHG_/, /GHG_SCOPE_1$/] },
      { key: 'Water', names: ['SSP', 'LIVESTOCK_DRINKING_WATER', 'INCLUDE_WATER_LICENSE_COSTS'], patterns: [/^WATER_/] },
      { key: 'GBF2', names: ['BIO_QUALITY_LAYER', 'BIO_CONTRIBUTION_LDS', 'EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS', 'ELASTIC_FAMILIES'], patterns: [/^GBF2_/, /^RENEWABLE_GBF2_/] },
      { key: 'GBF3', names: ['WRITE_GBF3_NVIS', 'ELASTIC_FAMILIES'], patterns: [/^GBF3_/] },
      { key: 'GBF4 SNES', names: ['GBF4_TARGET_SNES', 'WRITE_GBF4_SNES', 'ELASTIC_FAMILIES'], patterns: [/^GBF4_SNES_/] },
      { key: 'GBF4 ECNES', names: ['GBF4_TARGET_ECNES', 'WRITE_GBF4_ECNES', 'ELASTIC_FAMILIES'], patterns: [/^GBF4_ECNES_/] },
      { key: 'GBF8', names: ['SSP', 'BIO_ROW_GENERATION', 'ELASTIC_FAMILIES'], patterns: [/^GBF8_/] },
      { key: 'Biodiversity quality', names: ['BIO_QUALITY_LAYERS', 'BIO_QUALITY_LAYER'], patterns: [/^CONNECTIVITY_/, /^HCAS_/, /^BIO_CONTRIBUTION_/] },
      { key: 'Ag Managements', names: ['TECH_ADOPT_MULT', 'SAVBURN_COST_HA_YR', 'AGRICULTURAL_MANAGEMENT_USE_THRESHOLD', 'BIO_CONTRIBUTION_LDS', 'RENEWABLES_OPTIONS'], patterns: [/^AG_MANAGEMENT/, /HIR_/] },
      { key: 'Non-Ag Landuses', names: ['FENCING_COST_PER_M', 'NON_AGRICULTURAL_LU_BASE_CODE', 'BIO_CONTRIBUTION_ENV_PLANTING', 'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK', 'BIO_CONTRIBUTION_CARBON_PLANTING_BELT', 'BIO_CONTRIBUTION_RIPARIAN_PLANTING', 'BIO_CONTRIBUTION_AGROFORESTRY', 'BIO_CONTRIBUTION_BECCS', 'BIO_CONTRIBUTION_DESTOCKING', 'CARBON_EFFECTS_WINDOW', 'RISK_OF_REVERSAL', 'FIRE_RISK'], patterns: [/^NON_AG_LAND_USES/, /^(EP|CP|RP|AF)_/, /^RIPARIAN_PLANTING_/, /^AGROFORESTRY_/] },
      { key: 'Economics', names: ['OBJECTIVE', 'AMORTISE_UPFRONT_COSTS', 'DISCOUNT_RATE', 'AMORTISATION_PERIOD', 'TRANSITION_COST_MULT', 'DYNAMIC_PRICE', 'CARBON_PRICES_FIELD', 'CARBON_PRICE_COSTANT', 'REMOVE_IRRIG_COST', 'NEW_IRRIG_COST', 'INCLUDE_WATER_LICENSE_COSTS', 'FENCING_COST_PER_M', 'SAVBURN_COST_HA_YR'], patterns: [/MAINTENANCE_COST/, /ECOSYSTEM_SERVICES_BENEFIT/] },
      { key: 'Regional Land-use limit', names: ['EXCLUDE_NO_GO_LU', 'NO_GO_VECTORS'], patterns: [/^REGIONAL_ADOPTION_/] },
      { key: 'Renewable Energy', names: ['RE_TARGET_LEVEL', 'INSTALL_CAPACITY_MW_HA'], patterns: [/^RENEWABLE/, /^EXCLUDE_RENEWABLES_/] },
      { key: 'Yield', names: ['SSP', 'YIELD_RCP', 'RCP', 'CO2_FERT', 'PRODUCTIVITY_TREND'] },      // RCP: the name of YIELD_RCP in runs before 2026-11-17
      { key: 'Run and land uses', names: ['VERSION', 'INPUT_DIR', 'OUTPUT_DIR', 'RAW_DATA', 'SSP', 'SCENARIO', 'RESFACTOR', 'SIM_YEARS', 'OBJECTIVE', 'LU2TYPE', 'LVSTK_K_FILE', 'STUBBLE_DSE_FILE', 'STUBBLE_LAND_USES'], patterns: [/^LUS_/] },
      { key: 'Solver', names: ['RESCALE_FACTOR', 'SOLVER_COEFF_MIN', 'THREADS', 'SCALE_FLAG', 'RETRY_PARAMS', 'BIO_ROW_GENERATION', 'ROUND_DECIMALS'], patterns: [/^BOUND_PROP_/, /^ELASTIC_/, /_TOLERANCE$/, /^ROWGEN_/] },
      { key: 'Output', names: ['VERBOSE'], patterns: [/^WRITE_/] },
    ];
    const OTHER = 'Other';
    const groupsOf = (name) => {
      const found = GROUPS.filter(group => (group.names || []).includes(name) || (group.patterns || []).some(pattern => pattern.test(name))).map(group => group.key);
      return found.length ? found : [OTHER];
    };

    // Every setting with the groups it belongs to, then narrowed by the search box
    const groupedSettings = computed(() => modelRunSettings.value.map(param => ({ ...param, groups: groupsOf(param.parameter) })));
    const searchedSettings = computed(() => {
      const term = searchTerm.value.toLowerCase();
      return term ? groupedSettings.value.filter(param => param.parameter.toLowerCase().includes(term) || String(param.val).toLowerCase().includes(term)) : groupedSettings.value;
    });
    // The buttons: 'All' then every group that holds a setting, each with its count (of what the search leaves)
    const filterButtons = computed(() => [
      { key: 'all', label: 'All', count: searchedSettings.value.length },
      ...[...GROUPS.map(group => group.key), OTHER]
        .map(key => ({ key, label: key, count: searchedSettings.value.filter(param => param.groups.includes(key)).length }))
        .filter(button => button.count > 0 || button.key === activeFilter.value),
    ]);
    // The cards: under 'All' one per group, a setting sitting in the first group that takes it; else the one group
    const filteredCategories = computed(() => {
      if (activeFilter.value !== 'all') {
        const params = searchedSettings.value.filter(param => param.groups.includes(activeFilter.value));
        return params.length ? [{ name: activeFilter.value, params }] : [];
      }
      return [...GROUPS.map(group => group.key), OTHER]
        .map(name => ({ name, params: searchedSettings.value.filter(param => param.groups[0] === name) }))
        .filter(category => category.params.length > 0);
    });

    // The setting whose pop-up is open ('<card>/<setting>', '' for none): opened by its i, closed by a click anywhere
    const openInfo = ref('');
    const toggleInfo = (key) => { openInfo.value = openInfo.value === key ? '' : key; };
    const closeInfo = () => { openInfo.value = ''; };
    onMounted(() => document.addEventListener('click', closeInfo));
    Vue.onUnmounted(() => document.removeEventListener('click', closeInfo));

    // The note of a setting: hand-written, in services/SettingsNotes.js (null for a setting that has none yet)
    const noteOf = (param) => (window.SettingsNotes || {})[param.parameter] || null;

    const getValueClass = (value) => {
      const length = typeof value === 'string' ? value.length : 0;
      if (length > 200) {
        return 'bg-blue-50 text-blue-700 border border-blue-200';
      } else if (length > 50) {
        return 'bg-yellow-50 text-yellow-700 border border-yellow-200';
      } else {
        return 'bg-green-50 text-green-700 border border-green-200';
      }
    };

    // Load scripts and data when the component is mounted
    onMounted(async () => {
      try {
        await loadScript("./data/Supporting_info.js", 'Supporting_info');
        await loadScript("./data/chart_option/Chart_default_options.js", 'Chart_default_options');
        await loadScript("./data/chart_option/chartMemLogOptions.js", 'chartMemLogOptions');

        modelRunSettings.value = window['Supporting_info']['model_run_settings'] || [];

        chartMemLogData.value = {
          ...window['Chart_default_options'],
          ...window['chartMemLogOptions'],
          series: window['Supporting_info'].mem_logs,
        };

      } catch (error) {
        console.error("Error loading dependencies for Settings view:", error);
      }
    });

    const _state = {
      activeTab,
      searchTerm,
      activeFilter,
      filterButtons,
      filteredCategories,
      openInfo,
      toggleInfo,
      noteOf,
      getValueClass,
      chartMemLogData,
    };
    window._debug[VIEW_NAME] = _state;
    return _state;
  },

  template: /*html*/`
    <div class="p-6">

        <!-- Header -->
        <div class="mb-6">
          <h1 class="text-2xl font-bold text-gray-800 mb-1">Model Settings & Logs</h1>
          <p class="text-sm text-gray-500">Configuration Overview and Memory Usage</p>
        </div>

        <!-- Tab Navigation -->
        <div class="mb-6">
          <div class="inline-flex border-b border-gray-200">
            <button
              @click="activeTab = 'settings'"
              :class="[
                'py-2 px-4 text-sm font-medium border-b-2 transition-all duration-200 -mb-px',
                activeTab === 'settings'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              ]"
            >
              Model Settings
            </button>
            <button
              @click="activeTab = 'memory'"
              :class="[
                'py-2 px-4 text-sm font-medium border-b-2 transition-all duration-200 -mb-px',
                activeTab === 'memory'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              ]"
            >
              Memory Log
            </button>
          </div>
        </div>

        <!-- Settings Tab Content -->
        <div v-if="activeTab === 'settings'">
          <!-- Search and Filters -->
          <div class="mb-6 space-y-3">
            <div class="flex flex-col lg:flex-row gap-3 items-stretch lg:items-center">
              <input
                v-model="searchTerm"
                type="text"
                placeholder="Search parameters..."
                class="flex-1 min-w-64 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 bg-white"
              >
              <div class="flex flex-wrap gap-2">
                <button
                  v-for="filter in filterButtons"
                  :key="filter.key"
                  @click="activeFilter = filter.key"
                  :class="[
                    'px-3 py-1.5 border rounded-lg cursor-pointer transition-all duration-200 text-xs',
                    activeFilter === filter.key
                      ? 'bg-blue-500 text-white border-blue-500'
                      : 'bg-white text-gray-600 border-gray-300 hover:bg-blue-50 hover:border-blue-300'
                  ]"
                >
                  {{ filter.label }} ({{ filter.count }})
                </button>
              </div>
            </div>
          </div>

          <!-- Parameters Grid -->
          <div v-if="filteredCategories.length === 0" class="text-center text-gray-500 italic p-10 bg-gray-50 rounded-lg text-sm">
            No parameters found matching your search.
          </div>

          <div v-else class="grid grid-cols-1 gap-4" :class="activeFilter === 'all' ? 'lg:grid-cols-2 xl:grid-cols-3' : 'max-w-4xl'">
            <div
              v-for="(category, index) in filteredCategories"
              :key="category.name"
              class="bg-white rounded-lg p-5 border border-gray-200"
            >
              <h3 class="text-sm font-bold text-gray-800 mb-4 pb-2 border-b-2 border-blue-400 flex items-center gap-2">
                {{ category.name }} ({{ category.params.length }})
              </h3>

              <div class="space-y-0">
                <div
                  v-for="param in category.params"
                  :key="param.parameter"
                  class="relative flex justify-between items-start gap-3 py-2 border-b border-gray-100 last:border-b-0 hover:bg-gray-50 transition-colors duration-150 min-h-[2.5rem]"
                >
                  <!-- The pop-up of the setting (opened by its i): its name, the note of settings.py, its groups (the value is on the row) -->
                  <div v-if="openInfo === category.name + '/' + param.parameter" @click.stop class="absolute z-50 right-2 top-full -mt-1 w-[min(34rem,90vw)] max-h-96 overflow-y-auto bg-white border border-gray-300 rounded-lg shadow-lg p-4 text-xs">
                    <div class="font-mono font-bold text-sm text-gray-800 break-all">{{ param.parameter }}</div>
                    <!-- The hand-written note of the setting (services/SettingsNotes.js): what it does, the detail, its options -->
                    <template v-if="noteOf(param)">
                      <p class="mt-2 text-[0.85rem] font-semibold text-gray-800 leading-snug">{{ noteOf(param).what }}</p>
                      <p v-for="(line, position) in noteOf(param).detail || []" :key="position" class="mt-1.5 text-gray-700 leading-relaxed">{{ line }}</p>
                      <template v-if="(noteOf(param).options || []).length">
                        <div class="mt-3 text-[0.65rem] font-semibold uppercase tracking-wide text-gray-400">Options</div>
                        <div v-for="option in noteOf(param).options" :key="option[0]" class="mt-1 flex items-baseline gap-2 rounded px-1.5 py-1"
                          :class="option[0] === String(param.val) ? 'bg-sky-50 ring-1 ring-sky-300' : ''">
                          <span class="shrink-0 px-1.5 py-0.5 rounded font-mono text-[0.68rem] border"
                            :class="option[0] === String(param.val) ? 'bg-sky-500 text-white border-sky-500' : 'bg-white text-gray-600 border-gray-300'">{{ option[0] }}</span>
                          <span class="text-gray-700 leading-snug">{{ option[1] }}</span>
                          <span v-if="option[0] === String(param.val)" class="ml-auto shrink-0 text-[0.62rem] font-semibold uppercase tracking-wide text-sky-600">this run</span>
                        </div>
                      </template>
                    </template>
                    <p v-else class="mt-1.5 text-gray-400 italic">No note for this setting yet: add it to services/SettingsNotes.js.</p>
                    <div class="mt-3 pt-2 border-t border-gray-100 flex flex-wrap gap-1">
                      <span v-for="group in param.groups" :key="group" class="px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 text-[0.65rem]">{{ group }}</span>
                    </div>
                  </div>
                  <span class="font-semibold text-gray-700 text-xs flex-1 min-w-0 max-w-[50%]" style="word-wrap: break-word; word-break: break-word; hyphens: auto; line-height: 1.4;">
                    {{ param.parameter }}
                  </span>
                  <span
                    :class="[
                      'font-medium text-xs font-mono flex-1 min-w-0 max-w-[50%] px-2 py-1 rounded',
                      getValueClass(param.val)
                    ]"
                    style="word-wrap: break-word; word-break: break-word; white-space: pre-wrap; line-height: 1.4; overflow-wrap: break-word;"
                  >
                    {{ param.val }}
                  </span>
                  <!-- The i of the setting: a click opens its pop-up, a click anywhere else closes it -->
                  <button @click.stop="toggleInfo(category.name + '/' + param.parameter)" :aria-label="'About ' + param.parameter"
                    class="shrink-0 w-5 h-5 mt-0.5 rounded-full border text-[0.65rem] font-serif italic font-bold leading-none"
                    :class="openInfo === category.name + '/' + param.parameter ? 'bg-sky-500 text-white border-sky-500' : 'bg-white text-sky-600 border-sky-300 hover:bg-sky-50'"
                  >i</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Memory Log Tab Content -->
        <div v-if="activeTab === 'memory'" class="bg-white rounded-lg p-5 border border-gray-200">
          <div class="flex items-center justify-start mb-4">
            <h2 class="text-lg font-bold text-gray-800">Memory Usage Log</h2>
          </div>
          <hr class="border-gray-200 mb-6">
          <div class="h-[600px]">
            <chart-container class="w-full h-full rounded-lg" :chartData="chartMemLogData"/>
          </div>
        </div>

    </div>
  `,
};