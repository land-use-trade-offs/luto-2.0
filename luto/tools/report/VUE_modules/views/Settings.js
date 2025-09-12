window.SettingsView = {
  setup(props, { emit }) {
    const { ref, onMounted, watch, computed } = Vue;
    const loadScript = window.loadScript;

    // Tab management
    const activeTab = ref('settings');

    // Settings related reactive variables
    const searchTerm = ref('');
    const activeFilter = ref('all');
    const modelRunSettings = ref([]);
    const chartMemLogData = ref({});

    // Categories definition for parameter organization
    const categories = {
      'Basic Configuration': {
        icon: '⚙️',
        keywords: ['VERSION', 'INPUT_DIR', 'OUTPUT_DIR', 'RAW_DATA', 'SSP', 'RCP', 'SCENARIO', 'OBJECTIVE', 'SIM_YEARS']
      },
      'Diet & Consumption': {
        icon: '🍽️',
        keywords: ['DIET_DOM', 'DIET_GLOB', 'WASTE', 'FEED_EFFICIENCY', 'CONVERGENCE', 'IMPORT_TREND', 'OFF_LAND_COMMODITIES', 'EGGS_AVG_WEIGHT']
      },
      'Economic Parameters': {
        icon: '💰',
        keywords: ['DISCOUNT_RATE', 'RESFACTOR', 'CARBON_PRICE', 'COST', 'AMORTISE', 'AMORTISATION', 'FENCING_COST', 'IRRIG_COST', 'MAINTENANCE', 'ECOSYSTEM_SERVICES']
      },
      'Risk & Environmental': {
        icon: '⚠️',
        keywords: ['RISK_OF_REVERSAL', 'FIRE_RISK', 'CO2_FERT', 'SOC_AMORTISATION']
      },
      'Biodiversity & Conservation': {
        icon: '🌿',
        keywords: ['BIO_CONTRIBUTION', 'GBF', 'BIODIVERSITY', 'HABITAT', 'CONNECTIVITY', 'EP_', 'RP_', 'DEGRADED_AREAS']
      },
      'Climate & GHG': {
        icon: '🌡️',
        keywords: ['GHG', 'CO2', 'CARBON', 'CLIMATE', 'EMISSIONS', 'SCOPE_1', 'CROP_GHG', 'LVSTK_GHG']
      },
      'Water Management': {
        icon: '💧',
        keywords: ['WATER', 'IRRIGATION', 'DRAINAGE', 'LIVESTOCK_DRINKING', 'LICENSE']
      },
      'Land Use & Planning': {
        icon: '🌾',
        keywords: ['LAND', 'AGRICULTURAL', 'NON_AG', 'PLANTING', 'AGROFORESTRY', 'NO_GO', 'REGIONAL_ADOPTION', 'CULL_MODE', 'MAX_LAND_USES']
      },
      'Carbon Plantings': {
        icon: '🌳',
        keywords: ['CP_BLOCK', 'CP_BELT', 'CARBON_PLANTING']
      },
      'Riparian & Agroforestry': {
        icon: '🌲',
        keywords: ['RIPARIAN', 'AF_', 'AGROFORESTRY', 'ROW_WIDTH', 'ROW_SPACING', 'PROPORTION', 'BUFFER_WIDTH', 'TORTUOSITY']
      },
      'Agricultural Management': {
        icon: '🚜',
        keywords: ['AG_MANAGEMENTS', 'HIR_', 'BEEF_HIR', 'SHEEP_HIR', 'SAVBURN', 'PRODUCTIVITY', 'EFFECT_YEARS', 'USE_THRESHOLD']
      },
      'Solver & Optimization': {
        icon: '🔧',
        keywords: ['SOLVER', 'WEIGHT', 'TOLERANCE', 'PRESOLVE', 'CROSSOVER', 'BARRIER', 'SCALE_FLAG', 'NUMERIC_FOCUS', 'BARHOMOGENOUS', 'CONSTRAINT_TYPE', 'PENALTY', 'ALPHA', 'BETA']
      },
      'Output & Processing': {
        icon: '📊',
        keywords: ['WRITE', 'OUTPUT', 'PARALLEL', 'GEOTIFFS', 'RESCALE_FACTOR', 'VERBOSE', 'KEEP_OUTPUTS', 'ROUND_DECMIALS']
      },
      'System Resources': {
        icon: '💻',
        keywords: ['THREADS', 'MEM', 'NCPUS', 'TIME', 'QUEUE', 'JOB_NAME', 'SOLVE_METHOD', 'AGGREGATE']
      },
      'Land Use Configuration': {
        icon: '🗺️',
        keywords: ['EXCLUDE_NO_GO', 'VECTORS', 'REVERSIBLE', 'BASE_CODE', 'PERCENTAGE']
      }
    };

    const filterButtons = ref([
      { key: 'all', label: 'All' },
      { key: 'BIO_', label: 'Biodiversity' },
      { key: 'GBF', label: 'GBF' },
      { key: 'GHG', label: 'GHG' },
      { key: 'WATER', label: 'Water' },
      { key: 'SOLVER', label: 'Solver' },
      { key: 'AG_', label: 'Agriculture' },
      { key: 'CP_', label: 'Carbon' },
      { key: 'COST', label: 'Costs' }
    ]);

    // Helper functions
    const categorizeParameters = (parameters) => {
      const categorized = {};
      const uncategorized = [];

      Object.keys(categories).forEach(cat => {
        categorized[cat] = [];
      });

      parameters.forEach(param => {
        let assigned = false;

        for (const [categoryName, categoryData] of Object.entries(categories)) {
          if (categoryData.keywords.some(keyword =>
            param.parameter.toUpperCase().includes(keyword.toUpperCase())
          )) {
            categorized[categoryName].push(param);
            assigned = true;
            break;
          }
        }

        if (!assigned) {
          uncategorized.push(param);
        }
      });

      if (uncategorized.length > 0) {
        categorized['Other'] = uncategorized;
      }

      return categorized;
    };

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

    // Computed properties
    const filteredParameters = computed(() => {
      let filtered = modelRunSettings.value;

      if (searchTerm.value) {
        filtered = filtered.filter(param =>
          param.parameter.toLowerCase().includes(searchTerm.value.toLowerCase()) ||
          param.val.toString().toLowerCase().includes(searchTerm.value.toLowerCase())
        );
      }

      if (activeFilter.value && activeFilter.value !== 'all') {
        filtered = filtered.filter(param =>
          param.parameter.includes(activeFilter.value)
        );
      }

      return filtered;
    });

    const filteredCategories = computed(() => {
      const categorized = categorizeParameters(filteredParameters.value);
      return Object.entries(categorized)
        .filter(([_, params]) => params.length > 0)
        .map(([name, params]) => ({
          name,
          icon: categories[name]?.icon || '📋',
          params
        }));
    });

    const stats = computed(() => {
      const totalParams = filteredParameters.value.length;
      const activeCategories = filteredCategories.value.length;
      const biodiversityParams = filteredParameters.value.filter(p => p.parameter.includes('BIO_')).length;
      const enabledFeatures = filteredParameters.value.filter(p =>
        p.val === 'on' || p.val === 'True' || p.val === 'true'
      ).length;

      return [
        { value: totalParams, label: 'Total Parameters' },
        { value: activeCategories, label: 'Active Categories' },
        { value: biodiversityParams, label: 'Biodiversity Params' },
        { value: enabledFeatures, label: 'Enabled Features' }
      ];
    });

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

    return {
      activeTab,
      searchTerm,
      activeFilter,
      filterButtons,
      filteredCategories,
      stats,
      getValueClass,
      chartMemLogData,
    };
  },

  template: `
    <div class="min-h-screen p-5" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
      <div class="max-w-7xl mx-auto" style="backdrop-filter: blur(16px); background: rgba(255, 255, 255, 0.95);" class="rounded-3xl p-8 shadow-2xl">
        
        <!-- Header -->
        <div class="text-center mb-10 p-6 rounded-2xl text-white shadow-lg" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
          <h1 class="text-2xl md:text-3xl font-bold mb-3">Model Settings & Logs</h1>
          <p class="text-sm opacity-90"><strong>Configuration Overview and Memory Usage</strong></p>
        </div>

        <!-- Tab Navigation -->
        <div class="mb-8">
          <div class="flex space-x-1 bg-gray-100 p-1 rounded-lg">
            <button 
              @click="activeTab = 'settings'"
              :class="[
                'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all duration-200',
                activeTab === 'settings' 
                  ? 'bg-white text-blue-600 shadow-md' 
                  : 'text-gray-600 hover:text-gray-800'
              ]"
            >
              ⚙️ Model Settings
            </button>
            <button 
              @click="activeTab = 'memory'"
              :class="[
                'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all duration-200',
                activeTab === 'memory' 
                  ? 'bg-white text-blue-600 shadow-md' 
                  : 'text-gray-600 hover:text-gray-800'
              ]"
            >
              📊 Memory Log
            </button>
          </div>
        </div>

        <!-- Settings Tab Content -->
        <div v-if="activeTab === 'settings'">
          <!-- Statistics -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-5 mb-8">
            <div v-for="(stat, index) in stats" :key="index" 
                 class="text-white p-5 rounded-2xl text-center shadow-lg transition-all duration-300 hover:transform hover:translateY(-2px)"
                 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
              <div class="text-xl md:text-2xl font-bold mb-2">{{ stat.value }}</div>
              <div class="text-xs opacity-90">{{ stat.label }}</div>
            </div>
          </div>

          <!-- Search and Filters -->
          <div class="mb-8 space-y-4">
            <div class="flex flex-col lg:flex-row gap-4 items-stretch lg:items-center">
              <input 
                v-model="searchTerm"
                type="text" 
                placeholder="🔍 Search parameters..."
                class="flex-1 min-w-64 px-3 py-2 border-2 border-gray-200 rounded-full text-sm transition-all duration-300 focus:outline-none focus:border-blue-400 focus:shadow-lg bg-white"
              >
              <div class="flex flex-wrap gap-3">
                <button 
                  v-for="filter in filterButtons" 
                  :key="filter.key"
                  @click="activeFilter = filter.key"
                  :class="[
                    'px-3 py-2 border-2 rounded-full cursor-pointer transition-all duration-300 hover:transform hover:translateY(-2px) text-xs',
                    activeFilter === filter.key 
                      ? 'bg-blue-500 text-white border-blue-500 shadow-lg' 
                      : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-blue-500 hover:text-white hover:border-blue-500'
                  ]"
                >
                  {{ filter.label }}
                </button>
              </div>
            </div>
          </div>

          <!-- Parameters Grid -->
          <div v-if="filteredCategories.length === 0" class="text-center text-gray-500 italic p-10 bg-gray-50 rounded-xl text-sm">
            No parameters found matching your search.
          </div>
          
          <div v-else class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <div 
              v-for="(category, index) in filteredCategories" 
              :key="category.name"
              class="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 transition-all duration-300 hover:transform hover:translateY(-5px)"
            >
              <h3 class="text-sm font-bold text-gray-800 mb-5 pb-3 border-b-2 border-blue-400 flex items-center gap-3">
                <span class="text-lg">{{ category.icon }}</span>
                {{ category.name }} ({{ category.params.length }})
              </h3>
              
              <div class="space-y-0">
                <div 
                  v-for="param in category.params" 
                  :key="param.parameter"
                  class="flex justify-between items-start gap-4 py-3 border-b border-gray-100 last:border-b-0 hover:bg-blue-50 hover:mx-[-1.5rem] hover:px-6 hover:rounded-lg transition-all duration-200 min-h-[3rem]"
                >
                  <span class="font-semibold text-gray-700 text-xs flex-1 min-w-0 max-w-[50%]" style="word-wrap: break-word; word-break: break-word; hyphens: auto; line-height: 1.4;">
                    {{ param.parameter }}
                  </span>
                  <span 
                    :class="[
                      'font-medium text-xs font-mono flex-1 min-w-0 max-w-[50%] px-2 py-1 rounded-lg',
                      getValueClass(param.val)
                    ]"
                    :title="param.val"
                    style="word-wrap: break-word; word-break: break-word; white-space: pre-wrap; line-height: 1.4; overflow-wrap: break-word;"
                  >
                    {{ param.val }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Memory Log Tab Content -->
        <div v-if="activeTab === 'memory'" class="bg-white rounded-2xl p-6 shadow-lg">
          <div class="flex items-center justify-start mb-4">
            <span class="text-lg mr-3">📊</span>
            <h2 class="text-lg font-bold text-gray-800">Memory Usage Log</h2>
          </div>
          <hr class="border-gray-300 mb-6">
          <div class="h-[600px]">
            <chart-container class="w-full h-full rounded-lg" :chartData="chartMemLogData"/>
          </div>
        </div>

      </div>
    </div>
  `,
};