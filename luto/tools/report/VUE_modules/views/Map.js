window['MapView'] = {
    name: 'MapView',
    setup() {
        const { inject, ref, watch, onMounted, onUnmounted, computed } = Vue;

        // Data|Map service
        const mapRegister = window.MapService.mapCategories["Dvar"];
        const loadScript = window.loadScriptWithTracking;
        const VIEW_NAME = "Map";

        const yearIndex = ref(0);
        const selectYear = ref(2020);
        const selectRegion = inject("globalSelectedRegion");

        const availableYears = ref([]);

        // "Land-use": lumap  dims=[lm]          combo=[water]
        // "Ag":       dvar_Ag dims=[lm,lu]       combo=[water,lu]
        // "Ag Mgt":   dvar_Am dims=[am,lm,lu]    combo=[am,water,lu]
        // "Non-Ag":   dvar_NonAg dims=[lu]       combo=[lu]
        const availableCategories = ["Land-use", "Ag", "Ag Mgt", "Non-Ag"];
        const availableAgMgt = ref([]);
        const availableWater = ref([]);
        const availableLanduse = ref([]);

        const selectCategory = ref("");
        const selectAgMgt = ref("");
        const selectWater = ref("");
        const selectLanduse = ref("");

        const previousSelections = ref({
            "Land-use": { water: "" },
            "Ag": { water: "", landuse: "" },
            "Ag Mgt": { agMgt: "", water: "", landuse: "" },
            "Non-Ag": { landuse: "" }
        });

        const dataLoaded = ref(false);
        const isLoadingData = ref(false);

        // ── Per-combo map layer loader ───────────────────────────────────────
        const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);
        const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

        // ── Helpers ──────────────────────────────────────────────────────────
        const CAT_TO_KEY = { "Land-use": "Lumap", "Ag": "Ag", "Ag Mgt": "Ag Mgt", "Non-Ag": "Non-Ag" };

        function getTree(cat) {
            const entry = mapRegister[CAT_TO_KEY[cat] || cat];
            return window[entry?.indexName]?.tree ?? [];
        }

        async function ensureIndexLoaded(cat) {
            const entry = mapRegister[CAT_TO_KEY[cat] || cat];
            if (entry && !window[entry.indexName]) {
                isLoadingData.value = true;
                await loadScript(entry.indexPath, entry.indexName, VIEW_NAME);
                isLoadingData.value = false;
            }
        }

        async function cascadeAndLoad(cat) {
            const prev = previousSelections.value[cat] || {};
            const key = CAT_TO_KEY[cat] || cat;
            const tree = getTree(cat);

            if (cat === "Land-use") {
                // tree is a flat list of lm values
                availableAgMgt.value = [];
                availableWater.value = Array.isArray(tree) ? tree : Object.keys(tree);
                availableLanduse.value = [];
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');
                await ensureComboLayer(mapRegister[key].layerPrefix, [selectWater.value]);

            } else if (cat === "Ag") {
                // tree: { lm: [lu] }
                availableAgMgt.value = [];
                availableWater.value = Object.keys(tree);
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');
                availableLanduse.value = tree[selectWater.value] || [];
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
                await ensureComboLayer(mapRegister[key].layerPrefix, [selectWater.value, selectLanduse.value]);

            } else if (cat === "Ag Mgt") {
                // tree: { am: { lm: [lu] } }
                availableAgMgt.value = Object.keys(tree);
                selectAgMgt.value = (prev.agMgt && availableAgMgt.value.includes(prev.agMgt))
                    ? prev.agMgt : (availableAgMgt.value[0] || '');
                availableWater.value = Object.keys(tree[selectAgMgt.value] || {});
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');
                availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
                await ensureComboLayer(mapRegister[key].layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);

            } else if (cat === "Non-Ag") {
                // tree is a flat list of lu values
                availableAgMgt.value = [];
                availableWater.value = [];
                availableLanduse.value = Array.isArray(tree) ? tree : Object.keys(tree);
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
                await ensureComboLayer(mapRegister[key].layerPrefix, [selectLanduse.value]);
            }
        }

        onMounted(async () => {
            await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
            availableYears.value = window.Supporting_info.years;
            selectYear.value = availableYears.value[0] || 2020;

            const initCat = availableCategories[0]; // "Land-use"
            await ensureIndexLoaded(initCat);
            await cascadeAndLoad(initCat);
            selectCategory.value = initCat;
            dataLoaded.value = true;
        });

        onUnmounted(() => {
            window.MemoryService.cleanupViewData(VIEW_NAME);
        });

        watch(yearIndex, (newIndex) => {
            selectYear.value = availableYears.value[newIndex];
        });

        watch(selectCategory, async (newCat, oldCat) => {
            if (oldCat) {
                previousSelections.value[oldCat] = {
                    agMgt: selectAgMgt.value,
                    water: selectWater.value,
                    landuse: selectLanduse.value
                };
            }
            await ensureIndexLoaded(newCat);
            await cascadeAndLoad(newCat);
        });

        watch(selectAgMgt, async (newAgMgt) => {
            if (selectCategory.value !== "Ag Mgt") return;
            const tree = getTree("Ag Mgt");
            const prev = previousSelections.value["Ag Mgt"] || {};
            availableWater.value = Object.keys(tree[newAgMgt] || {});
            selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                ? prev.water : (availableWater.value[0] || '');
            availableLanduse.value = tree[newAgMgt]?.[selectWater.value] || [];
            selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                ? prev.landuse : (availableLanduse.value[0] || '');
            await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [newAgMgt, selectWater.value, selectLanduse.value]);
        });

        watch(selectWater, async (newWater) => {
            if (selectCategory.value === "Land-use") {
                await ensureComboLayer(mapRegister["Lumap"].layerPrefix, [newWater]);
            } else if (selectCategory.value === "Ag") {
                const tree = getTree("Ag");
                const prev = previousSelections.value["Ag"] || {};
                availableLanduse.value = tree[newWater] || [];
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
                await ensureComboLayer(mapRegister["Ag"].layerPrefix, [newWater, selectLanduse.value]);
            } else if (selectCategory.value === "Ag Mgt") {
                const tree = getTree("Ag Mgt");
                const prev = previousSelections.value["Ag Mgt"] || {};
                availableLanduse.value = tree[selectAgMgt.value]?.[newWater] || [];
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
                await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, newWater, selectLanduse.value]);
            }
        });

        watch(selectLanduse, async (newLanduse) => {
            if (selectCategory.value === "Ag") {
                await ensureComboLayer(mapRegister["Ag"].layerPrefix, [selectWater.value, newLanduse]);
            } else if (selectCategory.value === "Ag Mgt") {
                await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, newLanduse]);
            } else if (selectCategory.value === "Non-Ag") {
                await ensureComboLayer(mapRegister["Non-Ag"].layerPrefix, [newLanduse]);
            }
        });

        const _state = {
            yearIndex, selectYear, selectRegion,
            availableYears, availableCategories,
            availableAgMgt, availableWater, availableLanduse,
            selectCategory, selectAgMgt, selectWater, selectLanduse,
            selectMapData,
            dataLoaded, isLoadingData,
        };
        window._debug && (window._debug[VIEW_NAME] = _state);
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

            <!-- Ag Mgt buttons (Ag Mgt category only) -->
            <div v-if="dataLoaded && selectCategory === 'Ag Mgt' && availableAgMgt.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
                <span class="text-[0.8rem] mr-1 font-medium">Ag Mgt:</span>
                <button v-for="(val, key) in availableAgMgt" :key="key"
                    @click="selectAgMgt = val"
                    class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
                    :class="{'bg-sky-500 text-white': selectAgMgt === val}">
                    {{ val }}
                </button>
            </div>

            <!-- Water buttons (Land-use, Ag, Ag Mgt) -->
            <div v-if="dataLoaded && selectCategory !== 'Non-Ag' && availableWater.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
                <span class="text-[0.8rem] mr-1 font-medium">Water:</span>
                <button v-for="(val, key) in availableWater" :key="key"
                    @click="selectWater = val"
                    class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
                    :class="{'bg-sky-500 text-white': selectWater === val}">
                    {{ val }}
                </button>
            </div>

            <!-- Landuse buttons (Ag, Ag Mgt, Non-Ag) -->
            <div v-if="dataLoaded && selectCategory !== 'Land-use' && availableLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
                <span class="text-[0.8rem] mr-1 font-medium">Landuse:</span>
                <button v-for="(val, key) in availableLanduse" :key="key"
                    @click="selectLanduse = val"
                    class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
                    :class="{'bg-sky-500 text-white': selectLanduse === val}">
                    {{ val }}
                </button>
            </div>
        </div>

        <!-- Map container -->
        <div style="position: relative; width: 100%; height: 100%;">

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

          <regions-map
            :mapData="selectMapData"
            style="width: 100%; height: 100%;">
          </regions-map>
        </div>

    </div>
    `
};
