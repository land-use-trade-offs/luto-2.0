window['MapView'] = {
    name: 'MapView',
    setup() {
        const { inject, ref, watch, onMounted, onUnmounted, computed, nextTick } = Vue;

        // Data|Map service
        const mapRegister = window.MapService.mapCategories["Dvar"];        // MapService was registered in the index.html        [MapService.js]
        const loadScript = window.loadScriptWithTracking;

        // View identification for memory management
        const VIEW_NAME = "Map";

        // Global selection state
        const yearIndex = ref(0);
        const selectYear = ref(2020);
        const selectRegion = inject("globalSelectedRegion");

        // Available variables
        const availableYears = ref([]);

        // Available selections
        // "Land-use": lumap [Water] → [year]
        // "Ag":       dvar_Ag [Water] → [LU] → [year]
        // "Ag Mgt":   dvar_Am [AgMgt] → [Water] → [LU] → [year]
        // "Non-Ag":   dvar_NonAg [LU] → [year]
        const availableCategories = ["Land-use", "Ag", "Ag Mgt", "Non-Ag"];
        const availableAgMgt = ref([]);
        const availableWater = ref([]);
        const availableLanduse = ref([]);

        // Selection state
        const selectCategory = ref("");
        const selectAgMgt = ref("");
        const selectWater = ref("");
        const selectLanduse = ref("");

        // Previous selections memory
        const previousSelections = ref({
            "Land-use": { water: "" },
            "Ag":       { water: "", landuse: "" },
            "Ag Mgt":   { agMgt: "", water: "", landuse: "" },
            "Non-Ag":   { landuse: "" }
        });

        // UI state
        const dataLoaded = ref(false);

        // Raw data refs (set in onMounted)
        const lumap   = ref(null);
        const agData  = ref(null);
        const amData  = ref(null);
        const nonAgData = ref(null);

        const mapReady = computed(() => {
            if (!selectCategory.value) return false;
            if (selectCategory.value === "Land-use") return !!(lumap.value && selectWater.value);
            if (selectCategory.value === "Ag")       return !!(agData.value  && selectWater.value && selectLanduse.value);
            if (selectCategory.value === "Ag Mgt")   return !!(amData.value  && selectAgMgt.value && selectWater.value && selectLanduse.value);
            if (selectCategory.value === "Non-Ag")   return !!(nonAgData.value && selectLanduse.value);
            return false;
        });

        const selectMapData = computed(() => {
            if (!mapReady.value) return {};
            const year = selectYear.value;
            if (selectCategory.value === "Land-use")
                return lumap.value?.[selectWater.value]?.[year] || {};
            if (selectCategory.value === "Ag")
                return agData.value?.[selectWater.value]?.[selectLanduse.value]?.[year] || {};
            if (selectCategory.value === "Ag Mgt")
                return amData.value?.[selectAgMgt.value]?.[selectWater.value]?.[selectLanduse.value]?.[year] || {};
            if (selectCategory.value === "Non-Ag")
                return nonAgData.value?.[selectLanduse.value]?.[year] || {};
            return {};
        });

        // Legend data — only show for categorical (int) layers, not float
        const selectLegend = computed(() => {
            const mapObj = selectMapData.value;
            if (!mapObj || mapObj.intOrFloat !== 'int') return {};
            return mapObj.legend || {};
        });

        onMounted(async () => {
            await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
            await loadScript(mapRegister["Ag"]['path'],     mapRegister["Ag"]['name'],     VIEW_NAME);
            await loadScript(mapRegister["Ag Mgt"]['path'], mapRegister["Ag Mgt"]['name'], VIEW_NAME);
            await loadScript(mapRegister["Non-Ag"]['path'], mapRegister["Non-Ag"]['name'], VIEW_NAME);
            await loadScript(mapRegister["Lumap"]['path'],  mapRegister["Lumap"]['name'],  VIEW_NAME);

            lumap.value    = window[mapRegister["Lumap"]['name']];
            agData.value   = window[mapRegister["Ag"]['name']];
            amData.value   = window[mapRegister["Ag Mgt"]['name']];
            nonAgData.value = window[mapRegister["Non-Ag"]['name']];

            // Initial selections
            availableYears.value = window.Supporting_info.years;
            selectCategory.value = availableCategories[0];

            await nextTick(() => {
                dataLoaded.value = true;
            });
        });

        // Memory cleanup on component unmount
        onUnmounted(() => {
            window.MemoryService.cleanupViewData(VIEW_NAME);
        });

        // Watchers
        watch(yearIndex, (newIndex) => {
            selectYear.value = availableYears.value[newIndex];
        });

        // Category → cascade all downstream
        watch(selectCategory, (newCat, oldCat) => {
            if (oldCat) {
                previousSelections.value[oldCat] = {
                    agMgt: selectAgMgt.value,
                    water: selectWater.value,
                    landuse: selectLanduse.value
                };
            }
            const prev = previousSelections.value[newCat] || {};

            if (newCat === "Land-use") {
                availableAgMgt.value   = [];
                availableWater.value   = Object.keys(lumap.value || {});
                availableLanduse.value = [];
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');

            } else if (newCat === "Ag") {
                availableAgMgt.value   = [];
                availableWater.value   = Object.keys(agData.value || {});
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');
                availableLanduse.value = Object.keys(agData.value?.[selectWater.value] || {});
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');

            } else if (newCat === "Ag Mgt") {
                availableAgMgt.value = Object.keys(amData.value || {});
                selectAgMgt.value = (prev.agMgt && availableAgMgt.value.includes(prev.agMgt))
                    ? prev.agMgt : (availableAgMgt.value[0] || '');
                availableWater.value = Object.keys(amData.value?.[selectAgMgt.value] || {});
                selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                    ? prev.water : (availableWater.value[0] || '');
                availableLanduse.value = Object.keys(amData.value?.[selectAgMgt.value]?.[selectWater.value] || {});
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');

            } else if (newCat === "Non-Ag") {
                availableAgMgt.value   = [];
                availableWater.value   = [];
                availableLanduse.value = Object.keys(nonAgData.value || {});
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
            }
        });

        // AgMgt → Water → LU  (Ag Mgt only)
        watch(selectAgMgt, (newAgMgt) => {
            if (selectCategory.value !== "Ag Mgt") return;
            const prev = previousSelections.value["Ag Mgt"] || {};
            availableWater.value = Object.keys(amData.value?.[newAgMgt] || {});
            selectWater.value = (prev.water && availableWater.value.includes(prev.water))
                ? prev.water : (availableWater.value[0] || '');
            availableLanduse.value = Object.keys(amData.value?.[newAgMgt]?.[selectWater.value] || {});
            selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                ? prev.landuse : (availableLanduse.value[0] || '');
        });

        // Water → LU  (Ag and Ag Mgt)
        watch(selectWater, (newWater) => {
            if (selectCategory.value === "Ag") {
                const prev = previousSelections.value["Ag"] || {};
                availableLanduse.value = Object.keys(agData.value?.[newWater] || {});
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
            } else if (selectCategory.value === "Ag Mgt") {
                const prev = previousSelections.value["Ag Mgt"] || {};
                availableLanduse.value = Object.keys(amData.value?.[selectAgMgt.value]?.[newWater] || {});
                selectLanduse.value = (prev.landuse && availableLanduse.value.includes(prev.landuse))
                    ? prev.landuse : (availableLanduse.value[0] || '');
            }
        });

        const _state = {
            yearIndex,
            selectYear,
            selectRegion,

            availableYears,
            availableCategories,
            availableAgMgt,
            availableWater,
            availableLanduse,

            selectCategory,
            selectAgMgt,
            selectWater,
            selectLanduse,

            selectMapData,
            selectLegend,

            dataLoaded,
            mapReady,
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
        <regions-map
            :mapData="selectMapData"
            style="width: 100%; height: 100%;">
        </regions-map>

        <!-- Legend -->
        <div v-if="selectLegend && Object.keys(selectLegend).length > 0" class="absolute top-[20px] right-[20px] z-[1001] bg-white/70 p-3 rounded-lg max-w-[300px]">
            <div class="font-bold text-sm mb-2 text-gray-600">{{ selectCategory }}</div>
            <div class="flex flex-col space-y-1">
                <div v-for="(color, label) in selectLegend" :key="label" class="flex items-center">
                    <span class="inline-block w-[12px] h-[12px] mr-[6px]" :style="{ backgroundColor: color }"></span>
                    <span class="text-xs text-gray-600">{{ label }}</span>
                </div>
            </div>
        </div>

    </div>
    `
};
