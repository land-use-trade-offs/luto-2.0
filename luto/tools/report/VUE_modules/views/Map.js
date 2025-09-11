window['MapView'] = {
    setup() {
        const { inject, ref, watch, onMounted, onUnmounted, computed, nextTick } = Vue;

        // Data|Map service
        const mapRegister = window.MapService.mapCategories["Dvar"];        // MapService was registered in the index.html        [MapService.js]
        const loadScript = window.loadScriptWithTracking;
        
        // View identification for memory management
        const VIEW_NAME = "Map";                               // DataConstructor has been registered in index.html  [helpers.js]

        // Global selection state
        const yearIndex = ref(0);
        const selectYear = ref(2020);
        const selectRegion = inject("globalSelectedRegion");

        // Available variables
        const availableYears = ref([]);

        // Available selections
        const availableCategories = ["Land-use", "Water-supply", "Ag", "Ag Mgt", "Non-Ag"];
        const availableLanduse = ref([]);

        // Selection state
        const selectCategory = ref("");
        const selectLanduse = ref("");

        // Previous selections memory
        const previousSelections = ref({
            "Land-use": { landuse: "" },
            "Water-supply": { landuse: "" },
            "Ag": { landuse: "" },
            "Ag Mgt": { landuse: "" },
            "Non-Ag": { landuse: "" }
        });

        // UI state
        const dataLoaded = ref(false);
        const mapReady = computed(() => {
            if (!selectCategory.value || !selectLanduse.value) {
                return false;
            }
            return dvarMaps.value && dvarMaps.value[selectCategory.value];
        });

        // Reactive data
        const dvarMaps = ref({});
        const selectMapData = computed(() => {
            if (!mapReady.value) {
                return {};
            }
            return dvarMaps.value[selectCategory.value][selectLanduse.value][selectYear.value] || {};
        });

        // Legend data
        const selectLegend = computed(() => {
            if (!mapReady.value) {
                return {};
            }

            // Check if the currently selected map object has a legend
            const currentMapData = dvarMaps.value[selectCategory.value][selectLanduse.value][selectYear.value];

            if (currentMapData && currentMapData.legend) {
                return currentMapData.legend;
            }

            // No legend available for this map object
            return {};
        });

        onMounted(async () => {
            await loadScript("./data/Supporting_info.js", "Supporting_info", VIEW_NAME);
            await loadScript(mapRegister["Ag"]['path'], mapRegister["Ag"]['name'], VIEW_NAME);
            await loadScript(mapRegister["Ag Mgt"]['path'], mapRegister["Ag Mgt"]['name'], VIEW_NAME);
            await loadScript(mapRegister["Non-Ag"]['path'], mapRegister["Non-Ag"]['name'], VIEW_NAME);
            await loadScript(mapRegister["Mosaic"]['path'], mapRegister["Mosaic"]['name'], VIEW_NAME);

            dvarMaps.value = {
                'Land-use': { 'Land-use': window[mapRegister["Mosaic"]['name']]['Land-use'] },
                'Water-supply': { 'Water-supply': window[mapRegister["Mosaic"]['name']]['Water-supply'] },
                'Ag': {
                    'ALL': window[mapRegister["Mosaic"]['name']]['Agricultural Land-use'],
                    ...window[mapRegister["Ag"]['name']]
                },
                'Ag Mgt': {
                    'ALL': window[mapRegister["Mosaic"]['name']]['Agricultural Management'],
                    ...window[mapRegister["Ag Mgt"]['name']]
                },
                'Non-Ag': {
                    'ALL': window[mapRegister["Mosaic"]['name']]['Non-agricultural Land-use'],
                    ...window[mapRegister["Non-Ag"]['name']]
                }
            };

            // Initial selections
            availableYears.value = window.Supporting_info.years;
            selectCategory.value = availableCategories[0];

            await nextTick(() => {
                dataLoaded.value = true;
            });
        });

        // Watchers
        watch(yearIndex, (newIndex) => {
            selectYear.value = availableYears.value[newIndex];
        });

        // Progressive selection watcher
        watch(selectCategory, (newCategory, oldCategory) => {
            // Save previous selections before switching
            if (oldCategory) {
                previousSelections.value[oldCategory] = { landuse: selectLanduse.value };
            }

            if (newCategory && dvarMaps.value[newCategory]) {
                availableLanduse.value = Object.keys(dvarMaps.value[newCategory] || {});
                const prevLanduse = previousSelections.value[newCategory].landuse;
                selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || '');
            }
        });

        watch(selectLanduse, (newLanduse) => {
            // Save current landuse selection
            if (selectCategory.value) {
                previousSelections.value[selectCategory.value].landuse = newLanduse;
            }
        });

        return {
            yearIndex,
            selectYear,
            selectRegion,

            availableYears,
            availableCategories,
            availableLanduse,

            selectCategory,
            selectLanduse,

            selectMapData,
            selectLegend,

            dataLoaded,
            mapReady,
        };

        // Memory cleanup on component unmount
        onUnmounted(() => {
            window.MemoryService.cleanupViewData(VIEW_NAME);
        });

        return {
            yearIndex,
            selectYear,
            selectRegion,

            availableYears,
            availableCategories,
            availableLanduse,

            selectCategory,
            selectLanduse,

            selectMapData,
            selectLegend,

            dataLoaded,
            mapReady,
        };
    },
    template: `
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
                :show-tooltip="false"
                :min="0"
                :max="availableYears.length - 1"
                :step="1"
                :format-tooltip="index => availableYears[index]"
                :marks="availableYears.reduce((acc, year, index) => ({ ...acc, [index]: year }), {})"
                @input="(index) => { yearIndex = index; selectYear = availableYears[index]; }"
            />
        </div>

        <!-- Data selection controls container -->
        <div class="absolute top-[285px] left-[20px] w-[320px] z-[1001] flex flex-col space-y-3 bg-white/70 p-2 rounded-lg">

            <!-- Category buttons -->
            <div class="flex items-center">
                <div class="flex space-x-1">
                    <span class="text-[0.8rem] mr-1 font-medium">Category:</span>
                    <button v-for="(val, key) in availableCategories" :key="key"
                        @click="selectCategory = val"
                        class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded"
                        :class="{'bg-sky-500 text-white': selectCategory === val}">
                        {{ val }}
                    </button>
                </div>
            </div>

            <!-- Landuse options -->
            <div class="flex items-start border-t border-white/10 pt-1">
                <div v-if="dataLoaded && availableLanduse.length > 0" class="flex flex-wrap gap-1 max-w-[300px]">
                    <span class="text-[0.8rem] mr-1 font-medium">{{ selectCategory === 'Ag Mgt' ? 'Ag Mgt' : 'Landuse' }}:</span>
                    <button v-for="(val, key) in availableLanduse" :key="key"
                        @click="selectLanduse = val"
                        class="bg-white text-[#1f1f1f] text-[0.6rem] px-1 py-1 rounded mb-1"
                        :class="{'bg-sky-500 text-white': selectLanduse === val}">
                        {{ val }}
                    </button>
                </div>
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