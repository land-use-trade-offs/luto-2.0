const { createApp } = Vue;

// Initialize app
const app = createApp({
    setup() {
        const { ref, provide } = Vue;
        const isCollapsed = ref(false);
        const globalSelectedRegion = ref('AUSTRALIA');
        const globalSelectedDataType = ref('Area');
        const globalMapViewpoint = ref({
            center: [-26, 126.5],
            zoom: 5,
            lastSelectedRegion: 'AUSTRALIA'
        });

        const updateSidebarCollapsed = (value) => {
            isCollapsed.value = value;
        };

        provide('isCollapsed', isCollapsed);
        provide('globalSelectedRegion', globalSelectedRegion);
        provide('globalSelectedDataType', globalSelectedDataType);
        provide('globalMapViewpoint', globalMapViewpoint);

        return {
            updateSidebarCollapsed,
            isCollapsed,
            globalSelectedRegion,
            globalSelectedDataType,
            globalMapViewpoint,
        };
    },
    template: `
    <div class="flex">
        <!-- Sidebar -->
        <div  class="bg-white pl-2 w-min-[50px] transform transition-all duration-300 ease-in-out"
            :class="{'w-[50px]': isCollapsed, 'w-[200px]': !isCollapsed}">
          <side-bar @update:isCollapsed="updateSidebarCollapsed"></side-bar>
        </div>
        <!-- Main content -->
        <div class="flex-1 bg-[#f8f9fe] mr-4 pl-4">
          <router-view></router-view>
        </div>
    </div>
    `
});

// Register other components
app.component("chart-container", window.Highchart);
app.component("side-bar", window.Sidebar);
app.component('map-geojson', window.map_geojson);
app.component('ranking-cards', window.RankingCards);
app.component('filterable-dropdown', window.FilterableDropdown);
app.component('regions-map', window.RegionsMap);

// Use modules
app.use(ElementPlus);
app.use(window.router);

// Mount the app
app.mount("#app");
