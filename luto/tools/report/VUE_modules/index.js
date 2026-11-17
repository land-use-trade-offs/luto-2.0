const { createApp } = Vue;

// Initialize app
const app = createApp({
    setup() {
        const { ref, provide } = Vue;
        const isCollapsed = ref(false);
        const globalSelectedRegion = ref('AUSTRALIA');
        const globalMapViewpoint = ref({
            center: window.AUSTRALIA_VIEW.center,          // helpers.js: the centre every map shares
            zoom: 5,
            lastSelectedRegion: 'AUSTRALIA'
        });

        const updateSidebarCollapsed = (value) => {
            isCollapsed.value = value;
        };

        provide('isCollapsed', isCollapsed);
        provide('globalSelectedRegion', globalSelectedRegion);
        provide('globalMapViewpoint', globalMapViewpoint);

        return {
            updateSidebarCollapsed,
            isCollapsed,
            globalSelectedRegion,
            globalMapViewpoint,
        };
    },
    template: /*html*/`
    <div class="flex">
        <!-- Sidebar -->
        <div  class="bg-white pl-2 shrink-0 sticky top-0 h-screen self-start overflow-y-auto transition-all duration-300 ease-in-out"
            :class="{'w-[50px]': isCollapsed, 'w-[240px]': !isCollapsed}">
          <side-bar @update:isCollapsed="updateSidebarCollapsed"></side-bar>
        </div>
        <!-- Main content -->
        <div class="flex-1 min-w-0 bg-[#f8f9fe] mr-4 pl-4">
          <router-view></router-view>
        </div>
    </div>
    `
});

// Register other components
app.component("chart-container", window.Highchart);
app.component("heatmap-container", window.HeatmapContainer);
app.component("side-bar", window.Sidebar);
app.component('map-geojson', window.map_geojson);
app.component('headline-cards', window.HeadlineCards);
app.component('chart-cards', window.ChartCards);
app.component('filterable-dropdown', window.FilterableDropdown);
app.component('regions-map', window.RegionsMap);

// Use modules
app.use(ElementPlus);
app.use(window.router);

// Mount the app
app.mount("#app");
