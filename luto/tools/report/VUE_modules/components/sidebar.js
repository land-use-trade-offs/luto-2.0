window.Sidebar = {
  emits: ['update:isCollapsed'],
  setup(props, { emit }) {
    const { ref, computed } = Vue;
    // Function to standardize SVG icons for consistent display
    const standardizeIcon = (icon) => {
      // Add a fixed viewBox to ensure consistent sizing
      return icon.replace('<svg', '<svg width="24" height="24"');
    };

    const navItems = [
      { id: "home", label: "Home", path: "/", icon: standardizeIcon(window.NavIcons.home) },
      { id: "area", label: "Area Analysis", path: "/area", icon: standardizeIcon(window.NavIcons.area) },
      { id: "production", label: "Production Analysis", path: "/production", icon: standardizeIcon(window.NavIcons.production) },
      { id: "economy", label: "Economics", path: "/economics", icon: standardizeIcon(window.NavIcons.economy) },
      { id: "GHG", label: "GHG Analysis", path: "/ghg", icon: standardizeIcon(window.NavIcons.GHG) },
      { id: "water", label: "Water Analysis", path: "/water", icon: standardizeIcon(window.NavIcons.water) },
      { id: "biodiversity", label: "Biodiversity", path: "/biodiversity", icon: standardizeIcon(window.NavIcons.biodiversity) },
      { id: "map", label: "Map View", path: "/map", icon: standardizeIcon(window.NavIcons.map) },
      { id: "settings", label: "Settings and Log", path: "/settings", icon: standardizeIcon(window.NavIcons.settings) },
    ];

    const CommonIcons = {
      Expand: standardizeIcon(window.CommonIcons.Expand),
      Collapse: standardizeIcon(window.CommonIcons.Collapse),
    }

    const isCollapsed = ref(false);

    const toggleCollapse = () => {
      isCollapsed.value = !isCollapsed.value;
      emit('update:isCollapsed', isCollapsed.value);
    };

    // Get route info for highlighting active menu item
    const route = VueRouter.useRoute();
    const activeIndex = computed(() => {
      return route.path;
    });

    return {
      navItems,
      CommonIcons,
      isCollapsed,
      toggleCollapse,
      activeIndex,
    };
  },

  template: `
    <div>

      <div class="flex items-center h-[80px]">
        <!-- Logo -->
        <div v-if="!isCollapsed" class="flex-1 flex items-center transition-opacity duration-300">
          <img class="rounded-full w-10 h-10" src="resources/LUTO.png" alt="LUTO 2.0" />
          <span class="ml-2 text-sm font-semibold">LUTO 2.0</span>
        </div>
        <!-- Toggle button -->
        <div class="w-6 h-6 items-center ml-2 cursor-pointer" @click="toggleCollapse">
          <span v-if="!isCollapsed" v-html="CommonIcons.Collapse"></span>
          <span v-else  v-html="CommonIcons.Expand"></span>
        </div>
      </div>

      <!-- Menu -->
      <nav>
        <ul class="transition-all duration-300 ease-in-out">
          <li v-for="item in navItems" :key="item.id">
            <router-link :to="item.path" class="flex flex-nowrap ml-2 py-3 cursor-pointer" 
              :class="{ 'bg-gray-50 border-l-4 border-blue-700': activeIndex === item.path }">
              <span v-html="item.icon" class="w-6 h-6 items-center justify-center"></span>
              <span v-if="!isCollapsed" class="ml-2 text-sm w-[180px]">{{ item.label }}</span>
            </router-link>
          </li>
        </ul>
      </nav>
    </div>
  `,
};
