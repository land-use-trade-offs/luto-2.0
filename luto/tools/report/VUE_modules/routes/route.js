const { createRouter, createWebHashHistory } = VueRouter;

// Define routes
const routes = [
  { path: "/", component: window.HomeView },
  { path: "/area", component: window.AreaView },
  { path: "/production", component: window.ProductionView },
  { path: "/economics", component: window.EconomicsView },
  { path: "/ghg", component: window.GHGView },
  { path: "/water", component: window.WaterView },
  { path: "/settings", component: window.SettingsView },
  { path: "/test", component: window.Test },
  { path: "/:pathMatch(.*)*", component: window.NotFound },
];

// Create router instance
window.router = createRouter({
  history: createWebHashHistory(),
  routes,
});
