// Map Data Service
// This service provides data about maps for different regions and categories

window.MapService = {

  mapCategories: {
    'Area': {
      'Ag': { 'path': 'data/map_layers/map_area_Ag.js', 'name': 'map_area_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_area_Am.js', 'name': 'map_area_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_area_NonAg.js', 'name': 'map_area_NonAg' },
    },
    'Production': {
      'Ag': { 'path': 'data/map_layers/map_quantities_Ag.js', 'name': 'map_quantities_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_quantities_Am.js', 'name': 'map_quantities_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_quantities_NonAg.js', 'name': 'map_quantities_NonAg' },
    },
    'Biodiversity': {
      'overall': {
        'Ag': { 'path': 'data/map_layers/map_bio_overall_Ag.js', 'name': 'map_bio_overall_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_overall_Am.js', 'name': 'map_bio_overall_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_overall_NonAg.js', 'name': 'map_bio_overall_NonAg' },
      },
      'GBF2': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF2_Ag.js', 'name': 'map_bio_GBF2_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF2_Am.js', 'name': 'map_bio_GBF2_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF2_NonAg.js', 'name': 'map_bio_GBF2_NonAg' },
      }
    },
    'Economics': {
      'Cost': {
        'Ag': { 'path': 'data/map_layers/map_cost_Ag.js', 'name': 'map_cost_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_cost_Am.js', 'name': 'map_cost_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_cost_NonAg.js', 'name': 'map_cost_NonAg' },
      },
      'Revenue': {
        'Ag': { 'path': 'data/map_layers/map_revenue_Ag.js', 'name': 'map_revenue_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_revenue_Am.js', 'name': 'map_revenue_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_revenue_NonAg.js', 'name': 'map_revenue_NonAg' },
      },
    },
    'GHG': {
      'Ag': { 'path': 'data/map_layers/map_GHG_Ag.js', 'name': 'map_GHG_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_GHG_Am.js', 'name': 'map_GHG_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_GHG_NonAg.js', 'name': 'map_GHG_NonAg' },
    },
    'Water': {
      'Ag': { 'path': 'data/map_layers/map_water_yield_Ag.js', 'name': 'map_water_yield_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_water_yield_Am.js', 'name': 'map_water_yield_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_water_yield_NonAg.js', 'name': 'map_water_yield_NonAg' },
    }
  }
}


