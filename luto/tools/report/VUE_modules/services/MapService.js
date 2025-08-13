// Map Data Service
// This service provides data about maps for different regions and categories

window.MapService = {

  mapCategories: {
    'Area': {
      'Ag': 'data/map_layers/map_area_Ag.js',
      'Ag Mgt': 'data/map_layers/map_area_Am.js',
      'Non-Ag': 'data/map_layers/map_area_NonAg.js',
    },
    'Production': {
      'Ag': 'data/map_layers/map_quantities_Ag.js',
      'Ag Mgt': 'data/map_layers/map_quantities_Am.js',
      'Non-Ag': 'data/map_layers/map_quantities_NonAg.js',
    },
    'Biodiversity': {
      'overall': {
        'Ag': 'data/map_layers/map_bio_overall_Ag.js',
        'Ag Mgt': 'data/map_layers/map_bio_overall_Am.js',
        'Non-Ag': 'data/map_layers/map_bio_overall_NonAg.js',
      },
      'GBF2': {
        'Ag': 'data/map_layers/map_bio_GBF2_Ag.js',
        'Ag Mgt': 'data/map_layers/map_bio_GBF2_Am.js',
        'Non-Ag': 'data/map_layers/map_bio_GBF2_NonAg.js',
      }
    },
    'Economics': {
      'Cost': {
        'Ag': 'data/map_layers/map_cost_Ag.js',
        'Ag Mgt': 'data/map_layers/map_cost_Am.js',
        'Non-Ag': 'data/map_layers/map_cost_NonAg.js',
      },
      'Revenue': {
        'Ag': 'data/map_layers/map_revenue_Ag.js',
        'Ag Mgt': 'data/map_layers/map_revenue_Am.js',
        'Non-Ag': 'data/map_layers/map_revenue_NonAg.js',
      },
    },
    'GHG': {
      'Ag': 'data/map_layers/map_GHG_Ag.js',
      'Ag Mgt': 'data/map_layers/map_GHG_Am.js',
      'Non-Ag': 'data/map_layers/map_GHG_NonAg.js',
    },
    'Water': {
      'Ag': 'data/map_layers/map_water_yield_Ag.js',
      'Ag Mgt': 'data/map_layers/map_water_yield_Am.js',
      'Non-Ag': 'data/map_layers/map_water_yield_NonAg.js',
    }
  }
}


