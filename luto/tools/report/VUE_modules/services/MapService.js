// Map Data Service
// This service provides data about maps for different regions and categories

window.MapService = {

  mapCategories: {
    'Area': {
      'Ag': { 'path': 'data/map_layers/map_area_Ag.js', 'name': 'map_area_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_area_Am.js', 'name': 'map_area_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_area_NonAg.js', 'name': 'map_area_NonAg' },
    },
    'Biodiversity': {
      'GBF2': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF2_Ag.js', 'name': 'map_bio_GBF2_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF2_Am.js', 'name': 'map_bio_GBF2_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF2_NonAg.js', 'name': 'map_bio_GBF2_NonAg' },
      },
      'GBF3_NVIS': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF3_NVIS_Ag.js', 'name': 'map_bio_GBF3_NVIS_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF3_NVIS_Am.js', 'name': 'map_bio_GBF3_NVIS_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF3_NVIS_NonAg.js', 'name': 'map_bio_GBF3_NVIS_NonAg' },
      },
      'GBF3_IBRA': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF3_IBRA_Ag.js', 'name': 'map_bio_GBF3_IBRA_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF3_IBRA_Am.js', 'name': 'map_bio_GBF3_IBRA_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF3_IBRA_NonAg.js', 'name': 'map_bio_GBF3_IBRA_NonAg' },
      },
      'GBF4_ECNES': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF4_ECNES_Ag.js', 'name': 'map_bio_GBF4_ECNES_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF4_ECNES_Am.js', 'name': 'map_bio_GBF4_ECNES_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF4_ECNES_NonAg.js', 'name': 'map_bio_GBF4_ECNES_NonAg' },
      },
      'GBF4_SNES': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF4_SNES_Ag.js', 'name': 'map_bio_GBF4_SNES_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF4_SNES_Am.js', 'name': 'map_bio_GBF4_SNES_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF4_SNES_NonAg.js', 'name': 'map_bio_GBF4_SNES_NonAg' },
      },
      'GBF8_GROUP': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF8_groups_Ag.js', 'name': 'map_bio_GBF8_groups_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF8_groups_Am.js', 'name': 'map_bio_GBF8_groups_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF8_groups_NonAg.js', 'name': 'map_bio_GBF8_groups_NonAg' },
      },
      'GBF8_SPECIES': {
        'Ag': { 'path': 'data/map_layers/map_bio_GBF8_Ag.js', 'name': 'map_bio_GBF8_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_GBF8_Am.js', 'name': 'map_bio_GBF8_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_GBF8_NonAg.js', 'name': 'map_bio_GBF8_NonAg' },
      },
      'quality': {
        'Sum': { 'path': 'data/map_layers/map_bio_overall_All.js', 'name': 'map_bio_overall_All' },
        'Ag': { 'path': 'data/map_layers/map_bio_overall_Ag.js', 'name': 'map_bio_overall_Ag' },
        'Ag Mgt': { 'path': 'data/map_layers/map_bio_overall_Am.js', 'name': 'map_bio_overall_Am' },
        'Non-Ag': { 'path': 'data/map_layers/map_bio_overall_NonAg.js', 'name': 'map_bio_overall_NonAg' },
      }
    },
    'Dvar': {
      'Ag': { 'path': 'data/map_layers/map_dvar_Ag.js', 'name': 'map_dvar_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_dvar_Am.js', 'name': 'map_dvar_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_dvar_NonAg.js', 'name': 'map_dvar_NonAg' },
      'Lumap': { 'path': 'data/map_layers/map_dvar_lumap.js', 'name': 'map_dvar_lumap' },
    },
    'Economics': {
      'Ag': {
        'Profit':                { 'path': 'data/map_layers/map_economics_Ag_profit.js',               'name': 'map_economics_Ag_profit' },
        'Revenue':               { 'path': 'data/map_layers/map_economics_Ag_revenue.js',              'name': 'map_economics_Ag_revenue' },
        'Cost':                  { 'path': 'data/map_layers/map_economics_Ag_cost.js',                 'name': 'map_economics_Ag_cost' },
        'Transition (Ag2Ag)':    { 'path': 'data/map_layers/map_economics_Ag_transition_ag2ag.js',    'name': 'map_economics_Ag_transition_ag2ag' },
        'Transition (Ag2NonAg)': { 'path': 'data/map_layers/map_economics_Ag_transition_ag2nonag.js', 'name': 'map_economics_Ag_transition_ag2nonag' },
      },
      'Ag Mgt': {
        'Profit':  { 'path': 'data/map_layers/map_economics_Am_profit.js',  'name': 'map_economics_Am_profit' },
        'Revenue': { 'path': 'data/map_layers/map_economics_Am_revenue.js', 'name': 'map_economics_Am_revenue' },
        'Cost':    { 'path': 'data/map_layers/map_economics_Am_cost.js',    'name': 'map_economics_Am_cost' },
      },
      'Non-Ag': {
        'Profit':  { 'path': 'data/map_layers/map_economics_NonAg_profit.js',  'name': 'map_economics_NonAg_profit' },
        'Revenue': { 'path': 'data/map_layers/map_economics_NonAg_revenue.js', 'name': 'map_economics_NonAg_revenue' },
        'Cost':    { 'path': 'data/map_layers/map_economics_NonAg_cost.js',    'name': 'map_economics_NonAg_cost' },
      },
    },
    'GHG': {
      'Ag': { 'path': 'data/map_layers/map_GHG_Ag.js', 'name': 'map_GHG_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_GHG_Am.js', 'name': 'map_GHG_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_GHG_NonAg.js', 'name': 'map_GHG_NonAg' },
    },
    'Production': {
      'Ag': { 'path': 'data/map_layers/map_quantities_Ag.js', 'name': 'map_quantities_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_quantities_Am.js', 'name': 'map_quantities_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_quantities_NonAg.js', 'name': 'map_quantities_NonAg' },
    },
    'Water': {
      'Ag': { 'path': 'data/map_layers/map_water_yield_Ag.js', 'name': 'map_water_yield_Ag' },
      'Ag Mgt': { 'path': 'data/map_layers/map_water_yield_Am.js', 'name': 'map_water_yield_Am' },
      'Non-Ag': { 'path': 'data/map_layers/map_water_yield_NonAg.js', 'name': 'map_water_yield_NonAg' },
    }
  }
}

