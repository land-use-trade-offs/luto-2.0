// Map Data Service
// This service provides data about maps for different regions and categories
// All entries use the split-file pattern:
//   indexPath / indexName  → load the index JS that lists valid dim combos
//   layerPrefix            → base variable name; individual combo files are
//                            <layerPrefix>__<safe(dim1)>__…__<safe(dimN)>.js
//
// Exception: 'mask' entries remain as {path, name} (GeoJSON overlay, not split).

window.MapService = {

  mapCategories: {
    'Area': {
      'Ag': { 'indexPath': 'data/map_layers/map_area_Ag__index.js', 'indexName': 'map_area_Ag__index', 'layerPrefix': 'map_area_Ag' },
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_area_Am__index.js', 'indexName': 'map_area_Am__index', 'layerPrefix': 'map_area_Am' },
      'Non-Ag': { 'indexPath': 'data/map_layers/map_area_NonAg__index.js', 'indexName': 'map_area_NonAg__index', 'layerPrefix': 'map_area_NonAg' },
    },
    'Biodiversity': {
      'GBF2': {
        'Sum': { 'indexPath': 'data/map_layers/map_bio_GBF2_Sum__index.js', 'indexName': 'map_bio_GBF2_Sum__index', 'layerPrefix': 'map_bio_GBF2_Sum' },
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF2_Ag__index.js', 'indexName': 'map_bio_GBF2_Ag__index', 'layerPrefix': 'map_bio_GBF2_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF2_Am__index.js', 'indexName': 'map_bio_GBF2_Am__index', 'layerPrefix': 'map_bio_GBF2_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF2_NonAg__index.js', 'indexName': 'map_bio_GBF2_NonAg__index', 'layerPrefix': 'map_bio_GBF2_NonAg' },
        'mask': { 'path': 'data/geo/biodiversity_GBF2_mask.js', 'name': 'BIO_GBF2_MASK' },
      },
      'GBF3_NVIS': {
        'Sum': { 'indexPath': 'data/map_layers/map_bio_GBF3_NVIS_Sum__index.js', 'indexName': 'map_bio_GBF3_NVIS_Sum__index', 'layerPrefix': 'map_bio_GBF3_NVIS_Sum' },
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF3_NVIS_Ag__index.js', 'indexName': 'map_bio_GBF3_NVIS_Ag__index', 'layerPrefix': 'map_bio_GBF3_NVIS_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF3_NVIS_Am__index.js', 'indexName': 'map_bio_GBF3_NVIS_Am__index', 'layerPrefix': 'map_bio_GBF3_NVIS_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF3_NVIS_NonAg__index.js', 'indexName': 'map_bio_GBF3_NVIS_NonAg__index', 'layerPrefix': 'map_bio_GBF3_NVIS_NonAg' },
      },
      'GBF4_ECNES': {
        'Sum': { 'indexPath': 'data/map_layers/map_bio_GBF4_ECNES_Sum__index.js', 'indexName': 'map_bio_GBF4_ECNES_Sum__index', 'layerPrefix': 'map_bio_GBF4_ECNES_Sum' },
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF4_ECNES_Ag__index.js', 'indexName': 'map_bio_GBF4_ECNES_Ag__index', 'layerPrefix': 'map_bio_GBF4_ECNES_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF4_ECNES_Am__index.js', 'indexName': 'map_bio_GBF4_ECNES_Am__index', 'layerPrefix': 'map_bio_GBF4_ECNES_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF4_ECNES_NonAg__index.js', 'indexName': 'map_bio_GBF4_ECNES_NonAg__index', 'layerPrefix': 'map_bio_GBF4_ECNES_NonAg' },
      },
      'GBF4_SNES': {
        'Sum': { 'indexPath': 'data/map_layers/map_bio_GBF4_SNES_Sum__index.js', 'indexName': 'map_bio_GBF4_SNES_Sum__index', 'layerPrefix': 'map_bio_GBF4_SNES_Sum' },
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF4_SNES_Ag__index.js', 'indexName': 'map_bio_GBF4_SNES_Ag__index', 'layerPrefix': 'map_bio_GBF4_SNES_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF4_SNES_Am__index.js', 'indexName': 'map_bio_GBF4_SNES_Am__index', 'layerPrefix': 'map_bio_GBF4_SNES_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF4_SNES_NonAg__index.js', 'indexName': 'map_bio_GBF4_SNES_NonAg__index', 'layerPrefix': 'map_bio_GBF4_SNES_NonAg' },
      },
      'GBF8_GROUP': {
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF8_groups_Ag__index.js', 'indexName': 'map_bio_GBF8_groups_Ag__index', 'layerPrefix': 'map_bio_GBF8_groups_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF8_groups_Am__index.js', 'indexName': 'map_bio_GBF8_groups_Am__index', 'layerPrefix': 'map_bio_GBF8_groups_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF8_groups_NonAg__index.js', 'indexName': 'map_bio_GBF8_groups_NonAg__index', 'layerPrefix': 'map_bio_GBF8_groups_NonAg' },
      },
      'GBF8_SPECIES': {
        'Ag': { 'indexPath': 'data/map_layers/map_bio_GBF8_Ag__index.js', 'indexName': 'map_bio_GBF8_Ag__index', 'layerPrefix': 'map_bio_GBF8_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_GBF8_Am__index.js', 'indexName': 'map_bio_GBF8_Am__index', 'layerPrefix': 'map_bio_GBF8_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_GBF8_NonAg__index.js', 'indexName': 'map_bio_GBF8_NonAg__index', 'layerPrefix': 'map_bio_GBF8_NonAg' },
      },
      'quality': {
        'Sum': { 'indexPath': 'data/map_layers/map_bio_overall_All__index.js', 'indexName': 'map_bio_overall_All__index', 'layerPrefix': 'map_bio_overall_All' },
        'Ag': { 'indexPath': 'data/map_layers/map_bio_overall_Ag__index.js', 'indexName': 'map_bio_overall_Ag__index', 'layerPrefix': 'map_bio_overall_Ag' },
        'Ag Mgt': { 'indexPath': 'data/map_layers/map_bio_overall_Am__index.js', 'indexName': 'map_bio_overall_Am__index', 'layerPrefix': 'map_bio_overall_Am' },
        'Non-Ag': { 'indexPath': 'data/map_layers/map_bio_overall_NonAg__index.js', 'indexName': 'map_bio_overall_NonAg__index', 'layerPrefix': 'map_bio_overall_NonAg' },
      }
    },
    'Dvar': {
      'Ag': { 'indexPath': 'data/map_layers/map_dvar_Ag__index.js', 'indexName': 'map_dvar_Ag__index', 'layerPrefix': 'map_dvar_Ag' },
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_dvar_Am__index.js', 'indexName': 'map_dvar_Am__index', 'layerPrefix': 'map_dvar_Am' },
      'Non-Ag': { 'indexPath': 'data/map_layers/map_dvar_NonAg__index.js', 'indexName': 'map_dvar_NonAg__index', 'layerPrefix': 'map_dvar_NonAg' },
      'Lumap': { 'indexPath': 'data/map_layers/map_dvar_lumap__index.js', 'indexName': 'map_dvar_lumap__index', 'layerPrefix': 'map_dvar_lumap' },
    },
    'Economics': {
      'Sum': {
        'Profit': { 'indexPath': 'data/map_layers/map_economics_Sum_profit__index.js', 'indexName': 'map_economics_Sum_profit__index', 'layerPrefix': 'map_economics_Sum_profit' },
      },
      'Ag': {
        'Profit': { 'indexPath': 'data/map_layers/map_economics_Ag_profit__index.js', 'indexName': 'map_economics_Ag_profit__index', 'layerPrefix': 'map_economics_Ag_profit' },
        'Revenue': { 'indexPath': 'data/map_layers/map_economics_Ag_revenue__index.js', 'indexName': 'map_economics_Ag_revenue__index', 'layerPrefix': 'map_economics_Ag_revenue' },
        'Cost': { 'indexPath': 'data/map_layers/map_economics_Ag_cost__index.js', 'indexName': 'map_economics_Ag_cost__index', 'layerPrefix': 'map_economics_Ag_cost' },
        'Transition (Ag2Ag)': { 'indexPath': 'data/map_layers/map_economics_Ag_transition_ag2ag__index.js', 'indexName': 'map_economics_Ag_transition_ag2ag__index', 'layerPrefix': 'map_economics_Ag_transition_ag2ag' },
        'Transition (NonAg2Ag)': { 'indexPath': 'data/map_layers/map_economics_Ag_transition_nonag2ag__index.js', 'indexName': 'map_economics_Ag_transition_nonag2ag__index', 'layerPrefix': 'map_economics_Ag_transition_nonag2ag' },
      },
      'Ag Mgt': {
        'Profit': { 'indexPath': 'data/map_layers/map_economics_Am_profit__index.js', 'indexName': 'map_economics_Am_profit__index', 'layerPrefix': 'map_economics_Am_profit' },
        'Revenue': { 'indexPath': 'data/map_layers/map_economics_Am_revenue__index.js', 'indexName': 'map_economics_Am_revenue__index', 'layerPrefix': 'map_economics_Am_revenue' },
        'Cost': { 'indexPath': 'data/map_layers/map_economics_Am_cost__index.js', 'indexName': 'map_economics_Am_cost__index', 'layerPrefix': 'map_economics_Am_cost' },
      },
      'Non-Ag': {
        'Profit': { 'indexPath': 'data/map_layers/map_economics_NonAg_profit__index.js', 'indexName': 'map_economics_NonAg_profit__index', 'layerPrefix': 'map_economics_NonAg_profit' },
        'Revenue': { 'indexPath': 'data/map_layers/map_economics_NonAg_revenue__index.js', 'indexName': 'map_economics_NonAg_revenue__index', 'layerPrefix': 'map_economics_NonAg_revenue' },
        'Cost': { 'indexPath': 'data/map_layers/map_economics_NonAg_cost__index.js', 'indexName': 'map_economics_NonAg_cost__index', 'layerPrefix': 'map_economics_NonAg_cost' },
        'Transition (Ag2NonAg)': { 'indexPath': 'data/map_layers/map_economics_NonAg_transition_ag2non_ag__index.js', 'indexName': 'map_economics_NonAg_transition_ag2non_ag__index', 'layerPrefix': 'map_economics_NonAg_transition_ag2non_ag' },
        'Transition (NonAg2NonAg)': { 'indexPath': 'data/map_layers/map_economics_NonAg_transition_nonag2nonag__index.js', 'indexName': 'map_economics_NonAg_transition_nonag2nonag__index', 'layerPrefix': 'map_economics_NonAg_transition_nonag2nonag' },
      },
    },
    'GHG': {
      'Sum': { 'indexPath': 'data/map_layers/map_GHG_Sum__index.js', 'indexName': 'map_GHG_Sum__index', 'layerPrefix': 'map_GHG_Sum' },
      'Ag': { 'indexPath': 'data/map_layers/map_GHG_Ag__index.js', 'indexName': 'map_GHG_Ag__index', 'layerPrefix': 'map_GHG_Ag' },
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_GHG_Am__index.js', 'indexName': 'map_GHG_Am__index', 'layerPrefix': 'map_GHG_Am' },
      'Non-Ag': { 'indexPath': 'data/map_layers/map_GHG_NonAg__index.js', 'indexName': 'map_GHG_NonAg__index', 'layerPrefix': 'map_GHG_NonAg' },
    },
    'Production': {
      'Sum': { 'indexPath': 'data/map_layers/map_quantities_Sum__index.js', 'indexName': 'map_quantities_Sum__index', 'layerPrefix': 'map_quantities_Sum' },
      'Ag': { 'indexPath': 'data/map_layers/map_quantities_Ag__index.js', 'indexName': 'map_quantities_Ag__index', 'layerPrefix': 'map_quantities_Ag' },
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_quantities_Am__index.js', 'indexName': 'map_quantities_Am__index', 'layerPrefix': 'map_quantities_Am' },
      'Non-Ag': { 'indexPath': 'data/map_layers/map_quantities_NonAg__index.js', 'indexName': 'map_quantities_NonAg__index', 'layerPrefix': 'map_quantities_NonAg' },
    },
    'Water': {
      'Sum': { 'indexPath': 'data/map_layers/map_water_yield_Sum__index.js', 'indexName': 'map_water_yield_Sum__index', 'layerPrefix': 'map_water_yield_Sum' },
      'Ag': { 'indexPath': 'data/map_layers/map_water_yield_Ag__index.js', 'indexName': 'map_water_yield_Ag__index', 'layerPrefix': 'map_water_yield_Ag' },
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_water_yield_Am__index.js', 'indexName': 'map_water_yield_Am__index', 'layerPrefix': 'map_water_yield_Am' },
      'Non-Ag': { 'indexPath': 'data/map_layers/map_water_yield_NonAg__index.js', 'indexName': 'map_water_yield_NonAg__index', 'layerPrefix': 'map_water_yield_NonAg' },
    },
    'Renewable': {
      'Ag Mgt': { 'indexPath': 'data/map_layers/map_renewable_energy_Am__index.js', 'indexName': 'map_renewable_energy_Am__index', 'layerPrefix': 'map_renewable_energy_Am' },
    },
    'Transition': {
      'Area': {
        'Ag2Ag': { 'indexPath': 'data/map_layers/map_transition_area_ag2ag__index.js', 'indexName': 'map_transition_area_ag2ag__index', 'layerPrefix': 'map_transition_area_ag2ag' },
        'Ag2NonAg': { 'indexPath': 'data/map_layers/map_transition_area_ag2nonag__index.js', 'indexName': 'map_transition_area_ag2nonag__index', 'layerPrefix': 'map_transition_area_ag2nonag' },
      },
      'Cost': {
        'Ag2Ag': { 'indexPath': 'data/map_layers/map_transition_cost_ag2ag__index.js', 'indexName': 'map_transition_cost_ag2ag__index', 'layerPrefix': 'map_transition_cost_ag2ag' },
        'Ag2NonAg': { 'indexPath': 'data/map_layers/map_transition_cost_ag2nonag__index.js', 'indexName': 'map_transition_cost_ag2nonag__index', 'layerPrefix': 'map_transition_cost_ag2nonag' },
      },
    },
  }
}

