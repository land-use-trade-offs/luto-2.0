// Ranking Data Service
// This service provides data about rankings for different regions and metrics

window.DataService = {
  /**
   * Get ranking data based on the selected region and year
   * @param {String} selectRegion - The selected region (default: 'AUSTRALIA')
   * @param {String} selectYear - The selected year (default: 2020)
   * @returns {Object} The ranking data object
   */
  getRankingData(selectRegion = 'AUSTRALIA', selectYear = 2020) {
    try {
      // Helper function to safely access nested properties
      const safeAccess = (obj, path, defaultValue = null) => {
        try {
          return path.reduce((acc, key) => acc && acc[key], obj);
        } catch (e) {
          return defaultValue;
        }
      };      // Populate the rankingData based on the selected region
      const rankingData = {
        'Economics': {
          'Revenue': {
            'Rank': safeAccess(window, ['Economics_ranking', selectRegion, 'Revenue', 'Rank', selectYear]),
            'color': safeAccess(window, ['Economics_ranking', selectRegion, 'Revenue', 'color', selectYear]),
            'value': safeAccess(window, ['Economics_ranking', selectRegion, 'Revenue', 'value', selectYear]),
          },
          'Cost': {
            'Rank': safeAccess(window, ['Economics_ranking', selectRegion, 'Cost', 'Rank', selectYear]),
            'color': safeAccess(window, ['Economics_ranking', selectRegion, 'Cost', 'color', selectYear]),
            'value': safeAccess(window, ['Economics_ranking', selectRegion, 'Cost', 'value', selectYear]),
          },
          'Total': {
            'Rank': safeAccess(window, ['Economics_ranking', selectRegion, 'Total', 'Rank', selectYear]),
            'color': safeAccess(window, ['Economics_ranking', selectRegion, 'Total', 'color', selectYear]),
            'value': safeAccess(window, ['Economics_ranking', selectRegion, 'Total', 'value', selectYear]),
          },
        },
        'Area': {
          'Ag': {
            'Rank': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Landuse', 'Rank', selectYear]),
            'color': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Landuse', 'color', selectYear]),
            'value': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Landuse', 'value', selectYear]),
          },
          'Am': {
            'Rank': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Management', 'Rank', selectYear]),
            'color': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Management', 'color', selectYear]),
            'value': safeAccess(window, ['Area_ranking', selectRegion, 'Agricultural Management', 'value', selectYear]),
          },
          'NonAg': {
            'Rank': safeAccess(window, ['Area_ranking', selectRegion, 'Non-Agricultural Landuse', 'Rank', selectYear]),
            'color': safeAccess(window, ['Area_ranking', selectRegion, 'Non-Agricultural Landuse', 'color', selectYear]),
            'value': safeAccess(window, ['Area_ranking', selectRegion, 'Non-Agricultural Landuse', 'value', selectYear]),
          },
          'Total': {
            'Rank': safeAccess(window, ['Area_ranking', selectRegion, 'Total', 'Rank', selectYear]),
            'color': safeAccess(window, ['Area_ranking', selectRegion, 'Total', 'color', selectYear]),
            'value': safeAccess(window, ['Area_ranking', selectRegion, 'Total', 'value', selectYear]),
          },
        },
        'GHG': {
          'Emissions': {
            'Rank': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG emissions', 'Rank', selectYear]),
            'color': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG emissions', 'color', selectYear]),
            'value': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG emissions', 'value', selectYear]),
          },
          'Sequestration': {
            'Rank': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG sequestrations', 'Rank', selectYear]),
            'color': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG sequestrations', 'color', selectYear]),
            'value': safeAccess(window, ['GHG_ranking', selectRegion, 'GHG sequestrations', 'value', selectYear]),
          },
          'Total': {
            'Rank': safeAccess(window, ['GHG_ranking', selectRegion, 'Total', 'Rank', selectYear]),
            'color': safeAccess(window, ['GHG_ranking', selectRegion, 'Total', 'color', selectYear]),
            'value': safeAccess(window, ['GHG_ranking', selectRegion, 'Total', 'value', selectYear]),
          },
        },
        'Water': {
          'Ag': {
            'Rank': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Landuse', 'Rank', selectYear]),
            'color': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Landuse', 'color', selectYear]),
            'value': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Landuse', 'value', selectYear]),
          },
          'Am': {
            'Rank': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Management', 'Rank', selectYear]),
            'color': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Management', 'color', selectYear]),
            'value': safeAccess(window, ['Water_ranking', selectRegion, 'Agricultural Management', 'value', selectYear]),
          },
          'NonAg': {
            'Rank': safeAccess(window, ['Water_ranking', selectRegion, 'Non-Agricultural Landuse', 'Rank', selectYear]),
            'color': safeAccess(window, ['Water_ranking', selectRegion, 'Non-Agricultural Landuse', 'color', selectYear]),
            'value': safeAccess(window, ['Water_ranking', selectRegion, 'Non-Agricultural Landuse', 'value', selectYear]),
          },
          'Total': {
            'Rank': safeAccess(window, ['Water_ranking', selectRegion, 'Total', 'Rank', selectYear]),
            'color': safeAccess(window, ['Water_ranking', selectRegion, 'Total', 'color', selectYear]),
            'value': safeAccess(window, ['Water_ranking', selectRegion, 'Total', 'value', selectYear]),
          },
        },
        'Biodiversity': {
          'Ag': {
            'Rank': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Landuse', 'Rank', selectYear]),
            'color': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Landuse', 'color', selectYear]),
            'value': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Landuse', 'value', selectYear]),
          },
          'Am': {
            'Rank': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Management', 'Rank', selectYear]),
            'color': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Management', 'color', selectYear]),
            'value': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Agricultural Management', 'value', selectYear]),
          },
          'NonAg': {
            'Rank': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Non-Agricultural land-use', 'Rank', selectYear]),
            'color': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Non-Agricultural land-use', 'color', selectYear]),
            'value': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Non-Agricultural land-use', 'value', selectYear]),
          },
          'Total': {
            'Rank': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Total', 'Rank', selectYear]),
            'color': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Total', 'color', selectYear]),
            'value': safeAccess(window, ['Biodiversity_ranking', selectRegion, 'Total', 'value', selectYear]),
          },
        },
      };

      return rankingData;
    } catch (error) {
      console.error("Error loading ranking data:", error);
      return {};
    }
  },

  /**
   * Get subcategory keys for a given data type
   * @param {String} dataType - The data type (Economics, Area, GHG, Water, Biodiversity)
   * @returns {Array} Array of subcategory keys
   */
  getSubcategories(dataType) {
    return this.SubcategoryMapping[dataType] ? Object.keys(this.SubcategoryMapping[dataType]) : [];
  },

  // Mapping between UI subcategory names and actual data structure keys
  SubcategoryMapping: {
    'Economics': {
      'Revenue': 'Revenue',
      'Cost': 'Cost',
    },
    'Area': {
      'Ag': 'Agricultural Landuse',
      'Ag Mgt': 'Agricultural Management',
      'Non-Ag': 'Non-Agricultural Landuse',
      'Total': 'Total'
    },
    'GHG': {
      'Emissions': 'GHG emissions',
      'Sequestration': 'GHG sequestrations',
    },
    'Water': {
      'Ag': 'Agricultural Landuse',
      'Ag Mgt': 'Agricultural Management',
      'Non-Ag': 'Non-Agricultural Landuse',
      'Total': 'Total'
    },
    'Biodiversity': {
      'Ag': 'Agricultural Landuse',
      'Ag Mgt': 'Agricultural Management',
      'Non-Ag': 'Non-Agricultural land-use',
      'Total': 'Total'
    }
  },

  /**
   * Maps a UI subcategory to the actual data structure key
   * @param {String} dataType - The data type (Economics, Area, GHG, Water, Biodiversity)
   * @param {String} subcategory - The UI subcategory (Ag, Am, NonAg, Total, etc.)
   * @returns {String} The actual data structure key
   */
  mapSubcategory(dataType, subcategory) {
    // Default to 'Total' if no subcategory provided
    if (!subcategory) return 'Total';

    // Get the mapping for this data type
    const mapping = this.SubcategoryMapping[dataType];
    if (!mapping) return subcategory;

    // Return the mapped subcategory or the original if no mapping found
    return mapping[subcategory] || subcategory;
  },

  ChartPaths: {
    'Area': {
      'Ag': {
        'Landuse': 'data/Area_Ag_1_Land-use.js',
        'Water': 'data/Area_Ag_2_Water_supply.js'
      },
      'Ag Mgt': {
        'Mgt Type': 'data/Area_Am_1_Type.js',
        'Water': 'data/Area_Am_2_Water_supply.js',
        'Landuse': 'data/Area_Am_3_Land-use.js'
      },
      'Non-Ag': {
        'Landuse': 'data/Area_NonAg_1_Land-use.js'
      },
      'Overview': {
        'Landuse': 'data/Area_overview_1_Land-use.js',
        'Category': 'data/Area_overview_2_Category.js',
        'Source': 'data/Area_overview_3_Source.js'
      },
      'Ranking': 'data/Area_ranking.js',
      'Transition': {
        'Start-end': 'data/Area_transition_start_end.js',
        'Year to Year': 'data/Area_transition_year_to_year.js'
      }
    },
    'Biodiversity': {
      'Overview': {
        'Type': 'data/BIO_GBF2_overview_1_Type.js'
      },
      'Ag': {
        'Landuse': 'data/BIO_GBF2_split_Ag_1_Landuse.js'
      },
      'Ag Mgt': {
        'Landuse': 'data/BIO_GBF2_split_Am_1_Landuse.js',
        'Agri-Management': 'data/BIO_GBF2_split_Am_2_Agri-Management.js'
      },
      'Non-Ag': {
        'Landuse': 'data/BIO_GBF2_split_NonAg_1_Landuse.js'
      },
      'Ranking': 'data/Biodiversity_ranking.js'
    },
    'Economics': {
      'Overview': 'data/Economics_overview.js',
      'Ranking': 'data/Economics_ranking.js',
      'Ag Mgt': {
        'Management Type': 'data/Economics_split_AM_1_Management_Type.js',
        'Water supply': 'data/Economics_split_AM_2_Water_supply.js',
        'Landuse': 'data/Economics_split_AM_3_Land-use.js'
      },
      'Ag': {
        'Landuse': 'data/Economics_split_Ag_1_Land-use.js',
        'Type': 'data/Economics_split_Ag_2_Type.js',
        'Water supply': 'data/Economics_split_Ag_3_Water_supply.js'
      },
      'Non-Ag': {
        'Landuse': 'data/Economics_split_NonAg_1_Land-use.js'
      },
      'Transition': {
        'Matrix': {
          'Ag to Ag': 'data/Economics_transition_mat_ag2ag.js',
          'Ag to Non-Ag': 'data/Economics_transition_mat_ag2nonag.js',
          'Non-Ag to Ag': 'data/Economics_transition_mat_nonag2ag.js'
        },
        'Aggregated': {
          'Ag to Non-Ag': {
            'Cost type': 'data/Economics_transition_split_Ag2NonAg_1_Cost_type.js',
            'From landuse': 'data/Economics_transition_split_Ag2NonAg_2_From_land-use.js',
            'To landuse': 'data/Economics_transition_split_Ag2NonAg_3_To_land-use.js'
          },
          'Non-Ag to Ag': {
            'Cost type': 'data/Economics_transition_split_NonAg2Ag_1_Cost_type.js',
            'From landuse': 'data/Economics_transition_split_NonAg2Ag_2_From_land-use.js',
            'To landuse': 'data/Economics_transition_split_NonAg2Ag_3_To_land-use.js'
          },
          'Ag to Ag': {
            'Type': 'data/Economics_transition_split_ag2ag_1_Type.js',
            'From landuse': 'data/Economics_transition_split_ag2ag_2_From_land-use.js',
            'To landuse': 'data/Economics_transition_split_ag2ag_3_To_land-use.js'
          }
        }
      }
    },
    'GHG': {
      'Overview': 'data/GHG_overview.js',
      'Ranking': 'data/GHG_ranking.js',
      'Ag': {
        'GHG Category': 'data/GHG_split_Ag_1_GHG_Category.js',
        'Landuse': 'data/GHG_split_Ag_2_Land-use.js',
        'Landuse type': 'data/GHG_split_Ag_3_Land-use_type.js',
        'Source': 'data/GHG_split_Ag_4_Source.js',
        'Water supply': 'data/GHG_split_Ag_5_Water_supply.js'
      },
      'Ag Mgt': {
        'Landuse': 'data/GHG_split_Am_1_Land-use.js',
        'Landuse type': 'data/GHG_split_Am_2_Land-use_type.js',
        'Agricultural Management Type': 'data/GHG_split_Am_3_Agricultural_Management_Type.js',
        'Water supply': 'data/GHG_split_Am_4_Water_supply.js'
      },
      'Non-Ag': {
        'Landuse': 'data/GHG_split_NonAg_1_Land-use.js'
      },
      // Skip off land GHG because it is record of the whole Australia
      // 'Off-land': {
      //   'Emission Type': 'data/GHG_split_off_land_1_Emission_Type.js',
      //   'Emission Source': 'data/GHG_split_off_land_2_Emission_Source.js',
      //   'Commodity': 'data/GHG_split_off_land_3_Commodity.js'
      // }
    },
    'Production': {
      'Ag': 'data/Production_LUTO_1_Agricultural.js',
      'Non-A': 'data/Production_LUTO_2_Non-Agricultural.js',
      'Ag Mgt': 'data/Production_LUTO_3_Agricultural_Management.js',
      'Overview': 'data/Production_achive_percent.js',
      // Skip Demand because it is record of the whole Australia
      // 'Demand': {
      //   'Type': 'data/Production_demand_1_Type.js',
      //   'On Off Land': 'data/Production_demand_2_on_off_land.js',
      //   'Commodity': 'data/Production_demand_3_Commodity.js',
      //   'Limit': 'data/Production_demand_4_Limit.js'
      // },
      'Sum': {
        'Commodity': 'data/Production_sum_1_Commodity.js',
        'Type': 'data/Production_sum_2_Type.js'
      }
    },
    'Water': {
      'Overview': {
        'Landuse': 'data/Water_overview_NRM_region_1_Landuse.js',
        'Type': 'data/Water_overview_NRM_region_2_Type.js'
      },
      'Ranking': 'data/Water_ranking.js',
      'Ag': {
        'Landuse': 'data/Water_split_Ag_NRM_region_1_Landuse.js',
        'Water Supply': 'data/Water_split_Ag_NRM_region_2_Water_Supply.js'

      },
      'Ag Mgt': {
        'Water Supply': 'data/Water_split_Am_NRM_region_1_Water_Supply.js',
        'Landuse': 'data/Water_split_Am_NRM_region_2_Landuse.js',
        'Agri-Management': 'data/Water_split_Am_NRM_region_3_Agri-Management.js'
      },
      'Non-Ag': {
        'Landuse': 'data/Water_split_NonAg_NRM_region_1_Landuse.js'
      }
    }
  }

};
