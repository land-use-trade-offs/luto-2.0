// Chart Data Service
// This service provides data about charts for different metrics and categories

window.DataService = {
  chartCategories: {
    Area: {
      Ag: {
        "Land-use": {
          path: "data/Area_Ag_1_Land-use.js",
          name: "Area_Ag_1_Land-use",
        },
        "Water supply": {
          path: "data/Area_Ag_2_Water_supply.js",
          name: "Area_Ag_2_Water_supply",
        },
      },
      "Ag Mgt": {
        Type: { path: "data/Area_Am_1_Type.js", name: "Area_Am_1_Type" },
        "Water supply": {
          path: "data/Area_Am_2_Water_supply.js",
          name: "Area_Am_2_Water_supply",
        },
        "Land-use": {
          path: "data/Area_Am_3_Land-use.js",
          name: "Area_Am_3_Land-use",
        },
      },
      "Non-Ag": {
        "Land-use": {
          path: "data/Area_NonAg_1_Land-use.js",
          name: "Area_NonAg_1_Land-use",
        },
      },
      ranking: {
        path: "data/Area_ranking.js",
        name: "Area_ranking",
      },
      overview: {
        "Land-use": {
          path: "data/Area_overview_1_Land-use.js",
          name: "Area_overview_1_Land-use",
        },
        Category: {
          path: "data/Area_overview_2_Category.js",
          name: "Area_overview_2_Category",
        },
        Source: {
          path: "data/Area_overview_3_Source.js",
          name: "Area_overview_3_Source",
        },
      },
      transition: {
        "start end": {
          path: "data/Area_transition_start_end.js",
          name: "Area_transition_start_end",
        },
        "year to year": {
          path: "data/Area_transition_year_to_year.js",
          name: "Area_transition_year_to_year",
        },
      },
    },
    Production: {
      Ag: {
        path: "data/Production_LUTO_1_Agricultural.js",
        name: "Production_LUTO_1_Agricultural",
      },
      "Ag Mgt": {
        path: "data/Production_LUTO_3_Agricultural_Management.js",
        name: "Production_LUTO_3_Agricultural_Management",
      },
      "Non-Ag": {
        path: "data/Production_LUTO_2_Non-Agricultural.js",
        name: "Production_LUTO_2_Non-Agricultural",
      },
      overview: {
        path: "data/Production_achive_percent.js",
        name: "Production_achive_percent",
      },
      sum: {
        Commodity: {
          path: "data/Production_sum_1_Commodity.js",
          name: "Production_sum_1_Commodity",
        },
        Type: {
          path: "data/Production_sum_2_Type.js",
          name: "Production_sum_2_Type",
        },
      },
      demand: {
        Type: {
          path: "data/Production_demand_1_Type.js",
          name: "Production_demand_1_Type",
        },
        "on off land": {
          path: "data/Production_demand_2_on_off_land.js",
          name: "Production_demand_2_on_off_land",
        },
        Commodity: {
          path: "data/Production_demand_3_Commodity.js",
          name: "Production_demand_3_Commodity",
        },
        Limit: {
          path: "data/Production_demand_4_Limit.js",
          name: "Production_demand_4_Limit",
        },
      },
    },
    Economics: {
      Ag: {
        "Land-use": {
          path: "data/Economics_split_Ag_1_Land-use.js",
          name: "Economics_split_Ag_1_Land-use",
        },
        Type: {
          path: "data/Economics_split_Ag_2_Type.js",
          name: "Economics_split_Ag_2_Type",
        },
        "Water supply": {
          path: "data/Economics_split_Ag_3_Water_supply.js",
          name: "Economics_split_Ag_3_Water_supply",
        },
      },
      "Ag Mgt": {
        "Management Type": {
          path: "data/Economics_split_AM_1_Management_Type.js",
          name: "Economics_split_AM_1_Management_Type",
        },
        "Water supply": {
          path: "data/Economics_split_AM_2_Water_supply.js",
          name: "Economics_split_AM_2_Water_supply",
        },
        "Land-use": {
          path: "data/Economics_split_AM_3_Land-use.js",
          name: "Economics_split_AM_3_Land-use",
        },
      },
      "Non-Ag": {
        "Land-use": {
          path: "data/Economics_split_NonAg_1_Land-use.js",
          name: "Economics_split_NonAg_1_Land-use",
        },
      },
      ranking: {
        path: "data/Economics_ranking.js",
        name: "Economics_ranking",
      },
      overview: {
        path: "data/Economics_overview.js",
        name: "Economics_overview",
      },
      transition: {
        matrix: {
          ag2ag: {
            path: "data/Economics_transition_mat_ag2ag.js",
            name: "Economics_transition_mat_ag2ag",
          },
          ag2nonag: {
            path: "data/Economics_transition_mat_ag2nonag.js",
            name: "Economics_transition_mat_ag2nonag",
          },
          nonag2ag: {
            path: "data/Economics_transition_mat_nonag2ag.js",
            name: "Economics_transition_mat_nonag2ag",
          },
        },
        ag2ag: {
          Type: {
            path: "data/Economics_transition_split_ag2ag_1_Type.js",
            name: "Economics_transition_split_ag2ag_1_Type",
          },
          "From land-use": {
            path: "data/Economics_transition_split_ag2ag_2_From_land-use.js",
            name: "Economics_transition_split_ag2ag_2_From_land-use",
          },
          "To land-use": {
            path: "data/Economics_transition_split_ag2ag_3_To_land-use.js",
            name: "Economics_transition_split_ag2ag_3_To_land-use",
          },
        },
        ag2nonag: {
          "Cost type": {
            path: "data/Economics_transition_split_Ag2NonAg_1_Cost_type.js",
            name: "Economics_transition_split_Ag2NonAg_1_Cost_type",
          },
          "From land-use": {
            path: "data/Economics_transition_split_Ag2NonAg_2_From_land-use.js",
            name: "Economics_transition_split_Ag2NonAg_2_From_land-use",
          },
          "To land-use": {
            path: "data/Economics_transition_split_Ag2NonAg_3_To_land-use.js",
            name: "Economics_transition_split_Ag2NonAg_3_To_land-use",
          },
        },
        nonag2ag: {
          "Cost type": {
            path: "data/Economics_transition_split_NonAg2Ag_1_Cost_type.js",
            name: "Economics_transition_split_NonAg2Ag_1_Cost_type",
          },
          "From land-use": {
            path: "data/Economics_transition_split_NonAg2Ag_2_From_land-use.js",
            name: "Economics_transition_split_NonAg2Ag_2_From_land-use",
          },
          "To land-use": {
            path: "data/Economics_transition_split_NonAg2Ag_3_To_land-use.js",
            name: "Economics_transition_split_NonAg2Ag_3_To_land-use",
          },
        },
      },
    },
    GHG: {
      Ag: {
        "GHG Category": {
          path: "data/GHG_split_Ag_1_GHG_Category.js",
          name: "GHG_split_Ag_1_GHG_Category",
        },
        "Land-use": {
          path: "data/GHG_split_Ag_2_Land-use.js",
          name: "GHG_split_Ag_2_Land-use",
        },
        "Land-use type": {
          path: "data/GHG_split_Ag_3_Land-use_type.js",
          name: "GHG_split_Ag_3_Land-use_type",
        },
        Source: {
          path: "data/GHG_split_Ag_4_Source.js",
          name: "GHG_split_Ag_4_Source",
        },
        "Water supply": {
          path: "data/GHG_split_Ag_5_Water_supply.js",
          name: "GHG_split_Ag_5_Water_supply",
        },
      },
      "Ag Mgt": {
        "Land-use": {
          path: "data/GHG_split_Am_1_Land-use.js",
          name: "GHG_split_Am_1_Land-use",
        },
        "Land-use type": {
          path: "data/GHG_split_Am_2_Land-use_type.js",
          name: "GHG_split_Am_2_Land-use_type",
        },
        "Management Type": {
          path: "data/GHG_split_Am_3_Agricultural_Management_Type.js",
          name: "GHG_split_Am_3_Agricultural_Management_Type",
        },
        "Water supply": {
          path: "data/GHG_split_Am_4_Water_supply.js",
          name: "GHG_split_Am_4_Water_supply",
        },
      },
      "Non-Ag": {
        "Land-use": {
          path: "data/GHG_split_NonAg_1_Land-use.js",
          name: "GHG_split_NonAg_1_Land-use",
        },
      },
      ranking: {
        path: "data/GHG_ranking.js",
        name: "GHG_ranking",
      },
      overview: {
        path: "data/GHG_overview.js",
        name: "GHG_overview",
      },
      "off land": {
        "Emission Type": {
          path: "data/GHG_split_off_land_1_Emission_Type.js",
          name: "GHG_split_off_land_1_Emission_Type",
        },
        "Emission Source": {
          path: "data/GHG_split_off_land_2_Emission_Source.js",
          name: "GHG_split_off_land_2_Emission_Source",
        },
        Commodity: {
          path: "data/GHG_split_off_land_3_Commodity.js",
          name: "GHG_split_off_land_3_Commodity",
        },
      },
    },
    Water: {
      NRM: {
        Ag: {
          Landuse: {
            path: "data/Water_split_Ag_NRM_region_1_Landuse.js",
            name: "Water_split_Ag_NRM_region_1_Landuse",
          },
          "Water Supply": {
            path: "data/Water_split_Ag_NRM_region_2_Water_Supply.js",
            name: "Water_split_Ag_NRM_region_2_Water_Supply",
          },
        },
        "Ag Mgt": {
          "Water Supply": {
            path: "data/Water_split_Am_NRM_region_1_Water_Supply.js",
            name: "Water_split_Am_NRM_region_1_Water_Supply",
          },
          Landuse: {
            path: "data/Water_split_Am_NRM_region_2_Landuse.js",
            name: "Water_split_Am_NRM_region_2_Landuse",
          },
          "Management Type": {
            path: "data/Water_split_Am_NRM_region_3_Agri-Management.js",
            name: "Water_split_Am_NRM_region_3_Agri-Management",
          },
        },
        "Non-Ag": {
          Landuse: {
            path: "data/Water_split_NonAg_NRM_region_1_Landuse.js",
            name: "Water_split_NonAg_NRM_region_1_Landuse",
          },
        },
        overview: {
          "Landuse": {
            path: "data/Water_overview_NRM_region_1_Landuse.js",
            name: "Water_overview_NRM_region_1_Landuse",
          },
          "Type": {
            path: "data/Water_overview_NRM_region_2_Type.js",
            name: "Water_overview_NRM_region_2_Type",
          },
        },
        ranking: {
          path: "data/Water_ranking.js",
          name: "Water_ranking"
        },
      },
      Watershed: {
        overview: {
          Australia: {
            path: "data/Water_overview_AUSTRALIA.js",
            name: "Water_overview_AUSTRALIA",
          },
          landuse: {
            path: "data/Water_overview_landuse.js",
            name: "Water_overview_landuse",
          },
          overview: {
            path: "data/Water_overview_by_watershed_region.js",
            name: "Water_overview_by_watershed_region",
          },
        },

      },
    },
    Biodiversity: {
      GBF2: {
        Ag: {
          Landuse: {
            path: "data/BIO_GBF2_split_Ag_1_Landuse.js",
            name: "BIO_GBF2_split_Ag_1_Landuse",
          },
        },
        "Ag Mgt": {
          Landuse: {
            path: "data/BIO_GBF2_split_Am_1_Landuse.js",
            name: "BIO_GBF2_split_Am_1_Landuse",
          },
          "Agri-Management": {
            path: "data/BIO_GBF2_split_Am_2_Agri-Management.js",
            name: "BIO_GBF2_split_Am_2_Agri-Management",
          },
        },
        "Non-Ag": {
          Landuse: {
            path: "data/BIO_GBF2_split_NonAg_1_Landuse.js",
            name: "BIO_GBF2_split_NonAg_1_Landuse",
          },
        },
        overview: {
          Type: {
            path: "data/BIO_GBF2_overview_1_Type.js",
            name: "BIO_GBF2_overview_1_Type",
          },
        },
        ranking: {
          path: "data/Biodiversity_ranking.js",
          name: "Biodiversity_ranking",
        },
      },
      quality: {
        Ag: {
          Landuse: {
            path: "data/BIO_quality_split_Ag_1_Landuse.js",
            name: "BIO_quality_split_Ag_1_Landuse",
          },
        },
        "Ag Mgt": {
          Landuse: {
            path: "data/BIO_quality_split_Am_1_Landuse.js",
            name: "BIO_quality_split_Am_1_Landuse",
          },
          "Agri-Management": {
            path: "data/BIO_quality_split_Am_2_Agri-Management.js",
            name: "BIO_quality_split_Am_2_Agri-Management",
          },
        },
        "Non-Ag": {
          Landuse: {
            path: "data/BIO_quality_split_NonAg_1_Landuse.js",
            name: "BIO_quality_split_NonAg_1_Landuse",
          },
        },
        overview: {
          Type: {
            path: "data/BIO_quality_overview_1_Type.js",
            name: "BIO_quality_overview_1_Type",
          },
        }
      },
    },
  },
};
