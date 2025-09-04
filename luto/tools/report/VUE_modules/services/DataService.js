// Chart Data Service
// This service provides data about charts for different metrics and categories

window.DataService = {

  chartCategories: {
    Area: {
      Ag: {
        path: "data/Area_Ag.js",
        name: "Area_Ag",
      },
      "Ag Mgt": {
        path: "data/Area_Am.js",
        name: "Area_Am",
      },
      "Non-Ag": {
        path: "data/Area_NonAg.js",
        name: "Area_NonAg",
      },
      overview: {
        Category: {
          path: "data/Area_overview_2_Category.js",
          name: "Area_overview_2_Category",
        },
        "Land-use": {
          path: "data/Area_overview_1_Land-use.js",
          name: "Area_overview_1_Land-use",
        },
        Source: {
          path: "data/Area_overview_3_Source.js",
          name: "Area_overview_3_Source",
        },
      },
      ranking: {
        path: "data/Area_ranking.js",
        name: "Area_ranking",
      },
    },
    Biodiversity: {
      GBF2: {
        Ag: {
          path: "data/BIO_GBF2_split_Ag_1_Landuse.js",
          name: "BIO_GBF2_split_Ag_1_Landuse",
        },
        "Ag Mgt": {
          "Agri-Management": {
            path: "data/BIO_GBF2_split_Am_2_Agri-Management.js",
            name: "BIO_GBF2_split_Am_2_Agri-Management",
          },
          Landuse: {
            path: "data/BIO_GBF2_split_Am_1_Landuse.js",
            name: "BIO_GBF2_split_Am_1_Landuse",
          },
        },
        "Non-Ag": {
          path: "data/BIO_GBF2_split_NonAg_1_Landuse.js",
          name: "BIO_GBF2_split_NonAg_1_Landuse",
        },
        overview: {
          path: "data/BIO_GBF2_overview_1_Type.js",
          name: "BIO_GBF2_overview_1_Type",
        },
      },
      quality: {
        Ag: {
          path: "data/BIO_quality_split_Ag_1_Landuse.js",
          name: "BIO_quality_split_Ag_1_Landuse",
        },
        "Ag Mgt": {
          "Agri-Management": {
            path: "data/BIO_quality_split_Am_2_Agri-Management.js",
            name: "BIO_quality_split_Am_2_Agri-Management",
          },
          Landuse: {
            path: "data/BIO_quality_split_Am_1_Landuse.js",
            name: "BIO_quality_split_Am_1_Landuse",
          },
        },
        "Non-Ag": {
          path: "data/BIO_quality_split_NonAg_1_Landuse.js",
          name: "BIO_quality_split_NonAg_1_Landuse",
        },
        overview: {
          path: "data/BIO_quality_overview_1_Type.js",
          name: "BIO_quality_overview_1_Type",
        },
      },
      ranking: {
        path: "data/Biodiversity_ranking.js",
        name: "Biodiversity_ranking",
      },
    },
    Economics: {
      Ag: {
        path: "data/Economics_Ag.js",
        name: "Economics_Ag",
      },
      "Ag Mgt": {
        path: "data/Economics_Am.js",
        name: "Economics_Am",
      },
      "Non-Ag": {
        path: "data/Economics_overview_Non_Ag.js",
        name: "Economics_overview_Non_Ag",
      },
      overview: {
        Ag: {
          path: "data/Economics_overview_Ag.js",
          name: "Economics_overview_Ag",
        },
        "Ag Mgt": {
          path: "data/Economics_overview_Am.js",
          name: "Economics_overview_Am",
        },
        "Non-Ag": {
          path: "data/Economics_overview_Non_Ag.js",
          name: "Economics_overview_Non_Ag",
        },
        sum: {
          path: "data/Economics_overview_sum.js",
          name: "Economics_overview_sum",
        },
      },
      ranking: {
        path: "data/Economics_ranking.js",
        name: "Economics_ranking",
      },
    },
    GHG: {
      Ag: {
        path: "data/GHG_Ag.js",
        name: "GHG_Ag",
      },
      "Ag Mgt": {
        path: "data/GHG_Am.js",
        name: "GHG_Am",
      },
      "Non-Ag": {
        path: "data/GHG_NonAg.js",
        name: "GHG_NonAg",
      },
      overview: {
        path: "data/GHG_overview.js",
        name: "GHG_overview",
      },
      ranking: {
        path: "data/GHG_ranking.js",
        name: "GHG_ranking",
      },
    },
    Production: {
      Ag: {
        path: "data/Production_Ag.js",
        name: "Production_Ag",
      },
      "Ag Mgt": {
        path: "data/Production_Am.js",
        name: "Production_Am",
      },
      "Non-Ag": {
        path: "data/Production_NonAg.js",
        name: "Production_NonAg",
      },
      overview: {
        achieve: {
          path: "data/Production_overview_AUS_achive_percent.js",
          name: "Production_overview_AUS_achive_percent",
        },
        sum: {
          path: "data/Production_overview_sum.js",
          name: "Production_overview_sum",
        },
      },
    },
    Water: {
      NRM: {
        Ag: {
          path: "data/Water_Ag_NRM.js",
          name: "Water_Ag_NRM",
        },
        "Ag Mgt": {
          path: "data/Water_Am_NRM.js",
          name: "Water_Am_NRM",
        },
        "Non-Ag": {
          path: "data/Water_NonAg_NRM.js",
          name: "Water_NonAg_NRM",
        },
        overview: {
          Type: {
            path: "data/Water_overview_NRM_Type.js",
            name: "Water_overview_NRM_Type",
          },
        },
        ranking: {
          path: "data/Water_ranking_NRM.js",
          name: "Water_ranking_NRM",
        },
      },
      Watershed: {
        overview: {
          path: "data/Water_overview_watershed.js",
          name: "Water_overview_watershed",
        },
      },
    },
  },
};