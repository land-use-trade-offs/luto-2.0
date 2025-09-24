// Chart Data Service
// This service provides data about charts for different metrics and categories

window.ChartService = {

  "chartCategories": {
    "Area": {
      "Ag": {
        "path": "data/Area_Ag.js",
        "name": "Area_Ag",
      },
      "Ag Mgt": {
        "path": "data/Area_Am.js",
        "name": "Area_Am",
      },
      "Non-Ag": {
        "path": "data/Area_NonAg.js",
        "name": "Area_NonAg",
      },
      "overview": {
        "Land-use": {
          "path": "data/Area_overview_1_Land-use.js",
          "name": "Area_overview_1_Land-use",
        },
        "Category": {
          "path": "data/Area_overview_2_Category.js",
          "name": "Area_overview_2_Category",
        },
        "Source": {
          "path": "data/Area_overview_3_Source.js",
          "name": "Area_overview_3_Source",
        },
      },
      "ranking": {
        "path": "data/Area_ranking.js",
        "name": "Area_ranking",
      },
    },
    "Biodiversity": {
      "GBF2": {
        "Ag": {
          "path": "data/BIO_GBF2_Ag.js",
          "name": "BIO_GBF2_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF2_Am.js",
          "name": "BIO_GBF2_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF2_NonAg.js",
          "name": "BIO_GBF2_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF2_overview_Ag.js",
            "name": "BIO_GBF2_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF2_overview_Am.js",
            "name": "BIO_GBF2_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF2_overview_NonAg.js",
            "name": "BIO_GBF2_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF2_overview_sum.js",
            "name": "BIO_GBF2_overview_sum",
          },
        },
      },
      "GBF3": {
        "Ag": {
          "path": "data/BIO_GBF3_Ag.js",
          "name": "BIO_GBF3_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF3_Am.js",
          "name": "BIO_GBF3_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF3_NonAg.js",
          "name": "BIO_GBF3_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF3_overview_Ag.js",
            "name": "BIO_GBF3_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF3_overview_Am.js",
            "name": "BIO_GBF3_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF3_overview_NonAg.js",
            "name": "BIO_GBF3_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF3_overview_sum.js",
            "name": "BIO_GBF3_overview_sum",
          },
        },
      },
      "GBF4_SNES": {
        "Ag": {
          "path": "data/BIO_GBF4_SNES_Ag.js",
          "name": "BIO_GBF4_SNES_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF4_SNES_Am.js",
          "name": "BIO_GBF4_SNES_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF4_SNES_NonAg.js",
          "name": "BIO_GBF4_SNES_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF4_SNES_overview_Ag.js",
            "name": "BIO_GBF4_SNES_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF4_SNES_overview_Am.js",
            "name": "BIO_GBF4_SNES_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF4_SNES_overview_NonAg.js",
            "name": "BIO_GBF4_SNES_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF4_SNES_overview_sum.js",
            "name": "BIO_GBF4_SNES_overview_sum",
          },
        },
      },
      "GBF4_ECNES": {
        "Ag": {
          "path": "data/BIO_GBF4_ECNES_Ag.js",
          "name": "BIO_GBF4_ECNES_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF4_ECNES_Am.js",
          "name": "BIO_GBF4_ECNES_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF4_ECNES_NonAg.js",
          "name": "BIO_GBF4_ECNES_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF4_ECNES_overview_Ag.js",
            "name": "BIO_GBF4_ECNES_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF4_ECNES_overview_Am.js",
            "name": "BIO_GBF4_ECNES_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF4_ECNES_overview_NonAg.js",
            "name": "BIO_GBF4_ECNES_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF4_ECNES_overview_sum.js",
            "name": "BIO_GBF4_ECNES_overview_sum",
          },
        },
      },
      "GBF8_GROUP": {
        "Ag": {
          "path": "data/BIO_GBF8_GROUP_Ag.js",
          "name": "BIO_GBF8_GROUP_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF8_GROUP_Am.js",
          "name": "BIO_GBF8_GROUP_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF8_GROUP_NonAg.js",
          "name": "BIO_GBF8_GROUP_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF8_GROUP_overview_Ag.js",
            "name": "BIO_GBF8_GROUP_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF8_GROUP_overview_Am.js",
            "name": "BIO_GBF8_GROUP_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF8_GROUP_overview_NonAg.js",
            "name": "BIO_GBF8_GROUP_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF8_GROUP_overview_sum.js",
            "name": "BIO_GBF8_GROUP_overview_sum",
          },
        },
      },
      "GBF8_SPECIES": {
        "Ag": {
          "path": "data/BIO_GBF8_SPECIES_Ag.js",
          "name": "BIO_GBF8_SPECIES_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_GBF8_SPECIES_Am.js",
          "name": "BIO_GBF8_SPECIES_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_GBF8_SPECIES_NonAg.js",
          "name": "BIO_GBF8_SPECIES_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_GBF8_SPECIES_overview_Ag.js",
            "name": "BIO_GBF8_SPECIES_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_GBF8_SPECIES_overview_Am.js",
            "name": "BIO_GBF8_SPECIES_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_GBF8_SPECIES_overview_NonAg.js",
            "name": "BIO_GBF8_SPECIES_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_GBF8_SPECIES_overview_sum.js",
            "name": "BIO_GBF8_SPECIES_overview_sum",
          },
        },
      },
      "quality": {
        "Ag": {
          "path": "data/BIO_quality_Ag.js",
          "name": "BIO_quality_Ag",
        },
        "Ag Mgt": {
          "path": "data/BIO_quality_Am.js",
          "name": "BIO_quality_Am",
        },
        "Non-Ag": {
          "path": "data/BIO_quality_NonAg.js",
          "name": "BIO_quality_NonAg",
        },
        "overview": {
          "Ag": {
            "path": "data/BIO_quality_overview_Ag.js",
            "name": "BIO_quality_overview_Ag",
          },
          "Ag Mgt": {
            "path": "data/BIO_quality_overview_Am.js",
            "name": "BIO_quality_overview_Am",
          },
          "Non-Ag": {
            "path": "data/BIO_quality_overview_NonAg.js",
            "name": "BIO_quality_overview_NonAg",
          },
          "sum": {
            "path": "data/BIO_quality_overview_sum.js",
            "name": "BIO_quality_overview_sum",
          },
        },
        "ranking": {
          "path": "data/BIO_quality_ranking.js",
          "name": "BIO_quality_ranking",
        },
      },
      "ranking": {
        "path": "data/BIO_ranking.js",
        "name": "BIO_ranking",
      },
    },
    "Economics": {
      "Ag": {
        "path": "data/Economics_Ag.js",
        "name": "Economics_Ag",
      },
      "Ag Mgt": {
        "path": "data/Economics_Am.js",
        "name": "Economics_Am",
      },
      "Non-Ag": {
        "path": "data/Economics_overview_Non_Ag.js",
        "name": "Economics_overview_Non_Ag",
      },
      "overview": {
        "Ag": {
          "path": "data/Economics_overview_Ag.js",
          "name": "Economics_overview_Ag",
        },
        "Ag Mgt": {
          "path": "data/Economics_overview_Am.js",
          "name": "Economics_overview_Am",
        },
        "Non-Ag": {
          "path": "data/Economics_overview_Non_Ag.js",
          "name": "Economics_overview_Non_Ag",
        },
        "sum": {
          "path": "data/Economics_overview_sum.js",
          "name": "Economics_overview_sum",
        },
      },
      "ranking": {
        "path": "data/Economics_ranking.js",
        "name": "Economics_ranking",
      },
    },
    "GHG": {
      "Ag": {
        "path": "data/GHG_Ag.js",
        "name": "GHG_Ag",
      },
      "Ag Mgt": {
        "path": "data/GHG_Am.js",
        "name": "GHG_Am",
      },
      "Non-Ag": {
        "path": "data/GHG_NonAg.js",
        "name": "GHG_NonAg",
      },
      "overview": {
        "Ag": {
          "path": "data/GHG_overview_Ag.js",
          "name": "GHG_overview_Ag",
        },
        "Ag Mgt": {
          "path": "data/GHG_overview_Am.js",
          "name": "GHG_overview_Am",
        },
        "Non-Ag": {
          "path": "data/GHG_overview_NonAg.js",
          "name": "GHG_overview_NonAg",
        },
        "sum": {
          "path": "data/GHG_overview_sum.js",
          "name": "GHG_overview_sum",
        },
      },
      "ranking": {
        "path": "data/GHG_ranking.js",
        "name": "GHG_ranking",
      },
    },
    "Production": {
      "Ag": {
        "path": "data/Production_Ag.js",
        "name": "Production_Ag",
      },
      "Ag Mgt": {
        "path": "data/Production_Am.js",
        "name": "Production_Am",
      },
      "Non-Ag": {
        "path": "data/Production_NonAg.js",
        "name": "Production_NonAg",
      },
      "overview": {
        "achieve": {
          "path": "data/Production_overview_AUS_achive_percent.js",
          "name": "Production_overview_AUS_achive_percent",
        },
        "Domestic": {
          "path": "data/Production_overview_Domestic.js",
          "name": "Production_overview_Domestic",
        },
        "Exports": {
          "path": "data/Production_overview_Exports.js",
          "name": "Production_overview_Exports",
        },
        "Feed": {
          "path": "data/Production_overview_Feed.js",
          "name": "Production_overview_Feed",
        },
        "Imports": {
          "path": "data/Production_overview_Imports.js",
          "name": "Production_overview_Imports",
        },
        "sum": {
          "path": "data/Production_overview_demand_type.js",
          "name": "Production_overview_demand_type",
        },
      },
      "ranking": {
        "path": "data/Production_ranking.js",
        "name": "Production_ranking",
      },
    },
    "Water": {
      "NRM": {
        "Ag": {
          "path": "data/Water_Ag_NRM.js",
          "name": "Water_Ag_NRM",
        },
        "Ag Mgt": {
          "path": "data/Water_Am_NRM.js",
          "name": "Water_Am_NRM",
        },
        "Non-Ag": {
          "path": "data/Water_NonAg_NRM.js",
          "name": "Water_NonAg_NRM",
        },
        "overview": {
          "Ag": {
            "path": "data/Water_overview_NRM_Ag.js",
            "name": "Water_overview_NRM_Ag",
          },
          "Ag Mgt": {
            "path": "data/Water_overview_NRM_Am.js",
            "name": "Water_overview_NRM_Am",
          },
          "Non-Ag": {
            "path": "data/Water_overview_NRM_NonAg.js",
            "name": "Water_overview_NRM_NonAg",
          },
          "sum": {
            "path": "data/Water_overview_NRM_sum.js",
            "name": "Water_overview_NRM_sum",
          },
        },
        "ranking": {
          "path": "data/Water_ranking_NRM.js",
          "name": "Water_ranking_NRM",
        },
      },
      "Watershed": {
        "overview": {
          "path": "data/Water_overview_watershed.js",
          "name": "Water_overview_watershed",
        },
      },
    },
    "Supporting": {
      "info": {
        "path": "data/Supporting_info.js",
        "name": "Supporting_info",
      },
    },
  },
};