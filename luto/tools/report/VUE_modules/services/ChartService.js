// Chart Data Service
// This service provides data about charts for different metrics and categories

window.ChartService = {
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
        "Land-use": {
          path: "data/Area_overview_1_Land-use.js",
          name: "Area_overview_1_Land-use",
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
          path: "data/BIO_GBF2_Ag.js",
          name: "BIO_GBF2_Ag",
        },
        "Ag Mgt": {
          path: "data/BIO_GBF2_Am.js",
          name: "BIO_GBF2_Am",
        },
        "Non-Ag": {
          path: "data/BIO_GBF2_NonAg.js",
          name: "BIO_GBF2_NonAg",
        },
        overview: {
          sum: {
            path: "data/BIO_GBF2_overview_sum.js",
            name: "BIO_GBF2_overview_sum",
          },
        },
      },
      GBF3_NVIS: {
        Sum: {
          path: "data/BIO_GBF3_NVIS_Sum.js",
          name: "BIO_GBF3_NVIS_Sum",
          paged: true,
        },
        Ag: {
          path: "data/BIO_GBF3_NVIS_Ag.js",
          name: "BIO_GBF3_NVIS_Ag",
          paged: true,
        },
        "Ag Mgt": {
          path: "data/BIO_GBF3_NVIS_Am.js",
          name: "BIO_GBF3_NVIS_Am",
          paged: true,
        },
        "Non-Ag": {
          path: "data/BIO_GBF3_NVIS_NonAg.js",
          name: "BIO_GBF3_NVIS_NonAg",
          paged: true,
        },
        overview: {
          sum: {
            path: "data/BIO_GBF3_NVIS_overview_sum.js",
            name: "BIO_GBF3_NVIS_overview_sum",
            paged: true,
          },
        },
        // every (region, item) with a target: its score and target per year (Home's target cards)
        targets: {
          path: "data/BIO_GBF3_NVIS_target_summary.js",
          name: "BIO_GBF3_NVIS_target_summary",
        },
      },
      GBF4_SNES: {
        Sum: {
          path: "data/BIO_GBF4_SNES_Sum.js",
          name: "BIO_GBF4_SNES_Sum",
          paged: true,
        },
        Ag: {
          path: "data/BIO_GBF4_SNES_Ag.js",
          name: "BIO_GBF4_SNES_Ag",
          paged: true,
        },
        "Ag Mgt": {
          path: "data/BIO_GBF4_SNES_Am.js",
          name: "BIO_GBF4_SNES_Am",
          paged: true,
        },
        "Non-Ag": {
          path: "data/BIO_GBF4_SNES_NonAg.js",
          name: "BIO_GBF4_SNES_NonAg",
          paged: true,
        },
        overview: {
          sum: {
            path: "data/BIO_GBF4_SNES_overview_sum.js",
            name: "BIO_GBF4_SNES_overview_sum",
            paged: true,
          },
        },
        // every (region, item) with a target: its score and target per year (Home's target cards)
        targets: {
          path: "data/BIO_GBF4_SNES_target_summary.js",
          name: "BIO_GBF4_SNES_target_summary",
        },
      },
      GBF4_ECNES: {
        Sum: {
          path: "data/BIO_GBF4_ECNES_Sum.js",
          name: "BIO_GBF4_ECNES_Sum",
          paged: true,
        },
        Ag: {
          path: "data/BIO_GBF4_ECNES_Ag.js",
          name: "BIO_GBF4_ECNES_Ag",
          paged: true,
        },
        "Ag Mgt": {
          path: "data/BIO_GBF4_ECNES_Am.js",
          name: "BIO_GBF4_ECNES_Am",
          paged: true,
        },
        "Non-Ag": {
          path: "data/BIO_GBF4_ECNES_NonAg.js",
          name: "BIO_GBF4_ECNES_NonAg",
          paged: true,
        },
        overview: {
          sum: {
            path: "data/BIO_GBF4_ECNES_overview_sum.js",
            name: "BIO_GBF4_ECNES_overview_sum",
            paged: true,
          },
        },
        // every (region, item) with a target: its score and target per year (Home's target cards)
        targets: {
          path: "data/BIO_GBF4_ECNES_target_summary.js",
          name: "BIO_GBF4_ECNES_target_summary",
        },
      },
      GBF8_GROUP: {
        Ag: {
          path: "data/BIO_GBF8_GROUP_Ag.js",
          name: "BIO_GBF8_GROUP_Ag",
        },
        "Ag Mgt": {
          path: "data/BIO_GBF8_GROUP_Am.js",
          name: "BIO_GBF8_GROUP_Am",
        },
        "Non-Ag": {
          path: "data/BIO_GBF8_GROUP_NonAg.js",
          name: "BIO_GBF8_GROUP_NonAg",
        },
        overview: {
          sum: {
            path: "data/BIO_GBF8_GROUP_overview_sum.js",
            name: "BIO_GBF8_GROUP_overview_sum",
          },
        },
      },
      GBF8_SPECIES: {
        Ag: {
          path: "data/BIO_GBF8_SPECIES_Ag.js",
          name: "BIO_GBF8_SPECIES_Ag",
        },
        "Ag Mgt": {
          path: "data/BIO_GBF8_SPECIES_Am.js",
          name: "BIO_GBF8_SPECIES_Am",
        },
        "Non-Ag": {
          path: "data/BIO_GBF8_SPECIES_NonAg.js",
          name: "BIO_GBF8_SPECIES_NonAg",
        },
        overview: {
          sum: {
            path: "data/BIO_GBF8_SPECIES_overview_sum.js",
            name: "BIO_GBF8_SPECIES_overview_sum",
          },
        },
        // every species with a target: its score and target per year (Home's target cards)
        targets: {
          path: "data/BIO_GBF8_SPECIES_target_summary.js",
          name: "BIO_GBF8_SPECIES_target_summary",
        },
      },
      quality: {
        Ag: {
          path: "data/BIO_quality_Ag.js",
          name: "BIO_quality_Ag",
        },
        "Ag Mgt": {
          path: "data/BIO_quality_Am.js",
          name: "BIO_quality_Am",
        },
        "Non-Ag": {
          path: "data/BIO_quality_NonAg.js",
          name: "BIO_quality_NonAg",
        },
        overview: {
          sum: {
            path: "data/BIO_quality_overview_sum.js",
            name: "BIO_quality_overview_sum",
          },
        },
        ranking: {
          path: "data/BIO_quality_ranking.js",
          name: "BIO_quality_ranking",
        },
      },
      ranking: {
        path: "data/BIO_ranking.js",
        name: "BIO_ranking",
      },
    },
    Economics: {
      Sum: {
        Profit: { path: "data/Economics_Sum.js", name: "Economics_Sum" },
      },
      Ag: {
        Profit: {
          path: "data/Economics_Ag_profit.js",
          name: "Economics_Ag_profit",
        },
        Revenue: {
          path: "data/Economics_Ag_revenue.js",
          name: "Economics_Ag_revenue",
        },
        Cost: { path: "data/Economics_Ag_cost.js", name: "Economics_Ag_cost" },
        "Transition (Ag2Ag)": {
          path: "data/Economics_Ag_transition_ag2ag.js",
          name: "Economics_Ag_transition_ag2ag",
        },
        "Transition (NonAg2Ag)": {
          path: "data/Economics_Ag_transition_nonag2ag.js",
          name: "Economics_Ag_transition_nonag2ag",
        },
      },
      "Ag Mgt": {
        Profit: {
          path: "data/Economics_Am_profit.js",
          name: "Economics_Am_profit",
        },
        Revenue: {
          path: "data/Economics_Am_revenue.js",
          name: "Economics_Am_revenue",
        },
        Cost: { path: "data/Economics_Am_cost.js", name: "Economics_Am_cost" },
      },
      "Non-Ag": {
        Profit: {
          path: "data/Economics_NonAg_profit.js",
          name: "Economics_NonAg_profit",
        },
        Revenue: {
          path: "data/Economics_NonAg_revenue.js",
          name: "Economics_NonAg_revenue",
        },
        Cost: {
          path: "data/Economics_NonAg_cost.js",
          name: "Economics_NonAg_cost",
        },
        "Transition (Ag2NonAg)": {
          path: "data/Economics_NonAg_transition_ag2nonag.js",
          name: "Economics_NonAg_transition_ag2nonag",
        },
        "Transition (NonAg2NonAg)": {
          path: "data/Economics_NonAg_transition_nonag2nonag.js",
          name: "Economics_NonAg_transition_nonag2nonag",
        },
      },
      overview: {
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
      Sum: {
        path: "data/GHG_Sum.js",
        name: "GHG_Sum",
      },
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
        sum: {
          path: "data/GHG_overview_sum.js",
          name: "GHG_overview_sum",
        },
      },
      ranking: {
        path: "data/GHG_ranking.js",
        name: "GHG_ranking",
      },
    },
    Production: {
      Sum: {
        path: "data/Production_Sum.js",
        name: "Production_Sum",
      },
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
        // commodity -> [series]: the demand types and what the run has to produce, one commodity at a time
        commodity: {
          path: "data/Production_overview_demand_commodity.js",
          name: "Production_overview_demand_commodity",
        },
      },
      ranking: {
        path: "data/Production_ranking.js",
        name: "Production_ranking",
      },
    },
    Water: {
      NRM: {
        Sum: {
          path: "data/Water_Sum_NRM.js",
          name: "Water_Sum_NRM",
        },
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
    Renewable: {
      "Ag Mgt": {
        path: "data/Renewable_energy_Am.js",
        name: "Renewable_energy_Am",
      },
    },
    Inputs: {
      catalogue: { path: "data/Inputs_catalogue.js", name: "Inputs_catalogue" },
      tables: { path: "data/Inputs_tables.js", name: "Inputs_tables" },
      bundles: { path: "data/Inputs_bundles.js", name: "Inputs_bundles" },
      yield_trend: { path: "data/Inputs_yield_trend.js", name: "Inputs_yield_trend" },
      adoption_caps: { path: "data/Inputs_adoption_caps.js", name: "Inputs_adoption_caps" },
      transition_matrix: { path: "data/Inputs_transition_matrix.js", name: "Inputs_transition_matrix" },
      species: { path: "data/Inputs_species.js", name: "Inputs_species" },
      years: { path: "data/Inputs_years.js", name: "Inputs_years" },
    },
    Transition: {
      Area: {
        start_end: {
          path: "data/Transition_start_end_area.js",
          name: "Transition_start_end_area",
        },
        Ag2Ag: {
          path: "data/Transition_ag2ag_area.js",
          name: "Transition_ag2ag_area",
        },
        Ag2NonAg: {
          path: "data/Transition_ag2nonag_area.js",
          name: "Transition_ag2nonag_area",
        },
      },
      Cost: {
        Ag2Ag: {
          path: "data/Transition_ag2ag_cost.js",
          name: "Transition_ag2ag_cost",
        },
        Ag2NonAg: {
          path: "data/Transition_ag2nonag_cost.js",
          name: "Transition_ag2nonag_cost",
        },
      },
    },
    Supporting: {
      info: {
        path: "data/Supporting_info.js",
        name: "Supporting_info",
      },
    },
  },
};
