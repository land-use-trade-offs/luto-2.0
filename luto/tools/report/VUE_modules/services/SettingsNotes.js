// The notes of the settings, shown in the pop-up behind the i of each setting on the Settings page.
//
// Hand-written for a reader of the report, from the comments of luto/settings.py and the code that reads each
// setting. This file is the only source of the notes: nothing reads settings.py for them. When a setting, its
// options or its meaning change there, or a setting is added, update its entry here by hand; a setting with no
// entry shows "No note for this setting yet".
//
//   SettingsNotes[<setting>] = {
//     what:    one sentence on what the setting does,
//     detail:  [further sentences, each on its own line],
//     options: [[value, what the value means], ...]      (the value of the run is marked in the pop-up)
//   }
window.SettingsNotes = {
  VERSION: {
    what: "The version of the LUTO model that produced this run.",
  },
  INPUT_DIR: {
    what: "The folder the model reads its prepared input data from.",
  },
  OUTPUT_DIR: {
    what: "The folder runs are written to, one time-stamped sub-folder per run.",
  },
  RAW_DATA: {
    what: "The folder of raw source data that the data preparation step turns into model inputs.",
    detail: [
      "It is not read during a run.",
    ],
  },
  SSP: {
    what: "The climate and socio-economic pathway of the run.",
    detail: [
      "The first digit is the Shared Socioeconomic Pathway, the last two the warming level (245 = SSP2, RCP 4.5).",
      "It drives food demand, climate impacts on yield and on water, and the climate suitability of species.",
    ],
    options: [
      ["126", "SSP1, RCP 2.6: sustainability, low warming."],
      ["245", "SSP2, RCP 4.5: middle of the road."],
      ["370", "SSP3, RCP 7.0: regional rivalry, high warming. Its yield impacts are read from RCP 6.0, the nearest pathway the yield data has."],
      ["585", "SSP5, RCP 8.5: fossil-fuelled growth."],
    ],
  },
  YIELD_RCP: {
    what: "The warming pathway of the yield impact data, derived from SSP.",
    detail: [
      "Not set by hand: rcp4p5 is RCP 4.5.",
      "It only picks the file of the climate change impact on yield; water and biodiversity follow SSP. That data has no RCP 7.0, so SSP 370 shows rcp6p0 here.",
    ],
  },
  SCENARIO: {
    what: "The socio-economic pathway of the demand scenario, derived from SSP.",
    detail: [
      "Not set by hand: SSP 245 gives SSP2.",
    ],
  },
  RESFACTOR: {
    what: "How coarse the map of the run is.",
    detail: [
      "1 runs every 1 km cell. A value of 5 keeps the middle cell of every 5 x 5 block, which stands for the whole block.",
      "Coarser runs are faster and are used for testing.",
    ],
  },
  SIM_YEARS: {
    what: "The years the model solves, in order.",
    detail: [
      "Each year is solved from the land use of the year before it.",
    ],
  },
  OBJECTIVE: {
    what: "What the model optimises.",
    options: [
      ["maxprofit", "Maximise profit: revenue less costs. Over-production is held back by DEMAND_BOUNDS."],
      ["mincost", "Minimise the cost of meeting demand and the targets."],
    ],
  },
  DIET_DOM: {
    what: "The diet of Australians, which sets domestic food demand.",
    options: [
      ["BAU", "The projected trend continues."],
      ["FLX", "Flexitarian (EAT-Lancet)."],
      ["VEG", "Vegetarian."],
      ["VGN", "Vegan."],
    ],
  },
  DIET_GLOB: {
    what: "The diet of the rest of the world, which sets the demand for Australian exports.",
    options: [
      ["BAU", "The projected trend continues."],
      ["FLX", "Flexitarian (EAT-Lancet)."],
      ["VEG", "Vegetarian."],
      ["VGN", "Vegan."],
    ],
  },
  CONVERGENCE: {
    what: "The year by which the change in diet and waste is complete.",
    detail: [
      "The change is gradual up to this year and held after it.",
    ],
    options: [
      ["2050", "A fast transition."],
      ["2100", "A slow transition."],
    ],
  },
  IMPORT_TREND: {
    what: "How the share of food that Australia imports changes over time.",
    options: [
      ["Static", "Each commodity keeps its 2010 import share."],
      ["Trend", "The import share follows its historical rate of change."],
    ],
  },
  WASTE: {
    what: "How much food is lost and wasted along the supply chain.",
    options: [
      ["1", "Current levels of loss and waste."],
      ["0.5", "Loss and waste are halved by the convergence year."],
    ],
  },
  FEED_EFFICIENCY: {
    what: "How much feed livestock need for each unit they produce.",
    options: [
      ["BAU", "Current feed conversion."],
      ["High", "Improved feed conversion, so less feed demand."],
    ],
  },
  APPLY_DEMAND_MULTIPLIERS: {
    what: "Whether demand is adjusted by the AusTIMES energy model.",
    detail: [
      "When on, the demand for sugar, winter cereals and winter oilseeds is scaled for the crops that the energy pathway uses as bioenergy.",
      "It only applies when GHG_EMISSIONS_LIMITS is not off, since that setting names the pathway.",
    ],
  },
  DEMAND_BOUNDS: {
    what: "How far production may depart from demand, per commodity.",
    detail: [
      "Each commodity has a lower and an upper multiplier on its demand: [1.0, 1.0] means demand must be met exactly.",
      "Sheep and beef produce several products in fixed proportions (meat, wool, live exports), which cannot all match demand at once, so some of them are given a wider range.",
    ],
  },
  OFF_LAND_COMMODITIES: {
    what: "Commodities that are not produced on the land of the model.",
    detail: [
      "They are raised in sheds and ponds, so they take no land and have no demand constraint.",
      "Their emissions are still counted in the agriculture emissions limit.",
    ],
  },
  EGGS_AVG_WEIGHT: {
    what: "The average weight of an egg, in grams.",
    detail: [
      "Used to convert egg demand between numbers and tonnes.",
    ],
  },
  PRODUCTIVITY_TREND: {
    what: "How fast agricultural productivity grows over time.",
    detail: [
      "A multiplier on 2010 yield for each product and year.",
    ],
    options: [
      ["BAU", "The business-as-usual projection of the LUTO data."],
      ["LOW", "ABARES low projection."],
      ["MEDIUM", "ABARES medium projection."],
      ["HIGH", "ABARES high projection."],
      ["VERY_HIGH", "ABARES very high projection."],
    ],
  },
  CO2_FERT: {
    what: "Whether higher CO2 in the air is assumed to raise crop yields.",
    detail: [
      "It selects which FAO GAEZ v4 climate impact data the yields are scaled by.",
    ],
    options: [
      ["on", "Yields include the CO2 fertilisation effect."],
      ["off", "Yields respond to climate change only."],
    ],
  },
  CARBON_EFFECTS_WINDOW: {
    what: "The number of years over which the carbon stored by a planting is averaged.",
    detail: [
      "Trees store carbon quickly at first and slowly later. The model counts the total stored after this many years, divided by the same number of years, as the yearly rate.",
      "A short window gives a higher yearly rate. Allowed values: 50, 60, 70, 80 or 90.",
    ],
  },
  RISK_OF_REVERSAL: {
    what: "A discount on the carbon of trees and debris for the risk that it is released again.",
    detail: [
      "It follows the risk-of-reversal buffer of the Emissions Reduction Fund; reasonable values are 0.05 to 0.25.",
      "0 applies no further discount on top of fire risk.",
      "Soil carbon is not discounted.",
    ],
  },
  FIRE_RISK: {
    what: "Which estimate of fire loss is used to discount the carbon of trees and debris.",
    options: [
      ["low", "The optimistic estimate (5th percentile): about 80 % of the carbon survives on average."],
      ["med", "The median estimate: about 89 % survives."],
      ["high", "The 95th percentile estimate: about 93 % survives."],
    ],
  },
  GHG_EMISSIONS_LIMITS: {
    what: "Whether the run is held to greenhouse gas limits.",
    detail: [
      "When on, two limits apply in the years they bind: the emissions of the agriculture sector, and the net emissions of the land sector (LULUCF).",
      "Both follow national inventory history and government baseline projections. The level does not change these limits: it only names the AusTIMES pathway used for demand multipliers and the carbon price.",
    ],
    options: [
      ["off", "No emissions limit."],
      ["low", "Limits on; the AusTIMES 1.8 degree pathway."],
      ["high", "Limits on; the AusTIMES 1.5 degree pathway."],
    ],
  },
  GHG_TARGETS_DICT: {
    what: "The AusTIMES pathway that each level of GHG_EMISSIONS_LIMITS names.",
    detail: [
      "The pathway picks the demand multipliers and, if chosen, the carbon price. It no longer sets the emissions limits.",
    ],
  },
  GHG_BENCHMARK_ROWS: {
    what: "Emissions limits that are reported but not enforced.",
    detail: [
      "Empty: both limits are enforced.",
      "AG or LULUCF listed here: that limit is still computed and compared with its series in the outputs, but the model is free to exceed it.",
    ],
  },
  CARBON_PRICES_FIELD: {
    what: "Which carbon price the model pays for stored carbon and charges for emissions.",
    options: [
      ["CONSTANT", "One fixed price in every year, set by CARBON_PRICE_COSTANT."],
      ["AS_GHG", "The price path of the AusTIMES pathway named by GHG_EMISSIONS_LIMITS."],
      ["Default", "The default price path of the carbon price workbook."],
      ["100", "100 AUD per tonne in every year."],
      ["1.8C 67%", "The AusTIMES price path for 1.8 degrees."],
      ["1.5C 50%", "The AusTIMES price path for 1.5 degrees, 50 % chance."],
      ["1.5C 67%", "The AusTIMES price path for 1.5 degrees, 67 % chance."],
    ],
  },
  CARBON_PRICE_COSTANT: {
    what: "The fixed carbon price, in AUD per tonne of CO2e.",
    detail: [
      "Only used when CARBON_PRICES_FIELD is CONSTANT.",
      "0 means carbon has no price: plantings earn nothing for the carbon they store.",
    ],
  },
  USE_GHG_SCOPE_1: {
    what: "Whether only direct, on-farm emissions are counted.",
    detail: [
      "Direct (scope 1) emissions are those of the national inventory: soil nitrous oxide for crops; enteric methane, manure, dung and urine, and leaching for livestock.",
      "Emissions from electricity, fuel and the making of fertiliser and chemicals belong to the energy and industry accounts. Counting them here too would count their reduction twice.",
    ],
    options: [
      ["True", "Only direct emissions count."],
      ["False", "Every emission source in the data counts."],
    ],
  },
  CROP_GHG_SCOPE_1: {
    what: "The crop emission sources that count as direct emissions.",
    detail: [
      "Used when USE_GHG_SCOPE_1 is True. Soil nitrous oxide only.",
    ],
  },
  LVSTK_GHG_SCOPE_1: {
    what: "The livestock emission sources that count as direct emissions.",
    detail: [
      "Used when USE_GHG_SCOPE_1 is True: dung and urine, enteric methane, leaching and runoff, and manure management.",
    ],
  },
  AMORTISE_UPFRONT_COSTS: {
    what: "Whether one-off costs are spread over time.",
    options: [
      ["True", "Establishment and transition costs are paid as equal yearly amounts over AMORTISATION_PERIOD at DISCOUNT_RATE."],
      ["False", "They are paid in full in the year the land use changes."],
    ],
  },
  DISCOUNT_RATE: {
    what: "The interest rate used to spread one-off costs over time.",
    detail: [
      "0.07 is 7 % a year. Only used when AMORTISE_UPFRONT_COSTS is True.",
    ],
  },
  AMORTISATION_PERIOD: {
    what: "The number of years one-off costs are spread over.",
    detail: [
      "Only used when AMORTISE_UPFRONT_COSTS is True.",
    ],
  },
  TRANSITION_COST_MULT: {
    what: "A multiplier on the cost of changing land use.",
    detail: [
      "It scales establishment cost, water licence cost and the carbon penalty of clearing together.",
      "1 is the baseline; below 1 makes change cheaper, above 1 makes it harder.",
    ],
  },
  DYNAMIC_PRICE: {
    what: "Whether commodity prices respond to demand.",
    options: [
      ["True", "A price rises when demand is above its 2010 level and falls when below, by the demand elasticity of the commodity."],
      ["False", "Prices do not respond to demand."],
    ],
  },
  REMOVE_IRRIG_COST: {
    what: "The cost of removing irrigation, in AUD per hectare.",
    detail: [
      "Paid once when irrigated land becomes dryland or leaves agriculture.",
    ],
  },
  NEW_IRRIG_COST: {
    what: "The cost of setting up irrigation, in AUD per hectare.",
    detail: [
      "Paid once when dryland becomes irrigated.",
    ],
  },
  INCLUDE_WATER_LICENSE_COSTS: {
    what: "Whether changing land use buys or sells water licences.",
    detail: [
      "When on, a land use that needs more water pays for the extra licence, and one that needs less sells it back.",
      "With it on, irrigators in the Murray-Darling Basin tend to sell water when profit is maximised.",
    ],
    options: [
      ["1", "Licence costs are counted."],
      ["0", "Licence costs are ignored."],
    ],
  },
  FENCING_COST_PER_M: {
    what: "The cost of fencing, in AUD per metre.",
    detail: [
      "Paid when riparian plantings, agroforestry or farm forestry are established.",
    ],
  },
  AG_MANAGEMENTS: {
    what: "Which agricultural management options the model may use.",
    detail: [
      "An option set to False is not available in this run.",
    ],
  },
  AG_MANAGEMENTS_TO_LAND_USES: {
    what: "The land uses each agricultural management option can be applied to.",
  },
  AG_MANAGEMENTS_REVERSIBLE: {
    what: "Whether an agricultural management option can be dropped once adopted.",
    detail: [
      "False locks the option in: a cell that adopts it keeps it, and its land use, in all later years.",
    ],
  },
  TECH_ADOPT_MULT: {
    what: "A multiplier on how widely new farm technologies can be adopted.",
    detail: [
      "It scales the adoption ceilings of Asparagopsis, precision agriculture, AgTech EI and biochar.",
      "1 is the baseline; a ceiling never exceeds 100 % of the land.",
    ],
  },
  AG_MANAGEMENT_OBSERVED_CAPS: {
    what: "Limits agricultural management to the adoption that has actually been observed.",
    detail: [
      "None: no such limit.",
      "A file name: each option and land use is capped at its observed level, held flat after the last observed year. Used for calibration runs.",
    ],
  },
  SAVBURN_COST_HA_YR: {
    what: "The cost of early dry season savanna burning, in AUD per hectare and year.",
  },
  AGRICULTURAL_MANAGEMENT_USE_THRESHOLD: {
    what: "The smallest share of a cell at which a management option is reported as in use.",
    detail: [
      "It only tidies the outputs: smaller shares are left off the maps. It does not affect the solution.",
    ],
  },
  HIR_PRODUCTIVITY_CONTRIBUTION: {
    what: "The share of livestock production that land keeps under human-induced regeneration.",
    detail: [
      "0.5 means the land produces half of what it would without regeneration.",
    ],
  },
  HIR_CEILING_PERCENTAGE: {
    what: "How much of the benefit of destocking human-induced regeneration achieves.",
    detail: [
      "0.8 means regeneration reaches 80 % of the carbon and biodiversity gain of fully destocked natural land.",
    ],
  },
  HIR_RATE_FILE: {
    what: "Where the carbon that regeneration and destocked land regrow comes from.",
    detail: [
      "None: the carbon stock of natural land, spread over the regrowth window.",
      "A file name: the rate of regrowth measured from the registered regeneration projects (credits issued per hectare, by region), the same on every land use. Regions with no registered project have a rate of 0.",
    ],
  },
  HIR_BASELINE: {
    what: "Whether regeneration projects that already exist are left out of what the model can claim.",
    detail: [
      "None: not applied.",
      "A folder name: the regrowth and destocking of existing registered projects is taken as given, since the national inventory already counts them. The model is credited only for regeneration beyond them.",
    ],
  },
  BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining human-induced regeneration on beef land.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
    ],
  },
  SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining human-induced regeneration on sheep land.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
    ],
  },
  LVSTK_K_FILE: {
    what: "An optional calibration of livestock stocking rates.",
    detail: [
      "None: stocking rates come from the standard feed requirement data.",
      "A file name: each region and livestock type uses its own calibrated factor instead.",
    ],
  },
  STUBBLE_DSE_FILE: {
    what: "Whether sheep graze the stubble left after cereal crops.",
    detail: [
      "None: no stubble grazing.",
      "A file name: cereal land also carries sheep at the rate in the file. Their meat, wool and live exports count toward sheep demand, and their emissions toward the agriculture limit.",
    ],
  },
  STUBBLE_LAND_USES: {
    what: "The cereal land uses whose stubble sheep may graze.",
    detail: [
      "Only used when STUBBLE_DSE_FILE is set.",
    ],
  },
  LUS_CROPPING: {
    what: "The land uses treated as broadacre cropping.",
    detail: [
      "Farm technology options read their effects by land-use type; this is the cropping type.",
    ],
  },
  LUS_INT_CROPPING: {
    what: "The land uses treated as intensive cropping.",
    detail: [
      "Farm technology options read their effects by land-use type; this is the intensive cropping type.",
    ],
  },
  LUS_HORTICULTURE: {
    what: "The land uses treated as horticulture.",
    detail: [
      "Farm technology options read their effects by land-use type; this is the horticulture type.",
    ],
  },
  LU2TYPE: {
    what: "The type of each crop land use: cropping, intensive cropping or horticulture.",
    detail: [
      "Built from the three lists of land uses. Precision agriculture, AgTech EI and biochar read their effect on a land use from the data of its type.",
    ],
  },
  NON_AG_LAND_USES: {
    what: "Which non-agricultural land uses the model may use.",
    detail: [
      "A land use set to False is not available in this run.",
    ],
  },
  NON_AG_LAND_USES_REVERSIBLE: {
    what: "Whether a non-agricultural land use can be removed once established.",
    detail: [
      "False means land that is planted stays planted in all later years.",
      "Allowing removal can make later years infeasible: a catchment may sit close to its water limit and then be unable to undo the plantings that use the water.",
    ],
  },
  EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining environmental plantings.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
    ],
  },
  EP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR: {
    what: "The value of ecosystem services credited to environmental plantings.",
    detail: [
      "A yearly credit for ecosystem services, in AUD per hectare and year, subtracted from the maintenance cost of the land use.",
      "0 means no credit is counted.",
    ],
  },
  CP_BLOCK_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining carbon plantings (block).",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
    ],
  },
  CP_BLOCK_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR: {
    what: "The value of ecosystem services credited to carbon plantings (block).",
    detail: [
      "A yearly credit for ecosystem services, in AUD per hectare and year, subtracted from the maintenance cost of the land use.",
      "0 means no credit is counted.",
    ],
  },
  CP_BELT_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining the tree belts of farm forestry.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
      "Applied to the tree share of the hectare.",
    ],
  },
  CP_BELT_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR: {
    what: "The value of ecosystem services credited to the tree belts of farm forestry.",
    detail: [
      "A yearly credit for ecosystem services, in AUD per hectare and year, subtracted from the maintenance cost of the land use.",
      "0 means no credit is counted.",
    ],
  },
  CP_BELT_ROW_WIDTH: {
    what: "The width of a tree belt in farm forestry, in metres.",
  },
  CP_BELT_ROW_SPACING: {
    what: "The width of the pasture alley between two tree belts in farm forestry, in metres.",
  },
  CP_BELT_PROPORTION: {
    what: "The share of a farm forestry hectare that is under trees.",
    detail: [
      "Belt width divided by belt width plus alley. The rest of the hectare stays under sheep or beef.",
      "Cost, revenue, carbon, water use and biodiversity are all mixed in this proportion.",
    ],
  },
  CP_BELT_FENCING_LENGTH: {
    what: "The length of fence needed per hectare of farm forestry, in metres.",
    detail: [
      "Worked out from the belt width and spacing; it sets the fencing cost.",
    ],
  },
  RP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining riparian plantings.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
    ],
  },
  RP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR: {
    what: "The value of ecosystem services credited to riparian plantings.",
    detail: [
      "A yearly credit for ecosystem services, in AUD per hectare and year, subtracted from the maintenance cost of the land use.",
      "0 means no credit is counted.",
    ],
  },
  RIPARIAN_PLANTING_BUFFER_WIDTH: {
    what: "The width of the strip planted on each side of a stream, in metres.",
    detail: [
      "Together with the length of streams in a cell, it sets the share of the cell that riparian plantings can take.",
    ],
  },
  RIPARIAN_PLANTING_TORTUOSITY_FACTOR: {
    what: "A factor that turns the length of streams into the length of fence riparian plantings need.",
    detail: [
      "Fence length = 2 sides x this factor x stream length. It sets the fencing cost.",
    ],
  },
  AF_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR: {
    what: "The cost of maintaining the tree belts of agroforestry.",
    detail: [
      "A yearly cost in the objective, in AUD per hectare and year.",
      "Applied to the tree share of the hectare.",
    ],
  },
  AF_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR: {
    what: "The value of ecosystem services credited to the tree belts of agroforestry.",
    detail: [
      "A yearly credit for ecosystem services, in AUD per hectare and year, subtracted from the maintenance cost of the land use.",
      "0 means no credit is counted.",
    ],
  },
  AGROFORESTRY_ROW_WIDTH: {
    what: "The width of a tree belt in agroforestry, in metres.",
  },
  AGROFORESTRY_ROW_SPACING: {
    what: "The width of the pasture alley between two tree belts in agroforestry, in metres.",
  },
  AF_PROPORTION: {
    what: "The share of an agroforestry hectare that is under trees.",
    detail: [
      "Belt width divided by belt width plus alley. The rest of the hectare stays under sheep or beef.",
      "Cost, revenue, carbon, water use and biodiversity are all mixed in this proportion.",
    ],
  },
  AF_FENCING_LENGTH_HA: {
    what: "The length of fence needed per hectare of agroforestry, in metres.",
    detail: [
      "Worked out from the belt width and spacing; it sets the fencing cost.",
    ],
  },
  NON_AGRICULTURAL_LU_BASE_CODE: {
    what: "The number added to the codes of non-agricultural land uses on the land-use map.",
    detail: [
      "It keeps them apart from agricultural land uses: with 100, the first non-agricultural land use is mapped as 100.",
    ],
  },
  WATER_LIMITS: {
    what: "Whether the run is held to water limits.",
    detail: [
      "When on, the water left in each region must stay above a share of its historical yield (WATER_STRESS).",
      "Where climate change alone would take a region below that level, its limit is lowered to what the climate leaves, so the region stays feasible.",
    ],
    options: [
      ["on", "Water limits apply."],
      ["off", "No water limits."],
    ],
  },
  WATER_CLIMATE_CHANGE_IMPACT: {
    what: "Whether climate change alters how much water runs off the land.",
    options: [
      ["on", "Runoff follows the climate pathway of the run, year by year."],
      ["off", "Runoff stays at its historical level."],
    ],
  },
  WATER_REGION_DEF: {
    what: "The regions that water limits are applied to.",
    detail: [
      "Both are Bureau of Meteorology Geofabric regions.",
    ],
    options: [
      ["River Region", "River regions: many small regions."],
      ["Drainage Division", "Drainage divisions: a few large regions."],
    ],
  },
  WATER_STRESS: {
    what: "The share of its historical water yield that each region must keep.",
    detail: [
      "0.6 means the water left in a region must be at least 60 % of its historical yield.",
      "Domestic and industrial water use is taken out before the comparison.",
    ],
  },
  LIVESTOCK_DRINKING_WATER: {
    what: "Whether the drinking water of livestock counts as water use.",
    detail: [
      "Counting it can make dry regions such as the Pilbara hard to keep within their water limit.",
    ],
    options: [
      ["1", "Counted."],
      ["0", "Left out."],
    ],
  },
  BIO_QUALITY_LAYERS: {
    what: "The biodiversity quality layers the model can choose from.",
    detail: [
      "See BIO_QUALITY_LAYER for the one used in this run.",
    ],
  },
  BIO_QUALITY_LAYER: {
    what: "The layer that says how much each cell matters for biodiversity.",
    detail: [
      "It ranks every cell from least to most important. The rank is the weight of the cell in the biodiversity score, and it decides which cells are GBF2 priority degraded areas.",
    ],
    options: [
      ["Suitability", "Ranked across about 10,000 species by climate suitability."],
      ["SNES_likely", "Ranked across threatened species, where they are likely to occur."],
      ["SNES_likely_may", "Threatened species, where they are likely or may occur."],
      ["ECNES_likely", "Ranked across threatened ecological communities, likely range."],
      ["ECNES_likely_may", "Threatened ecological communities, likely or possible range."],
      ["MNES_likely", "Threatened species and communities together, likely range."],
      ["MNES_likely_may", "Threatened species and communities together, likely or possible range."],
      ["RHI", "The Relative Habitat Importance layer published by DCCEEW."],
    ],
  },
  CONNECTIVITY_SOURCE: {
    what: "Whether biodiversity quality is weighted by how connected a cell is to other habitat.",
    options: [
      ["NCI", "Weighted by the National Connectivity Index of DCCEEW."],
      ["DWI", "Weighted by the distance to the nearest natural land."],
      ["NONE", "No connectivity weighting."],
    ],
  },
  CONNECTIVITY_LB: {
    what: "How strongly connectivity affects biodiversity quality.",
    detail: [
      "It is the share of its quality that the least connected cell keeps: 0.7 means connectivity can lower quality by at most 30 %.",
      "Allowed values: 0.5, 0.6, 0.7, 0.8 or 0.9.",
    ],
  },
  HCAS_CONTRIBUTION_PERCENTILE: {
    what: "How much biodiversity value agricultural land keeps, by land use.",
    detail: [
      "The values come from the Habitat Condition Assessment System (HCAS): the condition observed on land under each use, from 0 to 1.",
      "For example, a value of 0.22 for beef on modified land means that land keeps 22 % of the value of undisturbed land.",
    ],
    options: [
      ["10", "The 10th percentile of observed condition (pessimistic)."],
      ["25", "The 25th percentile."],
      ["50", "The median."],
      ["75", "The 75th percentile."],
      ["90", "The 90th percentile (optimistic)."],
      ["CSV_DEFINED", "The values set by hand in the input table."],
      ["AG_UNIFORM", "One value for all modified agricultural land (HCAS_AG_UNIFORM_CONTRIBUTION)."],
    ],
  },
  HCAS_AG_UNIFORM_CONTRIBUTION: {
    what: "The single biodiversity value given to all modified agricultural land.",
    detail: [
      "Only used when HCAS_CONTRIBUTION_PERCENTILE is AG_UNIFORM. Natural land keeps its own values.",
      "0 means only natural land and non-agricultural land uses count for biodiversity.",
    ],
  },
  BIO_CONTRIBUTION_LDS: {
    what: "The share of its biodiversity value that savanna keeps under late dry season fires.",
    detail: [
      "Hot late-season fires damage habitat. Land eligible for savanna burning counts at this share until early dry season burning is applied, which restores its full value.",
    ],
  },
  BIO_CONTRIBUTION_ENV_PLANTING: {
    what: "The share of full biodiversity value that environmental plantings provide.",
    detail: [
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK: {
    what: "The share of full biodiversity value that carbon plantings (block) provide.",
    detail: [
      "Low, because they are single-species plantings.",
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_CARBON_PLANTING_BELT: {
    what: "The share of full biodiversity value that the tree belts of farm forestry provide.",
    detail: [
      "Low, because they are single-species plantings.",
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_RIPARIAN_PLANTING: {
    what: "The share of full biodiversity value that riparian plantings provide.",
    detail: [
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_AGROFORESTRY: {
    what: "The share of full biodiversity value that the tree belts of agroforestry provide.",
    detail: [
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_BECCS: {
    what: "The share of full biodiversity value that BECCS land provides.",
    detail: [
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
    ],
  },
  BIO_CONTRIBUTION_DESTOCKING: {
    what: "The share of full biodiversity value that destocked natural land provides.",
    detail: [
      "The biodiversity score of a cell is its biodiversity quality x this share x its area.",
      "1 = the full value of the cell in its natural state; 0 = no value.",
      "The cost of destocking grows with the habitat gained: this share less the value the land had under grazing.",
    ],
  },
  GBF2_TARGET: {
    what: "How much degraded land must be restored (Global Biodiversity Framework, Target 2).",
    detail: [
      "Target 2 asks that 30 % of degraded ecosystems are under restoration by 2030.",
      "Here it is a minimum biodiversity score over the priority degraded areas. The percentages of each level are in GBF2_TARGETS_DICT.",
    ],
    options: [
      ["off", "No restoration target."],
      ["low", "The low level."],
      ["medium", "The medium level."],
      ["high", "The high level."],
    ],
  },
  GBF2_TARGETS_DICT: {
    what: "The share of degradation to restore by 2030, 2050 and 2100, for each level of GBF2_TARGET.",
    detail: [
      "0.3 means 30 % of the gap between the current and the fully restored state of the priority degraded areas.",
      "The target rises in a straight line from zero in 2010.",
    ],
  },
  GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT: {
    what: "How much land counts as priority degraded area.",
    detail: [
      "Cells are ranked by biodiversity importance. A small share of top-ranked cells holds a large share of the benefit, so the ranking is cut where the benefit curve reaches this percentage.",
      "A higher cut keeps more cells as priority areas.",
    ],
  },
  GBF3_NVIS_TARGET: {
    what: "How much of each vegetation group must be conserved (Global Biodiversity Framework, Target 3).",
    detail: [
      "Target 3 asks that 30 % of land is conserved. Here each vegetation group, or bioregion, must reach a share of its original extent.",
    ],
    options: [
      ["off", "No vegetation targets."],
      ["medium", "Every group gets the medium level of GBF3_TARGETS_DICT."],
      ["high", "Every group gets the high level of GBF3_TARGETS_DICT."],
      ["SPECIFIED", "Each selected region gets its own level (GBF3_NVIS_SEL_REGION_TARGETS)."],
      ["CSV_DEFINED", "Each group gets the level set for it in the input table."],
    ],
  },
  GBF3_TARGETS_DICT: {
    what: "The percentage of its original extent each vegetation group must reach, for each level.",
    detail: [
      "Only read when the target setting is medium or high.",
      "The level moves in a straight line from its 2010 value to each of these years.",
    ],
  },
  GBF3_NVIS_TARGETS_OVERRIDE: {
    what: "A different target for individual vegetation groups in individual regions.",
    detail: [
      "Applied after the uniform level, whatever the target mode.",
      "Empty: no item is treated differently.",
    ],
  },
  GBF3_NVIS_TARGET_CLASS: {
    what: "How finely vegetation is classified.",
    detail: [
      "From the National Vegetation Information System (NVIS).",
    ],
    options: [
      ["NVIS_MVG", "Major vegetation groups: about 30."],
      ["NVIS_MVS", "Major vegetation subgroups: about 90."],
    ],
  },
  GBF3_NVIS_REGION_MODE: {
    what: "Whether vegetation targets are set for the country or region by region.",
    options: [
      ["AUSTRALIA", "One target per vegetation group for the whole country."],
      ["NRM", "One target per vegetation group in each selected NRM region."],
      ["IBRA_REG", "Targets per IBRA bioregion in place of vegetation groups."],
    ],
  },
  GBF3_NVIS_SEL_REGION_TARGETS: {
    what: "The regions that have vegetation targets, and their levels.",
    detail: [
      "The names of the regions decide where targets apply.",
      "Under the SPECIFIED mode the percentages give the level of each region; in the other modes only the names are used.",
    ],
  },
  GBF3_NVIS_MIN_AREA_HA: {
    what: "The smallest area of restorable land, in hectares, for a vegetation group to get a target in a region.",
    detail: [
      "With almost no land to act on, such a target could never be met and would make the year infeasible.",
    ],
  },
  GBF4_TARGET_SNES: {
    what: "How much habitat of each threatened species must be secured (Global Biodiversity Framework, Target 4).",
    detail: [
      "Covers the Species of National Environmental Significance listed under the EPBC Act.",
    ],
    options: [
      ["off", "No species targets."],
      ["medium", "Every species gets the medium level of GBF4_SNES_TARGETS_DICT."],
      ["high", "Every species gets the high level of GBF4_SNES_TARGETS_DICT."],
      ["SPECIFIED", "Species with a target in the input table; each selected region gets its own level."],
      ["CSV_DEFINED", "Species with a target in the input table, at the level set there."],
    ],
  },
  GBF4_TARGET_ECNES: {
    what: "How much of each threatened ecological community must be secured (Global Biodiversity Framework, Target 4).",
    detail: [
      "Covers the Ecological Communities of National Environmental Significance listed under the EPBC Act.",
    ],
    options: [
      ["off", "No community targets."],
      ["medium", "Every community gets the medium level of GBF4_ECNES_TARGETS_DICT."],
      ["high", "Every community gets the high level of GBF4_ECNES_TARGETS_DICT."],
      ["SPECIFIED", "Communities with a target in the input table; each selected region gets its own level."],
      ["CSV_DEFINED", "Communities with a target in the input table, at the level set there."],
    ],
  },
  GBF4_SNES_PRESENCE_CLASS: {
    what: "Which part of the mapped range of a species counts as its habitat.",
    options: [
      ["LIKELY", "Only where the species is likely to occur."],
      ["LIKELY_AND_MAYBE", "Where it is likely to occur or may occur."],
    ],
  },
  GBF4_ECNES_PRESENCE_CLASS: {
    what: "Which part of the mapped range of a community counts.",
    options: [
      ["LIKELY", "Only where the community is likely to occur."],
      ["LIKELY_AND_MAYBE", "Where it is likely to occur or may occur."],
    ],
  },
  GBF4_SNES_TARGETS_DICT: {
    what: "The percentage of its range each species must reach, for each level.",
    detail: [
      "Only read when the target setting is medium or high.",
      "The level moves in a straight line from its 2010 value to each of these years.",
    ],
  },
  GBF4_ECNES_TARGETS_DICT: {
    what: "The percentage of its range each community must reach, for each level.",
    detail: [
      "Only read when the target setting is medium or high.",
      "The level moves in a straight line from its 2010 value to each of these years.",
    ],
  },
  GBF4_SNES_TARGETS_OVERRIDE: {
    what: "A different target for individual species in individual regions.",
    detail: [
      "Applied after the uniform level, whatever the target mode.",
      "Empty: no item is treated differently.",
    ],
  },
  GBF4_ECNES_TARGETS_OVERRIDE: {
    what: "A different target for individual communities in individual regions.",
    detail: [
      "Applied after the uniform level, whatever the target mode.",
      "Empty: no item is treated differently.",
      "A target above what a community could ever reach is lowered to that level.",
    ],
  },
  GBF4_SNES_CAP_MARGIN: {
    what: "A safety margin, in percentage points, between a species target and the most the species could reach.",
    detail: [
      "A target is never set above what restoring all available land would achieve, less this margin.",
      "Without it, a target sitting exactly at the limit leaves the model no room.",
    ],
  },
  GBF4_SNES_REGION_MODE: {
    what: "Whether species targets are set for the country or region by region.",
    options: [
      ["AUSTRALIA", "One target per species for the whole country."],
      ["NRM", "One target per species in each selected NRM region."],
    ],
  },
  GBF4_SNES_SEL_REGION_TARGETS: {
    what: "The regions that have species targets, and their levels.",
    detail: [
      "The names of the regions decide where targets apply.",
      "Under the SPECIFIED mode the percentages give the level of each region; in the other modes only the names are used.",
    ],
  },
  GBF4_SNES_MIN_AREA_HA: {
    what: "The smallest area of habitat, in hectares, for a species to get a target in a region.",
    detail: [
      "With almost no land to act on, such a target could never be met and would make the year infeasible.",
    ],
  },
  GBF4_ECNES_REGION_MODE: {
    what: "Whether community targets are set for the country or region by region.",
    options: [
      ["AUSTRALIA", "One target per community for the whole country."],
      ["NRM", "One target per community in each selected NRM region."],
    ],
  },
  GBF4_ECNES_SEL_REGION_TARGETS: {
    what: "The regions that have community targets, and their levels.",
    detail: [
      "The names of the regions decide where targets apply.",
      "Under the SPECIFIED mode the percentages give the level of each region; in the other modes only the names are used.",
    ],
  },
  GBF4_ECNES_MIN_AREA_HA: {
    what: "The smallest area, in hectares, for a community to get a target in a region.",
    detail: [
      "With almost no land to act on, such a target could never be met and would make the year infeasible.",
    ],
  },
  GBF8_TARGET: {
    what: "How much climate-suitable habitat each species must keep (Global Biodiversity Framework, Target 8).",
    detail: [
      "Target 8 aims to reduce the impact of climate change on biodiversity. Each of about 10,600 species has a map of where the climate suits it, changing over the years.",
    ],
    options: [
      ["off", "No such targets."],
      ["medium", "Every species gets the medium level of GBF8_TARGETS_DICT."],
      ["high", "Every species gets the high level of GBF8_TARGETS_DICT."],
      ["CSV_DEFINED", "Only species with a level filled in by hand in the input table."],
    ],
  },
  GBF8_TARGETS_DICT: {
    what: "The percentage of its baseline suitable habitat each species must keep, for each level.",
    detail: [
      "Only read when the target setting is medium or high.",
      "The level moves in a straight line from its 2010 value to each of these years.",
    ],
  },
  EXCLUDE_NO_GO_LU: {
    what: "Whether some land uses are kept out of areas drawn by the user.",
    options: [
      ["True", "The land uses in NO_GO_VECTORS are not allowed inside their areas."],
      ["False", "No such areas."],
    ],
  },
  NO_GO_VECTORS: {
    what: "The land uses with no-go areas, and the map file of each area.",
    detail: [
      "Only used when EXCLUDE_NO_GO_LU is True.",
    ],
  },
  REGIONAL_ADOPTION_CONSTRAINTS: {
    what: "Whether there are limits on how much of a region a land use may take.",
    detail: [
      "These limits stand for how much land-use change a region is likely to accept.",
    ],
    options: [
      ["off", "No regional limits."],
      ["individual_cap", "A limit per region and land use, read from the regional adoption table of the inputs."],
      ["non_ag_sum_cap", "One limit on all non-agricultural land uses together in each region."],
    ],
  },
  REGIONAL_ADOPTION_ZONE: {
    what: "The kind of region used for the per-land-use limits.",
    detail: [
      "Only used in the individual_cap mode.",
    ],
    options: [
      ["ABARES_AAGIS", "ABARES farm survey regions."],
      ["LGA_CODE", "Local government areas."],
      ["NRM_CODE", "Natural resource management regions."],
      ["IBRA_ID", "IBRA bioregions."],
      ["SLA_5DIGIT", "Statistical local areas."],
    ],
  },
  REGIONAL_ADOPTION_NON_AG_REGION: {
    what: "The kind of region used for the limit on all non-agricultural land together.",
    detail: [
      "Only used in the non_ag_sum_cap mode.",
    ],
    options: [
      ["NRM", "Natural resource management regions."],
      ["State", "States."],
    ],
  },
  REGIONAL_ADOPTION_NON_AG_CAP: {
    what: "The largest share of a region, in percent, that all non-agricultural land uses together may take.",
    detail: [
      "Only used in the non_ag_sum_cap mode. 15 means at most 15 % of each region.",
    ],
  },
  REGIONAL_ADOPTION_NON_AG_CAP_REGIONS: {
    what: "The regions the combined non-agricultural limit applies to.",
    detail: [
      "Empty: every region is limited.",
      "Otherwise only the regions named here; all others are unlimited.",
    ],
  },
  REGIONAL_ADOPTION_NON_AG_CAP_OVERRIDE: {
    what: "A different combined limit for individual regions.",
    detail: [
      "A region named here uses its own percentage in place of REGIONAL_ADOPTION_NON_AG_CAP.",
    ],
  },
  RENEWABLES_OPTIONS: {
    what: "Which renewable energy technologies the model may install on farmland.",
    detail: [
      "A technology set to False is not available, and has no generation target, in this run.",
    ],
  },
  EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS: {
    what: "Whether renewable energy is kept out of the land most important for biodiversity.",
    options: [
      ["True", "No solar or wind on cells above the biodiversity cut-offs (RENEWABLE_GBF2_CUT_SOLAR / _WIND)."],
      ["False", "Solar and wind may be installed there."],
    ],
  },
  RENEWABLE_GBF2_CUT_WIND: {
    what: "How much high-biodiversity land is closed to wind farms.",
    detail: [
      "The same kind of cut-off as GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT, set separately. A higher value closes more land.",
    ],
  },
  RENEWABLE_GBF2_CUT_SOLAR: {
    what: "How much high-biodiversity land is closed to solar farms.",
    detail: [
      "The same kind of cut-off as GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT, set separately. A higher value closes more land.",
    ],
  },
  EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK: {
    what: "Whether renewable energy is kept out of the land most important for nationally protected species and communities.",
    options: [
      ["True", "No solar or wind on cells above the cut-offs (RENEWABLE_EPBC_MNES_CUT_SOLAR / _WIND)."],
      ["False", "Solar and wind may be installed there."],
    ],
  },
  RENEWABLE_EPBC_MNES_CUT_SOLAR: {
    what: "How much land important for protected species and communities is closed to solar farms.",
    detail: [
      "A higher value closes more land.",
    ],
  },
  RENEWABLE_EPBC_MNES_CUT_WIND: {
    what: "How much land important for protected species and communities is closed to wind farms.",
    detail: [
      "A higher value closes more land.",
    ],
  },
  RENEWABLE_TARGET_SCENARIO_TARGETS: {
    what: "The scenario that sets how much electricity solar and wind must generate.",
    detail: [
      "Targets are per state, technology and year.",
    ],
    options: [
      ["AEMO 2026 ISP - Accelerated Transition", "AEMO Integrated System Plan, fastest transition."],
      ["AEMO 2026 ISP - Step Change", "AEMO Integrated System Plan, central scenario."],
      ["AEMO 2026 ISP - Slower Growth", "AEMO Integrated System Plan, slowest transition."],
      ["Gladstone - Core", "Gladstone study, core scenario."],
      ["Gladstone - BESS Sensitivity", "Gladstone study, battery storage sensitivity."],
    ],
  },
  RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS: {
    what: "The scenario behind the maps of renewable energy cost and performance.",
    detail: [
      "The maps give installation cost, running cost, capacity factor and transmission loss for each cell.",
    ],
  },
  RE_TARGET_LEVEL: {
    what: "The kind of region renewable energy targets are set for.",
    detail: [
      "Only states are supported at present.",
    ],
    options: [
      ["STATE", "One target per state."],
      ["NRM", "Not yet supported."],
    ],
  },
  INSTALL_CAPACITY_MW_HA: {
    what: "The generating capacity installed per hectare, in megawatts, for each technology.",
  },
  RENEWABLES_ADOPTION_LIMITS: {
    what: "The largest share of suitable land each technology may take.",
    detail: [
      "1.0 means no limit; 0.5 means at most half.",
    ],
  },
  RESCALE_FACTOR: {
    what: "The size the numbers of each constraint are scaled to before solving.",
    detail: [
      "The inputs span many orders of magnitude (tonnes of water, tiny biodiversity scores). Each constraint is rescaled around this value so the solver stays numerically stable.",
      "It does not change the solution.",
    ],
  },
  SOLVER_COEFF_MIN: {
    what: "The smallest number kept in the optimisation model.",
    detail: [
      "Terms smaller than this are dropped, because very small numbers beside very large ones make the solver unreliable.",
      "Chosen by testing: small enough that dropping them does not noticeably change results.",
    ],
  },
  BOUND_PROP_REL_TOL: {
    what: "The tolerance of the feasibility check that runs before each solve.",
    detail: [
      "Before solving, every constraint is checked against the most and least it could possibly reach. A constraint is only called impossible when it misses by more than this margin.",
    ],
  },
  BOUND_PROP_DROP_FAMILIES: {
    what: "Kinds of constraints that are removed before solving when they are certain to hold.",
    detail: [
      "A constraint that every possible solution satisfies can be left out without changing the result, which makes the model smaller.",
      "Empty: nothing is removed.",
    ],
  },
  BOUND_PROP_ON_IMPOSSIBLE: {
    what: "What happens when the pre-solve check finds a constraint that cannot be met.",
    options: [
      ["stop", "The year stops before solving, and the report names the constraint."],
      ["solve", "The problem is reported and the solve goes ahead anyway."],
    ],
  },
  ELASTIC_FAMILIES: {
    what: "The kinds of targets that are allowed to fall short.",
    detail: [
      "Empty: every target must be met, and a year with a target that cannot be met fails.",
      "A target of a kind listed here may be missed at a very high penalty (ELASTIC_PENALTY), so the run continues and the shortfall is reported.",
      "Kinds that can be listed: GBF2, GBF3_NVIS, GBF4_SNES, GBF4_ECNES, GBF8, ghg, water, demand, renewable, and the adoption limits.",
    ],
  },
  ELASTIC_DROP_SHORT: {
    what: "Whether targets that cannot be met are dropped and the year solved again.",
    options: [
      ["True", "A first solve finds the targets that fall short. They are removed, all others are enforced, and a second solve gives the stored result."],
      ["False", "One solve: targets that fall short stay in, at the price of the penalty."],
    ],
  },
  ELASTIC_DROP_TOL: {
    what: "How far a target must fall short, as a fraction of the target, to be dropped.",
    detail: [
      "Smaller shortfalls are numerical noise of the first solve, not real misses.",
      "Only used when ELASTIC_DROP_SHORT is True.",
    ],
  },
  ELASTIC_PASS1_PARAMS: {
    what: "The solver options of the first of the two solves.",
    detail: [
      "The first solve only has to find which targets fall short, so it uses a faster, less exact method.",
      "Only used when ELASTIC_DROP_SHORT is True.",
    ],
  },
  ELASTIC_PASS1_COST_SCALE: {
    what: "How much economic cost matters in the first of the two solves.",
    detail: [
      "1: cost and shortfall are traded off, so a target may be left short because it is expensive.",
      "A very small value: shortfall is minimised first, so only targets that truly cannot be met are left short.",
      "Only used when ELASTIC_DROP_SHORT is True.",
    ],
  },
  ELASTIC_PENALTY: {
    what: "The penalty for missing a target that is allowed to fall short, in AUD for a target missed in full.",
    detail: [
      "Set very high so a target is missed only when it cannot be met, not merely when it is costly.",
    ],
  },
  THREADS: {
    what: "The number of processor threads the solver uses.",
  },
  FEASIBILITY_TOLERANCE: {
    what: "How exactly the solution must satisfy each constraint.",
    detail: [
      "A smaller value is more exact and slower. It also sets the precision that numbers are rounded to.",
    ],
  },
  OPTIMALITY_TOLERANCE: {
    what: "How close to the best possible objective the solution must be.",
    detail: [
      "A smaller value is more exact and slower.",
    ],
  },
  BARRIER_CONVERGENCE_TOLERANCE: {
    what: "When the solver stops improving the solution.",
    detail: [
      "The solver stops when the gap between its current solution and the best possible value falls below this relative size.",
      "A smaller value is more exact and slower.",
    ],
  },
  SCALE_FLAG: {
    what: "Whether the solver rescales the model itself.",
    detail: [
      "Off here, because the model is already scaled before it reaches the solver, and testing showed the other options were slower or unstable.",
    ],
    options: [
      ["-1", "Automatic."],
      ["0", "No scaling by the solver."],
      ["1", "Equilibrium scaling."],
      ["2", "Geometric scaling."],
      ["3", "Multi-pass equilibrium scaling."],
    ],
  },
  RETRY_PARAMS: {
    what: "The solver options tried, in order, for each year.",
    detail: [
      "If the first set fails, the next is tried. Each set fixes the numerical care, the algorithm (2 = barrier), crossover, presolve and the homogeneous barrier option.",
    ],
  },
  BIO_ROW_GENERATION: {
    what: "Whether biodiversity targets are added to the model only as they are needed.",
    options: [
      ["True", "Targets already certain to be met are left out; the others are added in rounds as they are found to fall short. Needed when thousands of species have targets, to keep the model small enough to solve."],
      ["False", "Every target is built into the model from the start."],
    ],
  },
  ROWGEN_EAGER_MAX_ENTRIES: {
    what: "The size below which all targets of a kind are added at once.",
    detail: [
      "Small families of targets are added whole in the first round; only very large ones are added gradually.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_BATCH_MIN: {
    what: "The fewest targets added per round.",
    detail: [
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_BATCH_SHARE: {
    what: "The share of the targets found short that is added per round.",
    detail: [
      "The hardest are added first, since fixing them usually lifts the rest.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_BATCH_MAX: {
    what: "The most targets added per round.",
    detail: [
      "Each round costs one full solve.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_MAX_ROUNDS: {
    what: "The most rounds allowed in one solve.",
    detail: [
      "A solve that reaches it stops without having converged, and the log says so.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_SHORT_TOL: {
    what: "How far below its target a score must be to count as short.",
    detail: [
      "A relative tolerance.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROWGEN_SCREEN_TOL: {
    what: "The margin by which a target must clear the pre-check to be called certain.",
    detail: [
      "A relative tolerance.",
      "Only used when BIO_ROW_GENERATION is True.",
    ],
  },
  ROUND_DECIMALS: {
    what: "The number of decimal places values are rounded to.",
    detail: [
      "It removes meaningless digits; it follows the feasibility tolerance.",
    ],
  },
  WRITE_OUTPUTS: {
    what: "Whether results are written at the end of the run.",
    detail: [
      "Turned off for quick tests and debugging.",
    ],
  },
  WRITE_DEBUG_MPS: {
    what: "Whether each year's optimisation model is saved to a file before it is solved.",
    detail: [
      "Used to investigate a year that never finishes. The files are large.",
    ],
  },
  WRITE_REPORT_MAX_MEM_MB: {
    what: "The most memory, in megabytes, that writing the report may use.",
  },
  WRITE_CHUNK_SIZE: {
    what: "The size of the pieces in which map data is written to file.",
  },
  WRITE_GBF3_NVIS: {
    what: "Which vegetation group scores and maps are written to the outputs.",
    options: [
      ["off", "Not written."],
      ["all", "Every item of the input table, whether or not it had a target."],
      ["selected", "Only the items the model actually held to a target."],
    ],
  },
  WRITE_GBF4_SNES: {
    what: "Which threatened species scores and maps are written to the outputs.",
    detail: [
      "Writing all of about 2,000 species takes several hours.",
    ],
    options: [
      ["off", "Not written."],
      ["all", "Every item of the input table, whether or not it had a target."],
      ["selected", "Only the items the model actually held to a target."],
    ],
  },
  WRITE_GBF4_ECNES: {
    what: "Which ecological community scores and maps are written to the outputs.",
    options: [
      ["off", "Not written."],
      ["all", "Every item of the input table, whether or not it had a target."],
      ["selected", "Only the items the model actually held to a target."],
    ],
  },
  VERBOSE: {
    what: "Whether detailed progress is printed while the model runs.",
    options: [
      ["1", "Yes."],
      ["0", "No."],
    ],
  },
};
