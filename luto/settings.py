# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

""" LUTO model settings. """

import os
import pandas as pd

# ---------------------------------------------------------------------------- #
# LUTO model version.                                                                 #
# ---------------------------------------------------------------------------- #
VERSION = '2.3'

############### Set some Spyder options
pd.set_option('display.width', 470)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)

# ---------------------------------------------------------------------------- #
# Directories.                                                                 #
# ---------------------------------------------------------------------------- #

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'input'
RAW_DATA = '../raw_data'


# ---------------------------------------------------------------------------- #
# Scenario parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Climate change assumptions. Options include '126', '245', '370', '585'
SSP = '245'
RCP = 'rcp' + SSP[1] + 'p' + SSP[2]  # Representative Concentration Pathway string identifier e.g., 'rcp4p5'.

# Set demand parameters which define requirements for Australian production of agricultural commodities
SCENARIO = SSP_NUM = 'SSP' + SSP[0] # SSP1, SSP2, SSP3, SSP4, SSP5
DIET_DOM = 'BAU'                    # 'BAU', 'FLX', 'VEG', 'VGN' - domestic diets in Australia
DIET_GLOB = 'BAU'                   # 'BAU', 'FLX', 'VEG', 'VGN' - global diets
CONVERGENCE = 2050                  # 2050 or 2100 - date at which dietary transformation is completed (velocity of transformation)
IMPORT_TREND = 'Static'             # 'Static' (assumes 2010 shares of imports for each commodity) or 'Trend' (follows historical rate of change in shares of imports for each commodity)
WASTE = 1                           # 1 for full waste, 0.5 for half waste 
FEED_EFFICIENCY = 'BAU'             # 'BAU' or 'High'

# Add CO2 fertilisation effects on agricultural production from GAEZ v4 
CO2_FERT = 'on'   # or 'off'

# Fire impacts on carbon sequestration
RISK_OF_REVERSAL = 0.05  # Risk of reversal buffer under ERF (reasonable values range from 0.05 [100 years] to 0.25 [25 years]) https://www.cleanenergyregulator.gov.au/ERF/Choosing-a-project-type/Opportunities-for-the-land-sector/Risk-of-reversal-buffer
FIRE_RISK = 'med'   # Options are 'low', 'med', 'high'. Determines whether to take the 5th, 50th, or 95th percentile of modelled fire impacts.
""" 
    Mean FIRE_RISK cell values (%)...
    FD_RISK_PERC_5TH    80.3967
    FD_RISK_MEDIAN      89.2485
    FD_RISK_PERC_95TH   93.2735
"""


# ---------------------------------------------------------------------------- #
# Economic parameters
# ---------------------------------------------------------------------------- #

# Amortise upfront (i.e., establishment and transitions) costs 
AMORTISE_UPFRONT_COSTS = False

# Discount rate for amortisation
DISCOUNT_RATE = 0.05     # 0.05 = 5% pa.

# Set amortisation period
AMORTISATION_PERIOD = 30 # years



# ---------------------------------------------------------------------------- #
# Model parameters
# ---------------------------------------------------------------------------- #

# Optionally coarse-grain spatial domain (faster runs useful for testing)
RESFACTOR = 1          # set to 1 to run at full spatial resolution, > 1 to run at reduced resolution. E.g. RESFACTOR 5 selects every 5 x 5 cell

# How does the model run over time 
MODE = 'snapshot'   # 'snapshot' runs for target year only, 'timeseries' runs each year from base year to target year

# If MODE == 'timeseries', these two settings determine whether the model is allowed to remove features previously added. For example 
# if a cell has Riparian planing added if NON_AG_REVERSIBLE = False then Riparian planting will remain on that solve.
AG_MANAGEMENT_REVERSIBLE = True
NON_AG_REVERSIBLE = True

# Define the objective function
# OBJECTIVE = 'maxrev' # maximise revenue (price x quantity - costs)                 **** Must use DEMAND_CONSTRAINT_TYPE = 'soft' ****
OBJECTIVE = 'mincost'  # minimise cost (transitions costs + annual production costs) **** Use either DEMAND_CONSTRAINT_TYPE = 'soft' or 'hard' ****

# Specify how demand should be met in the solver
DEMAND_CONSTRAINT_TYPE = 'hard'  # Adds demand as a constraint in the solver (linear programming approach)
# DEMAND_CONSTRAINT_TYPE = 'soft'  # Adds demand as a type of slack variable in the solver (goal programming approach)

# Penalty in objective function to balance influence of demand versus cost when DEMAND_CONSTRAINT_TYPE = 'soft'
# 1e5 works well (i.e., demand are met), demands not met with anything less 
PENALTY = 1e5

# Write GeoTiffs to output directory: True or False
WRITE_OUTPUT_GEOTIFFS = False


# ---------------------------------------------------------------------------- #
# Gurobi parameters
# ---------------------------------------------------------------------------- #

# Select Gurobi algorithm used to solve continuous models or the initial root relaxation of a MIP model. Default is automatic. 
SOLVE_METHOD = 2  # 'automatic: -1, primal simplex: 0, dual simplex: 1, barrier: 2, concurrent: 3, deterministic concurrent: 4, deterministic concurrent simplex: 5

# Presolve parameters (switching both to 0 solves numerical problems)
PRESOLVE = 0     # automatic (-1), off (0), conservative (1), or aggressive (2)
AGGREGATE = 0    # Controls the aggregation level in presolve. The options are off (0), moderate (1), or aggressive (2). In rare instances, aggregation can lead to an accumulation of numerical errors. Turning it off can sometimes improve solution accuracy (it did not fix sub-optimal termination issue)

# Print detailed output to screen
VERBOSE = 1

# Relax the tolerances for feasibility and optimality
FEASIBILITY_TOLERANCE = 1e-2              # Primal feasility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2
OPTIMALITY_TOLERANCE = 1e-2               # Dual feasility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2
BARRIER_CONVERGENCE_TOLERANCE = 1e-5      # Range from 1e-2 to 1e-8 (default), that larger the number the faster but the less exact the solve. 1e-5 is a good compromise between optimality and speed.

# Whether to use crossover in barrier solve. 0 = off, -1 = automatic. Auto cleans up sub-optimal termination errors without much additional compute time (apart from 2050 when it sometimes never finishes).
CROSSOVER = 0

# Parameters for dealing with numerical issues. NUMERIC_FOCUS = 2 fixes most things but roughly doubles solve time.
SCALE_FLAG = -1     # Scales the rows and columns of the model to improve the numerical properties of the constraint matrix. -1: Auto, 0: No scaling, 1: equilibrium scaling (First scale each row to make its largest nonzero entry to be magnitude one, then scale each column to max-norm 1), 2: geometric scaling, 3: multi-pass equilibrium scaling. Testing revealed that 1 tripled solve time, 3 led to numerical problems.
NUMERIC_FOCUS = 0   # Controls the degree to which the code attempts to detect and manage numerical issues. Default (0) makes an automatic choice, with a slight preference for speed. Settings 1-3 increasingly shift the focus towards being more careful in numerical computations. NUMERIC_FOCUS = 1 is ok, but 2 increases solve time by ~4x
BARHOMOGENOUS = -1  # Useful for recognizing infeasibility or unboundedness. At the default setting (-1), it is only used when barrier solves a node relaxation for a MIP model. 0 = off, 1 = on. It is a bit slower than the default algorithm (3x slower in testing).

# Number of threads to use in parallel algorithms (e.g., barrier)
THREADS = 32


# ---------------------------------------------------------------------------- #
# Non-agricultural land usage parameters
# ---------------------------------------------------------------------------- #
NON_AGRICULTURAL_LU_BASE_CODE = 100         # Non-agricultural land uses will appear on the land use map
                                            # offset by this amount (e.g. land use 0 will appear as 100)

# Environmental Plantings Parameters
ENV_PLANTING_COST_PER_HA_PER_YEAR = 100     # Yearly cost of maintaining one hectare of environmental plantings
CARBON_PRICE_PER_TONNE = 100                # Price of carbon per tonne - determines EP revenue in the model

# Riparian Planting Parameters
RIPARIAN_PLANTING_COST_PER_HA_PER_YEAR = 100
RIPARIAN_PLANTINGS_BUFFER_WIDTH = 20
# RIPARIAN_PLANTINGS_FENCING_COST_PER_HA = 10 * 100  # $10 per metre, 100 metres per hectare
RIPARIAN_PLANTINGS_FENCING_COST_PER_M = 2           # $ per linear metre
RIPARIAN_PLANTINGS_TORTUOSITY_FACTOR = 0.5

# Agroforestry Parameters
AGROFORESTRY_COST_PER_HA_PER_YEAR = 100
AGROFORESTRY_ROW_WIDTH = 20
AGROFORESTRY_ROW_SPACING = 40
# AGROFORESTRY_FENCING_COST_PER_HA = 10 * 100  # $10 per metre, 100 metres per hectare
AGROFORESTRY_FENCING_COST_PER_M = 2           # $ per linear metre
AF_PROPORTION = AGROFORESTRY_ROW_WIDTH / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
no_alleys_per_ha = 100 / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
AF_FENCING_LENGTH = 100 * no_alleys_per_ha * 2 # Length of fencing required per ha in metres
                    

# ---------------------------------------------------------------------------- #
# Agricultural management parameters
# ---------------------------------------------------------------------------- #

AGRICULTURAL_MANAGEMENT_USE_THRESHOLD = 0.1  # The minimum value an agricultural management variable must take for the
                                             # write_output function to consider it being used on a cell


# ---------------------------------------------------------------------------- #
# Environmental parameters
# ---------------------------------------------------------------------------- #

# Greenhouse gas emissions limits and parameters *******************************
GHG_EMISSIONS_LIMITS = 'on'        # 'on' or 'off'

GHG_LIMITS_TYPE = 'file' # 'dict' or 'file'

# Set emissions limits in dictionary below (i.e., year: tonnes)
GHG_LIMITS = {                     
              2010: 90 * 1e6,    # Agricultural emissions in 2010 in tonnes CO2e
              2050: -337 * 1e6,  # GHG emissions target and year (can add more years/targets)
              2100: -337 * 1e6   # GHG emissions target and year (can add more years/targets)
             }

# Take data from 'GHG_targets.xlsx', options include: 'None', '1.5C (67%)', '1.5C (50%)', or '1.8C (67%)'
GHG_LIMITS_FIELD = '1.5C (67%)'    

SOC_AMORTISATION = 30           # Number of years over which to spread (average) soil carbon accumulation



# Water use limits and parameters *******************************

WATER_USE_LIMITS = 'on'               # 'on' or 'off'
WATER_LIMITS_TYPE = 'water_stress'    # 'water_stress' or 'pct_ag'

# If WATER_LIMITS_TYPE = 'pct_ag'...       
# Set reduction in water use as percentage of 2010 irrigation water use
WATER_USE_REDUCTION_PERCENTAGE = 0  

# If WATER_LIMITS_TYPE = 'water_stress'...                                           
# (0.25 follows Aqueduct classification of 0.4 but leaving 0.15 for urban/industrial/indigenous use).
# Safe and just Earth system boundaries says 0.2 inclusive of domestic/industrial https://www.nature.com/articles/s41586-023-06083-8  
WATER_STRESS_FRACTION = 0.2          

# Regionalisation to enforce water use limits by
WATER_REGION_DEF = 'DD'                 # 'RR' for River Region, 'DD' for Drainage Division



# Biodiversity limits and parameters *******************************

# Set the weighting of landscape connectivity on biodiversity value (0 (no influence) - 1 (full influence))
CONNECTIVITY_WEIGHTING = 1

# Set livestock impact on biodiversity (0 = no impact, 1 = total annihilation)
BIODIV_LIVESTOCK_IMPACT = 0.5

# Set benefit level of EP, AF, and RP (0 = none, 1 = full)
REFORESTATION_BIODIVERSITY_BENEFIT = 0.7

# Set biodiversity target (0 - 1 e.g., 0.3 = 30% of total achievable Zonation biodiversity benefit)
BIODIVERSITY_LIMITS = 'on'             # 'on' or 'off'
BIODIV_TARGET = 0.3
BIODIV_TARGET_ACHIEVEMENT_YEAR = 2030



# ---------------------------------------------------------------------------- #
# Cell Culling
# ---------------------------------------------------------------------------- #

CULL_MODE = 'absolute'      # cull to include at most MAX_LAND_USES_PER_CELL
# CULL_MODE = 'percentage'    # cull the LAND_USAGE_THRESHOLD_PERCENTAGE % most expensive options
# CULL_MODE = 'none'          # do no culling

MAX_LAND_USES_PER_CELL = 12 
LAND_USAGE_CULL_PERCENTAGE = 0.15




""" NON-AGRICULTURAL LAND USES (indexed by k)
0: 'Environmental Plantings'
1: 'Riparian Plantings'
2: 'Agroforestry'


AGRICULTURAL MANAGEMENT OPTIONS (indexed by a)
0: (None)
1: 'Asparagopsis taxiformis'
2: 'Precision Agriculture'
3: 'Ecological Grazing'


DRAINAGE DIVISIONS
 1: 'Tanami-Timor Sea Coast',
 2: 'South Western Plateau',
 3: 'South West Coast',
 4: 'Tasmania',
 5: 'South East Coast (Victoria)',
 6: 'South Australian Gulf',
 7: 'Murray-Darling Basin',
 8: 'Pilbara-Gascoyne',
 9: 'North Western Plateau',
 10: 'South East Coast (NSW)',
 11: 'Carpentaria Coast',
 12: 'Lake Eyre Basin',
 13: 'North East Coast'


RIVER REGIONS
 1: 'ADELAIDE RIVER',
 2: 'ALBANY COAST',
 3: 'ARCHER-WATSON RIVERS',
 4: 'ARTHUR RIVER',
 5: 'ASHBURTON RIVER',
 6: 'AVOCA RIVER',
 7: 'AVON RIVER-TYRELL LAKE',
 8: 'BAFFLE CREEK',
 9: 'BARRON RIVER',
 10: 'BARWON RIVER-LAKE CORANGAMITE',
 11: 'BATHURST-MELVILLE ISLANDS',
 12: 'BEGA RIVER',
 13: 'BELLINGER RIVER',
 14: 'BENANEE-WILLANDRA CREEK',
 15: 'BILLABONG-YANCO CREEKS',
 16: 'BLACK RIVER',
 17: 'BLACKWOOD RIVER',
 18: 'BLYTH RIVER',
 19: 'BORDER RIVERS',
 20: 'BOYNE RIVER',
 21: 'BRISBANE RIVER',
 22: 'BROKEN RIVER',
 23: 'BROUGHTON RIVER',
 24: 'BRUNSWICK RIVER',
 25: 'BUCKINGHAM RIVER',
 26: 'BULLO RIVER-LAKE BANCANNIA',
 27: 'BUNYIP RIVER',
 28: 'BURDEKIN RIVER',
 29: 'BURNETT RIVER',
 30: 'BURRUM RIVER',
 31: 'BUSSELTON COAST',
 32: 'CALLIOPE RIVER',
 33: 'CALVERT RIVER',
 34: 'CAMPASPE RIVER',
 35: 'CAPE LEVEQUE COAST',
 36: 'CARDWELL COAST',
 37: 'CASTLEREAGH RIVER',
 38: 'CLARENCE RIVER',
 39: 'CLYDE RIVER-JERVIS BAY',
 40: 'COAL RIVER',
 41: 'COLLIE-PRESTON RIVERS',
 42: 'CONDAMINE-CULGOA RIVERS',
 43: 'COOPER CREEK',
 44: 'CURTIS ISLAND',
 45: 'DAINTREE RIVER',
 46: 'DALY RIVER',
 47: 'DARLING RIVER',
 48: 'DE GREY RIVER',
 49: 'DENMARK RIVER',
 50: 'DERWENT RIVER',
 51: 'DIAMANTINA-GEORGINA RIVERS',
 52: 'DON RIVER',
 53: 'DONNELLY RIVER',
 54: 'DRYSDALE RIVER',
 55: 'DUCIE RIVER',
 56: 'EAST ALLIGATOR RIVER',
 57: 'EAST COAST',
 58: 'EAST GIPPSLAND',
 59: 'EMBLEY RIVER',
 60: 'ENDEAVOUR RIVER',
 61: 'ESPERANCE COAST',
 62: 'EYRE PENINSULA',
 63: 'FINNISS RIVER',
 64: 'FITZMAURICE RIVER',
 65: 'FITZROY RIVER (QLD)',
 66: 'FITZROY RIVER (WA)',
 67: 'FLEURIEU PENINSULA',
 68: 'FLINDERS-CAPE BARREN ISLANDS',
 69: 'FLINDERS-NORMAN RIVERS',
 70: 'FORTESCUE RIVER',
 71: 'FORTH RIVER',
 72: 'FRANKLAND-DEEP RIVERS',
 73: 'FRASER ISLAND',
 74: 'GAIRDNER',
 75: 'GASCOYNE RIVER',
 76: 'GAWLER RIVER',
 77: 'GLENELG RIVER',
 78: 'GOOMADEER RIVER',
 79: 'GORDON RIVER',
 80: 'GOULBURN RIVER',
 81: 'GOYDER RIVER',
 82: 'GREENOUGH RIVER',
 83: 'GROOTE EYLANDT',
 84: 'GWYDIR RIVER',
 85: 'HASTINGS RIVER',
 86: 'HAUGHTON RIVER',
 87: 'HAWKESBURY RIVER',
 88: 'HERBERT RIVER',
 89: 'HINCHINBROOK ISLAND',
 90: 'HOLROYD RIVER',
 91: 'HOPKINS RIVER',
 92: 'HUNTER RIVER',
 93: 'HUON RIVER',
 94: 'ISDELL RIVER',
 95: 'JARDINE RIVER',
 96: 'JEANNIE RIVER',
 97: 'JOHNSTONE RIVER',
 98: 'KANGAROO ISLAND',
 99: 'KARUAH RIVER',
 100: 'KEEP RIVER',
 101: 'KENT RIVER',
 102: 'KIEWA RIVER',
 103: 'KING EDWARD RIVER',
 104: 'KING ISAND',
 105: 'KING-HENTY RIVERS',
 106: 'KINGSTON COAST',
 107: 'KOLAN RIVER',
 108: 'KOOLATONG RIVER',
 109: 'LACHLAN RIVER',
 110: 'LAKE EYRE',
 111: 'LAKE TORRENS-MAMBRAY COAST',
 112: 'LENNARD RIVER',
 113: 'LIMMEN BIGHT RIVER',
 114: 'LITTLE RIVER',
 115: 'LIVERPOOL RIVER',
 116: 'LOCKHART RIVER',
 117: 'LODDON RIVER',
 118: 'LOGAN-ALBERT RIVERS',
 119: 'LOWER MALLEE',
 120: 'LOWER MURRAY RIVER',
 121: 'MACLEAY RIVER',
 122: 'MACQUARIE-BOGAN RIVERS',
 123: 'MACQUARIE-TUGGERAH LAKES',
 124: 'MANNING RIVER',
 125: 'MAROOCHY RIVER',
 126: 'MARY RIVER (NT)',
 127: 'MARY RIVER (QLD)',
 128: 'MERSEY RIVER',
 129: 'MILLICENT COAST',
 130: 'MITCHELL-COLEMAN RIVERS (QLD)',
 131: 'MITCHELL-THOMSON RIVERS',
 132: 'MOONIE RIVER',
 133: 'MOORE-HILL RIVERS',
 134: 'MORNING INLET',
 135: 'MORNINGTON ISLAND',
 136: 'MORUYA RIVER',
 137: 'MOSSMAN RIVER',
 138: 'MOYLE RIVER',
 139: 'MULGRAVE-RUSSELL RIVERS',
 140: 'MURCHISON RIVER',
 141: 'MURRAY RIVER (WA)',
 142: 'MURRAY RIVERINA',
 143: 'MURRUMBIDGEE RIVER',
 144: 'MYPONGA RIVER',
 145: 'McARTHUR RIVER',
 146: 'NAMOI RIVER',
 147: 'NICHOLSON-LEICHHARDT RIVERS',
 148: 'NOOSA RIVER',
 149: 'NORMANBY RIVER',
 150: 'NULLARBOR',
 151: "O'CONNELL RIVER",
 152: 'OLIVE-PASCOE RIVERS',
 153: 'ONKAPARINGA RIVER',
 154: 'ONSLOW COAST',
 155: 'ORD-PENTECOST RIVERS',
 156: 'OTWAY COAST',
 157: 'OVENS RIVER',
 158: 'PAROO RIVER',
 159: 'PIEMAN RIVER',
 160: 'PINE RIVER',
 161: 'PIONEER RIVER',
 162: 'PIPER-RINGAROOMA RIVERS',
 163: 'PLANE CREEK',
 164: 'PORT HEDLAND COAST',
 165: 'PORTLAND COAST',
 166: 'PRINCE REGENT RIVER',
 167: 'PROSERPINE RIVER',
 168: 'RICHMOND RIVER',
 169: 'ROBINSON RIVER',
 170: 'ROPER RIVER',
 171: 'ROSIE RIVER',
 172: 'ROSS RIVER',
 173: 'RUBICON RIVER',
 174: 'SALT LAKE',
 175: 'SANDY CAPE COAST',
 176: 'SANDY DESERT',
 177: 'SETTLEMENT CREEK',
 178: 'SHANNON RIVER',
 179: 'SHOALHAVEN RIVER',
 180: 'SHOALWATER CREEK',
 181: 'SMITHTON-BURNIE COAST',
 182: 'SNOWY RIVER',
 183: 'SOUTH ALLIGATOR RIVER',
 184: 'SOUTH COAST',
 185: 'SOUTH GIPPSLAND',
 186: 'SOUTH-WEST COAST',
 187: 'SPENCER GULF',
 188: 'STEWART RIVER',
 189: 'STRADBROKE ISLAND',
 190: 'STYX RIVER',
 191: 'SWAN COAST-AVON RIVER',
 192: 'SYDNEY COAST-GEORGES RIVER',
 193: 'TAMAR RIVER',
 194: 'TORRENS RIVER',
 195: 'TORRES STRAIT ISLANDS',
 196: 'TOWAMBA RIVER',
 197: 'TOWNS RIVER',
 198: 'TULLY-MURRAY RIVERS',
 199: 'TUROSS RIVER',
 200: 'TWEED RIVER',
 201: 'UPPER MALLEE',
 202: 'UPPER MURRAY RIVER',
 203: 'VICTORIA RIVER-WISO',
 204: 'WAKEFIELD RIVER',
 205: 'WALKER RIVER',
 206: 'WARD RIVER',
 207: 'WARREGO RIVER',
 208: 'WARREN RIVER',
 209: 'WATER PARK CREEK',
 210: 'WENLOCK RIVER',
 211: 'WERRIBEE RIVER',
 212: 'WHITSUNDAY ISLANDS',
 213: 'WILDMAN RIVER',
 214: 'WIMMERA RIVER',
 215: 'WOLLONGONG COAST',
 216: 'WOORAMEL RIVER',
 217: 'YANNARIE RIVER',
 218: 'YARRA RIVER'}
"""