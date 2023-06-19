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

"""
LUTO 2.0 settings.
"""


import os
import pandas as pd


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
# Parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Climate change assumptions. Options include '126', '245', '370', '585'
SSP = '245' # 'rcp4p5' # Representative Concentration Pathway string identifier.
RCP = 'rcp' + SSP[1] + 'p' + SSP[2]

# Economic parameters
DISCOUNT_RATE = 0.05     # 0.05 = 5% pa.
AMORTISATION_PERIOD = 30 # years

# Optionally coarse-grain spatial domain (faster runs useful for testing)
RESFACTOR = 1             # set to 1 to run at full spatial resolution
# SAMPLING = 'linear'     # Converts non-nodata cells to 1D array and selects every n'th cell for modelling
SAMPLING = 'quadratic'    # Selects cell from every n x n block from 2D array (raster map) for modelling (better as more regularly spaced)

# How does the model run over time 
# STYLE = 'snapshot'       # runs for target year only
STYLE = 'timeseries'   # runs each year from base year to target year

# Penalty in objective function
PENALTY = 1000000

# Column generation options
USE_COLUMN_GENERATION = True  # enable or disable using column generation for the solve
CLUSTER_SIZE = 1000           # initial max cluster size for the column generation solve

# Set Gurobi parameters

# Select Gurobi algorithm used to solve continuous models or the initial root relaxation of a MIP model.
# Set solve method. Default is automatic. Dual simplex uses less memory.
SOLVE_METHOD = 1

""" SOLVE METHODS
 'automatic':                       -1
 'primal simplex':                   0
 'dual simplex':                     1
 'barrier':                          2
 'concurrent':                       3
 'deterministic concurrent':         4
 'deterministic concurrent simplex': 5
"""

# Print detailed output to screen
VERBOSE = 0

# Environmental constraint settings. In general there is some sort of 'cap'
# ('hard', 'soft' or 'none') and an optional requirement to further minimise.

# Water:
WATER_CONSTRAINT_TYPE = 'hard'          # or 'soft' or None.
WATER_CONSTRAINT_MINIMISE = False       # or True. Whether to also minimise water use.
WATER_CONSTRAINT_WEIGHT = 1.0           # Minimisation weight in objective function.
WATER_YIELD_STRESS_FRACTION = 0.4       # Water stress if yields below this fraction.

WATER_REGION_DEF = 'RR'                 # 'RR' for River Region, 'DD' for Drainage Division
WATER_DRAINDIVS = list(range(1, 14, 1)) # List of drainage divisions to apply irrigation constraints to e.g., [1, 2].
WATER_RIVREGS = list(range(1, 219, 1))  # List of river regions to apply irrigation constraints to e.g., [1, 2].


""" DRAINAGE DIVISIONS
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
