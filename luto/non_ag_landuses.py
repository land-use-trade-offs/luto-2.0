"""
Contains a master dictionary that stores all of the non agricultural land uses
and whether they are currently enabled in the solver (True/False). 

To disable a non-agricultural land use, change the correpsonding value of the
NON_AG_LAND_USES dictionary to false.
"""

NON_AG_LAND_USES = {
    'Environmental Plantings': True,
    'Riparian Plantings': True,
    'Agroforestry': True,
    'Carbon Plantings (Block)': True,
    'Carbon Plantings (Belt)': True,
    'BECCS': True,
}