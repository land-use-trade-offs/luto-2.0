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

"""
If settings.MODE == 'timeseries', the values of the below dictionary determine whether the model is allowed to abandon non-agr.
land uses on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Environmental Plantings'
and the corresponding value in this dictionary is False, all cells using EP must also utilise this land use in all subsequent 
years.
"""

NON_AG_LAND_USES_REVERSIBLE = {
    'Environmental Plantings': False,
    'Riparian Plantings': False,
    'Agroforestry': False,
    'Carbon Plantings (Block)': False,
    'Carbon Plantings (Belt)': False,
    'BECCS': False,
}
