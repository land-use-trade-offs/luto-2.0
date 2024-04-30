"""
Contains a master list of all agricultural management options and
which land uses they correspond to. 

To disable an ag-mangement option, change the corresponding value in the
AG_MANAGEMENTS dictionary to False.
"""

AG_MANAGEMENTS = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
}

AG_MANAGEMENTS_TO_LAND_USES = {
    'Asparagopsis taxiformis': [
        'Beef - natural land',
        'Beef - modified land',
        'Sheep - natural land',
        'Sheep - modified land',
        'Dairy - natural land',
        'Dairy - modified land',
    ],
    'Precision Agriculture': [
        # Cropping:
        'Hay',
        'Summer cereals',
        'Summer legumes',
        'Summer oilseeds',
        'Winter cereals',
        'Winter legumes',
        'Winter oilseeds',
        # Intensive Cropping:
        'Cotton',
        'Other non-cereal crops',
        'Rice',
        'Sugar',
        'Vegetables',
        # Horticulture:
        'Apples',
        'Citrus',
        'Grapes',
        'Nuts',
        'Pears',
        'Plantation fruit',
        'Stone fruit',
        'Tropical stone fruit',
    ],
    'Ecological Grazing': [
        'Beef - modified land',
        'Sheep - modified land',
        'Dairy - modified land',
    ],
    'Savanna Burning': [
        'Beef - natural land',
        'Dairy - natural land',
        'Sheep - natural land',
        'Unallocated - natural land',
    ],
    'AgTech EI': [
        # Cropping:
        'Hay',
        'Summer cereals',
        'Summer legumes',
        'Summer oilseeds',
        'Winter cereals',
        'Winter legumes',
        'Winter oilseeds',
        # Intensive Cropping:
        'Cotton',
        'Other non-cereal crops',
        'Rice',
        'Sugar',
        'Vegetables',
        # Horticulture:
        'Apples',
        'Citrus',
        'Grapes',
        'Nuts',
        'Pears',
        'Plantation fruit',
        'Stone fruit',
        'Tropical stone fruit',
    ]
}

"""
If settings.MODE == 'timeseries', the values of the below dictionary determine whether the model is allowed to abandon agricultural
management options on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Asparagopsis taxiformis',
and the corresponding value in this dictionary is False, all cells using Asparagopsis taxiformis must also utilise this land use
and agricultural management combination in all subsequent years.

WARNING: changing to False will result in 'locking in' land uses on cells that utilise the agricultural management option for 
the rest of the simulation. This may be an unintended side effect.
"""

AG_MANAGEMENTS_REVERSIBLE = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
}
