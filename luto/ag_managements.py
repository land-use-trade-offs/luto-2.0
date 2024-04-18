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
}
