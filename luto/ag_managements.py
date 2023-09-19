from collections import defaultdict
import pandas as pd
from luto.settings import RAW_DATA


AG_MANAGEMENTS_INDEXING = {
    1: 'Asparagopsis taxiformis',
    2: 'Precision Agriculture',
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
    ]
}

# List of all agricultural managements, sorted by index value defined above
SORTED_AG_MANAGEMENTS = [item[1] for item in sorted(AG_MANAGEMENTS_INDEXING.items(), key=lambda item: item[0])]