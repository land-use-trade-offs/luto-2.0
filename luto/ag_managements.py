# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

import luto.settings as settings

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
    ],
    'Biochar': [
        # Cropping
        'Hay',
        'Summer cereals',
        'Summer legumes',
        'Summer oilseeds',
        'Winter cereals',
        'Winter legumes',
        'Winter oilseeds',
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

# Remove the am if it is set False (i.e., not a valid solution) in the settings
AG_MANAGEMENTS_TO_LAND_USES = {
    k:v  for k,v in AG_MANAGEMENTS_TO_LAND_USES.items() 
    if settings.AG_MANAGEMENTS[k]
}