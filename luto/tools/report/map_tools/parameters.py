from luto.tools.report.data_tools.parameters import AG_LANDUSE, NON_AG_LANDUSE


# The ag management names
ag_management = ['Asparagopsis taxiformis', 
                 'Precision Agriculture', 
                 'Ecological Grazing']


# The val-color(HEX) records for each map type
color_types ={
            # Integer rasters
            'lumap':   ['luto/tools/report/Assets/lumap_colors.csv',
                        'luto/tools/report/Assets/lumap_colors_grouped.csv'],
            'lmmap':    ['luto/tools/report/Assets/lm_colors.csv'],
            'ammap':    ['luto/tools/report/Assets/ammap_colors.csv'],
            'non_ag':   ['luto/tools/report/Assets/non_ag_colors.csv'],
            # Float rasters
            'Ag_LU':    ['luto/tools/report/Assets/float_img_colors.csv'],
            'Ag_Mgt':   ['luto/tools/report/Assets/float_img_colors.csv'],
            'Land_Mgt': ['luto/tools/report/Assets/float_img_colors.csv'],
            'Non-Ag':   ['luto/tools/report/Assets/float_img_colors.csv']
            }


map_multiple_lucc = {
             'lumap': 'Land-use all category',
             'lmmap': 'Dryland/Irrigated Land-use',
             'ammap': 'Agricultural Management',
             'non_ag': 'Non-Agricultural Land-use',
             'dry': 'Dryland',
             'irr': 'Irrigated Land',
             }

map_single_lucc = AG_LANDUSE + ag_management + NON_AG_LANDUSE
map_single_lucc = {k:k for k in map_single_lucc}

# Dictionary {k:v} for renaming the map names
# if <k> exists in the map name, the map full name will be <v>
map_basename_rename = {**map_multiple_lucc, **map_single_lucc}


# If more than one color_csv were used to create the map, 
# the map_note will be used to identify the color_csv
map_note = {'lumap': [None, 'grouped'],
            'lmmap': [None],
            'ammap': [None],
            'non_ag': [None],
            'Ag_LU': [None],
            'Ag_Mgt': [None],
            'Land_Mgt': [None],
            'Non-Ag': [None]
            }


# The data types for each map type
data_types = {'lumap': 'integer',
                'lmmap': 'integer',
                'ammap': 'integer',
                'non_ag': 'integer',
                'Ag_LU': 'float',
                'Ag_Mgt': 'float',
                'Land_Mgt': 'float',
                'Non-Ag': 'float'
                }


# The legend positions for each map type
legend_positions = {'lumap': 'bottomright',
                    'lmmap': 'bottomright',
                    'ammap': 'bottomright',
                    'non_ag': 'bottomright',
                    'Ag_LU': 'bottomright',
                    'Ag_Mgt': 'bottomright',
                    'Land_Mgt': 'bottomright',
                    'Non-Ag': 'bottomright'
                    }
