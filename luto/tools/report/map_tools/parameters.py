from luto.tools.report.data_tools.parameters import AG_LANDUSE, NON_AG_LANDUSE


# The ag management names
ag_management = ['Asparagopsis taxiformis', 
                 'Precision Agriculture', 
                 'Ecological Grazing']


# The val-color(HEX) records for each map type
color_types ={
            # Integer rasters
            'lumap':   ['luto/tools/report/Assests/lumap_colors.csv',
                        'luto/tools/report/Assests/lumap_colors_grouped.csv'],
            'lmmap':    ['luto/tools/report/Assests/lm_colors.csv'],
            'ammap':    ['luto/tools/report/Assests/ammap_colors.csv'],
            'non_ag':   ['luto/tools/report/Assests/non_ag_colors.csv'],
            # Float rasters
            'Ag_LU':    ['luto/tools/report/Assests/float_img_colors.csv'],
            'Ag_Mgt':   ['luto/tools/report/Assests/float_img_colors.csv'],
            'Land_Mgt': ['luto/tools/report/Assests/float_img_colors.csv'],
            'Non-Ag':   ['luto/tools/report/Assests/float_img_colors.csv']
            }

# Define the map names
map_basename_to_change = {
             'lumap': 'Land-use all category',
             'lmmap': 'Dryland/Irrigated Land-use',
             'ammap': 'Agricultural Management',
             'non_ag': 'Non-Agricultural Land-use',
             'dry': 'Dryland',
             'irr': 'Irrigated Land',
             }


map_basename_to_keep = AG_LANDUSE + ag_management + NON_AG_LANDUSE
map_basename_to_keep = {k:k for k in map_basename_to_keep}

map_basename_rename = {**map_basename_to_change, **map_basename_to_keep}


# How many maps will be made for each map-type
map_num = {k:list(range(len(v))) for k,v in color_types.items()}


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
