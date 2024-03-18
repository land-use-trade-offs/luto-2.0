from luto.tools.report.data_tools.parameters import AG_LANDUSE, NON_AG_LANDUSE


# The ag management names
ag_management = ['Asparagopsis taxiformis', 
                 'Precision Agriculture', 
                 'Ecological Grazing']

# lucc reanaming
lucc_rename = {'Asparagopsis taxiformis' : 'Methane Inhibitors',
               'Precision Agriculture' : 'AgTech Non-Energy'}


# The val-color(HEX) records for each map type
color_types ={
            # Integer rasters
            'lumap':  'luto/tools/report/Assets/lumap_colors_grouped.csv',
            'lmmap': 'luto/tools/report/Assets/lm_colors.csv',
            'ammap': 'luto/tools/report/Assets/ammap_colors.csv',
            'non_ag':'luto/tools/report/Assets/non_ag_colors.csv',
            # Float rasters
            'Ag_LU': 'luto/tools/report/Assets/float_img_colors.csv',
            'Ag_Mgt': 'luto/tools/report/Assets/float_img_colors.csv',
            'Land_Mgt':'luto/tools/report/Assets/float_img_colors.csv',
            'Non-Ag':'luto/tools/report/Assets/float_img_colors.csv'
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


# The parameters for legend
legend_params = {'lumap': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 23,
                            'framealpha': 0,
                            'columnspacing': 1},
                 
                 'lmmap': {'bbox_to_anchor': (0.1, 0.25),
                            'loc': 'upper left',
                            'ncol': 1,
                            'fontsize': 23,
                            'framealpha': 0,
                            'columnspacing': 1},
                 
                 'ammap': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 23,
                            'framealpha': 0,
                            'columnspacing': 1},
                 
                 'non_ag': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 23,
                            'framealpha': 0,
                            'columnspacing': 1},
                 
                 'Ag_LU': {'bbox_to_anchor': (0.09, 0.2), 
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 22,
                            'framealpha': 0},
                 
                 'Ag_Mgt': {'bbox_to_anchor': (0.09, 0.2), 
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 22,
                            'framealpha': 0},
                 
                 'Land_Mgt': {'bbox_to_anchor': (0.09, 0.2), 
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 22,
                            'framealpha': 0},
                 
                 'Non-Ag': {'bbox_to_anchor': (0.09, 0.2), 
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 22,
                            'framealpha': 0},
                }
