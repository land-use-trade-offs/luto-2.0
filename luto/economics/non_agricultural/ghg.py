import numpy as np
import pandas as pd

from luto.data import Data


def get_ghg_reduction_env_plantings(data: Data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of environmental plantings for each cell.
        Since environmental plantings reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA,columns=['ENV_PLANTINGS'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_rip_plantings(data: Data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of Riparian Plantings for each cell. Same as environmental plantings.
        Since riparian plantings reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA,columns=['RIP_PLANTINGS'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_agroforestry(data: Data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of agroforestry for each cell.
        Since agroforestry reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA,columns=['AGROFORESTRY'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_carbon_plantings_block(data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of carbon plantings (block) for each cell.
        Since carbon plantings reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA,columns=['CARBON_PLANTINGS_BLOCK'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
    

def get_ghg_reduction_carbon_plantings_belt(data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of carbon plantings (belt) for each cell.
        Since carbon plantings reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA,columns=['CARBON_PLANTINGS_BELT'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_beccs(data, aggregate) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
        Greenhouse gas emissions of agroforestry for each cell.
        Since agroforestry reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA,columns=['BECCS'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_matrix(data: Data, aggregate=True) -> np.ndarray:
    """
    Get the g_rk matrix containing non-agricultural greenhouse gas emissions.
    """

    env_plantings_ghg_matrix = get_ghg_reduction_env_plantings(data, aggregate)
    rip_plantings_ghg_matrix = get_ghg_reduction_rip_plantings(data, aggregate)
    agroforestry_ghg_matrix = get_ghg_reduction_agroforestry(data, aggregate)
    carbon_plantings_block_ghg_matrix = get_ghg_reduction_carbon_plantings_block(data, aggregate)
    carbon_plantings_belt_ghg_matrix = get_ghg_reduction_carbon_plantings_belt(data, aggregate)
    beccs_ghg_matrix = get_ghg_reduction_beccs(data, aggregate)
      
    if aggregate==True:
        # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
        non_agr_ghg_matrices = [
            env_plantings_ghg_matrix.reshape((data.NCELLS, 1)),
            rip_plantings_ghg_matrix.reshape((data.NCELLS, 1)),
            agroforestry_ghg_matrix.reshape((data.NCELLS, 1)),
            carbon_plantings_block_ghg_matrix.reshape((data.NCELLS, 1)),
            carbon_plantings_belt_ghg_matrix.reshape((data.NCELLS, 1)),
            beccs_ghg_matrix.reshape((data.NCELLS, 1)),
        ]
        return np.concatenate(non_agr_ghg_matrices, axis=1)
    
    elif aggregate==False:
        return pd.concat(
            [ env_plantings_ghg_matrix, 
              rip_plantings_ghg_matrix, 
              agroforestry_ghg_matrix, 
              carbon_plantings_block_ghg_matrix, 
              carbon_plantings_belt_ghg_matrix,
              beccs_ghg_matrix ], 
            axis=1
        )
    
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
