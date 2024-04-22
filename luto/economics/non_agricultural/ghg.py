import numpy as np
import pandas as pd
from luto.non_ag_landuses import NON_AG_LAND_USES

from luto.data import Data


def get_ghg_reduction_env_plantings(data: Data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA,columns=['ENV_PLANTINGS'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_rip_plantings(data: Data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA,columns=['RIP_PLANTINGS'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_agroforestry(data: Data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA,columns=['AGROFORESTRY'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_carbon_plantings_block(data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA,columns=['CARBON_PLANTINGS_BLOCK'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
    

def get_ghg_reduction_carbon_plantings_belt(data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA,columns=['CARBON_PLANTINGS_BELT'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_reduction_beccs(data, aggregate) -> np.ndarray|pd.DataFrame:
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
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA * 0
    elif aggregate==False:
        return pd.DataFrame(-np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA,columns=['BECCS'])
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_matrix(data: Data, aggregate=True) -> np.ndarray:
    """
    Get the g_rk matrix containing non-agricultural greenhouse gas emissions.

    Parameters:
    - data: The input data for calculating greenhouse gas emissions.
    - aggregate: A boolean flag indicating whether to aggregate the matrices or not. Default is True.

    Returns:
    - If aggregate is True, returns a numpy ndarray representing the aggregated g_rk matrix.
    - If aggregate is False, returns a pandas DataFrame representing the g_rk matrix.

    Raises:
    - KeyError: If the aggregate argument is not a boolean value.

    Note:
    - The function internally calls several other functions to calculate different components of the g_rk matrix.
    """

    non_agr_ghg_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_ghg_matrices['Environmental Plantings'] = get_ghg_reduction_env_plantings(data, aggregate)

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_ghg_matrices['Riparian Plantings'] = get_ghg_reduction_rip_plantings(data, aggregate)

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_ghg_matrices['Agroforestry'] = get_ghg_reduction_agroforestry(data, aggregate)

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_ghg_matrices['Carbon Plantings (Belt)'] = get_ghg_reduction_carbon_plantings_belt(data, aggregate)

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_ghg_matrices['Carbon Plantings (Block)'] = get_ghg_reduction_carbon_plantings_block(data, aggregate)

    if NON_AG_LAND_USES['BECCS']:
        non_agr_ghg_matrices['BECCS'] = get_ghg_reduction_beccs(data, aggregate)
      
    if aggregate==True:
        # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
        non_agr_ghg_matrices = [
            non_agr_ghg_matrix.reshape((data.NCELLS, 1)) for non_agr_ghg_matrix in non_agr_ghg_matrices.values()
        ]
        return np.concatenate(non_agr_ghg_matrices, axis=1)
    
    elif aggregate==False:
        return pd.concat(list(non_agr_ghg_matrices.values()), axis=1)
    else:
    # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
