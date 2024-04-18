import pandas as pd

# import luto settings
from luto.settings import (INPUT_DIR, 
                           SCENARIO, 
                           DIET_DOM, 
                           DIET_GLOB,
                           CONVERGENCE, 
                           IMPORT_TREND, 
                           WASTE, 
                           FEED_EFFICIENCY,
                           EGGS_AVG_WEIGHT)


def get_demand_df(egg_weight=EGGS_AVG_WEIGHT) -> pd.DataFrame:
    """
    Get the demand dataframe for off-land commodities.

    Args:
        egg_weight (int, optional): The weight of each egg in grams. Defaults to settings.EGGS_AVG_WEIGHT.

    Returns:
        pd.DataFrame: The demand dataframe for off-land commodities.
    """

    # Read the demand data
    dd = pd.read_hdf(f'{INPUT_DIR}/demand_projections.h5')

    # Select the demand data under the running scenario
    DEMAND_DATA = dd.loc[(SCENARIO, DIET_DOM, DIET_GLOB, CONVERGENCE,
                          IMPORT_TREND, WASTE, FEED_EFFICIENCY)].copy()

    # Convert eggs from count to tonnes
    DEMAND_DATA.loc['eggs'] = DEMAND_DATA.loc['eggs'] * egg_weight / 1000 / 1000

    # Filter the demand data to only include years up to the target year
    DEMAND_DATA_long = DEMAND_DATA.melt(ignore_index=False,
                                        value_name='Quantity (tonnes, ML)').reset_index()

    DEMAND_DATA_long.columns = ['COMMODITY', 'Type', 'Year', 'Quantity (tonnes, ML)']

    # Rename the columns, so that they are the same with LUTO naming convention
    DEMAND_DATA_long['Type'] = DEMAND_DATA_long['Type'].str.title()
    DEMAND_DATA_long['COMMODITY'] = DEMAND_DATA_long['COMMODITY'] \
        .apply(lambda x: x[0].upper() + x[1:].lower())

    # Sort the dataframe by year, commodity, and type,
    # where the commodity is sorted by the order in COMMODITIES_ALL
    DEMAND_DATA_long = DEMAND_DATA_long.set_index(['Year', 'COMMODITY', 'Type'])


    return DEMAND_DATA_long