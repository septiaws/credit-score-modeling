# Import library
import pandas as pd
import numpy as np

# Load configuration
import utils as utils


# Function to generate the WOE mapping dictionary
def get_woe_map_dict():
    """Get the WOE mapping dictionary"""
    # Load the WOE table
    WOE_table = utils.pickle_load(config_data['WOE_table_path'])

    # Initialize the dictionary
    WOE_map_dict = {}
    
    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'WOE']])                 # Then select the attribute & WOE
        
        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']
            WOE_map_dict[char][attribute] = woe

    # Validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    # Dump
    utils.pickle_dump(WOE_map_dict, config_data['WOE_map_dict_path'])

    return WOE_map_dict


# Next, transform the inputed data based on the map dictionary above.
# Update the config file to have the path for the new data contains the WOE values.

# Function to replace the raw data in the train set with WOE values
def transform_woe(raw_data=None, type=None, config_data=None):
    """Replace data value with WOE"""
    # Load the numerical columns
    num_cols = config_data['num_columns']

    # Load the WOE_map_dict
    WOE_map_dict = utils.pickle_load(config_data['WOE_map_dict_path'])

    # Load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    # Map the data
    woe_data = raw_data.copy()
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Map the data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

    # Validate
    print('Raw data shape : ', raw_data.shape)
    print('WOE data shape : ', woe_data.shape)

    # Dump data
    if type is not None:
        utils.pickle_dump(woe_data, config_data[f'X_{type}_woe_path'])

    return woe_data


# Execute the functions
if __name__ == "__main__":
    # 1. Load config file
    config_data = utils.config_load()

    # 2. Generate the WOE map dict
    get_woe_map_dict()

    # 3. Transform the raw train set into WOE values
    transform_woe(type='train', config_data=config_data)