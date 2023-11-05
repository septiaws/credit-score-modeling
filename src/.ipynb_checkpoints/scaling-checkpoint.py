# Import library
import pandas as pd
import numpy as np

# Load configuration
import utils as utils


# Function to convert the model's output into score points
def scaling():
    """Function to assign score points to each attribute"""
    
    # Define the references: score, odds, pdo
    pdo = config_data['pdo']
    score = config_data['score_ref']
    odds = config_data['odds_ref']

    # Load the best model
    best_model_path = config_data['best_model_path']
    best_model = utils.pickle_load(best_model_path)

    # Load the WOE table
    WOE_table_path = config_data['WOE_table_path']
    WOE_table = utils.pickle_load(WOE_table_path)

    # Load the best model's estimates table
    best_model_summary_path = config_data['best_model_summary_path']
    best_model_summary = utils.pickle_load(best_model_summary_path)

    # Calculate factor and offset
    factor = pdo/np.log(2)
    offset = score-(factor*np.log(odds))

    print('===================================================')
    print(f"Odds of good of {odds}:1 at {score} points score.")
    print(f"{pdo} PDO (points to double the odds of good).")
    print(f"Offset = {offset:.2f}")
    print(f"Factor = {factor:.2f}")
    print('===================================================')

    # Define n = number of characteristics
    n = best_model_summary.shape[0] - 1

    # Define b0
    b0 = best_model.intercept_[0]

    # Adjust characteristic name in best_model_summary_table
    num_cols = config_data['num_columns']
    for col in best_model_summary['Characteristic']:

        if col in num_cols:
            bin_col = col + '_bin'
        else:
            bin_col = col

        best_model_summary.replace(col, bin_col, inplace = True)

    # Merge tables to get beta/parameter estimate for each characteristic
    scorecards = pd.merge(left = WOE_table,
                        right = best_model_summary,
                        how = 'left',
                        on = ['Characteristic'])

    # Define beta and WOE
    beta = scorecards['Estimate']
    WOE = scorecards['WOE']

    # Calculate the score point for each attribute
    scorecards['Points'] = round((offset/n) - factor*((b0/n) + (beta*WOE)))
    scorecards['Points'] = scorecards['Points'].astype('int')

    # Validate
    print('Scorecards table shape : ', scorecards.shape)

    # Dump the scorecards
    scorecards_path = config_data['scorecards_path']
    utils.pickle_dump(scorecards, scorecards_path)

    return scorecards


# Generate the points map dict function
def get_points_map_dict():
    """Get the Points mapping dictionary"""
    # Load the scorecards table
    scorecards = utils.pickle_load(config_data['scorecards_path'])

    # Initialize the dictionary
    points_map_dict = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characterics
        current_data = (scorecards
                            [scorecards['Characteristic']==char]
                            [['Attribute', 'Points']])

        # Get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']
            points_map_dict[char][attribute] = points

    # Validate data
    print('Number of key : ', len(points_map_dict.keys()))

    # Dump
    utils.pickle_dump(points_map_dict, config_data['points_map_dict_path'])

    return points_map_dict


# Next, transform the raw input data into score points.
def transform_points(raw_data=None, type=None, config_data=None):
    """Replace data value with points"""
    # Load the numerical columns
    num_cols = config_data['num_columns']

    # Load the points_map_dict
    points_map_dict = utils.pickle_load(config_data['points_map_dict_path'])

    # Load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    # Map the data
    points_data = raw_data.copy()
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        points_data[col] = points_data[col].map(points_map_dict[map_col])

    # Dump data
    if type is not None:
        utils.pickle_dump(points_data, config_data[f'X_{type}_points_path'])

    return points_data


# Function to predict the credit score
def predict_score(raw_data, config_data):
    """Function to predict the credit score"""

    points = transform_points(raw_data = raw_data,
                            type = None,
                            config_data = config_data)

    score = int(points.sum(axis=1))

    utils.pickle_dump(score, config_data['score_path'])

    return score

# Execute the functions
if __name__ == "__main__":

    # 1. Load config file
    config_data = utils.config_load()

    # 2. Create the scorecards
    scaling()

    # 3. Generate the points map dict
    get_points_map_dict()

    # 4. Predict the score
    transform_points(type='train', config_data=config_data)