# Import Library
import pandas as pd

# Load Configuration
import utils as utils


# Create config file to load and dump data

# In[2]:


config_data = utils.config_load()
config_data


# In[3]:


def read_data():
    """Load data and dump data"""

    # Load data
    data_path = config_data['raw_dataset_path']
    data = pd.read_csv(data_path)

    # Validate data shape
    print("Data shape :", data.shape)

    # Pickle dumping (save the result)
    dump_path = config_data['dataset_path']
    utils.pickle_dump(data, dump_path)

    return data


# ## **1.2 Sample Splitting**
1. Split input & output data and dump them
2. Update the config file to contain
    - The input & output data path
    - The output variable name
    - The input columns name


def split_input_output():
    """Split input (predictors) and output (responses)"""

    # Load data
    dataset_path = config_data['dataset_path']
    data = utils.pickle_load(dataset_path)

    # Define y
    response_variable = config_data['response_variable']
    y = data[response_variable]

    # Define x
    X = data.drop(columns = [response_variable],
                axis = 1)

    # Validate the splitting
    print('y shape :', y.shape)
    print('X shape:', X.shape)

    # Dumping
    dump_path_predictors = config_data['predictors_set_path']
    utils.pickle_dump(X, dump_path_predictors)

    dump_path_response = config_data['response_set_path']
    utils.pickle_dump(y, dump_path_response)

    return X,y

# Import library
from sklearn.model_selection import train_test_split

def split_train_test():
    """Split train & test, then dump the data"""

    # Load the X & y
    X = utils.pickle_load(config_data['predictors_set_path'])
    y = utils.pickle_load(config_data['response_set_path'])

    X = X.drop(columns = ['Id'], axis = 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify = y,
                                                        test_size = config_data['test_size'],
                                                        random_state = 42)
    # Validate splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_test shape :', X_test.shape)
    print('y_test shape :', y_test.shape)

    # Dump data
    utils.pickle_dump(X_train, config_data['train_path'][0])
    utils.pickle_dump(y_train, config_data['train_path'][1])
    utils.pickle_dump(X_test, config_data['test_path'][0])
    utils.pickle_dump(y_test, config_data['test_path'][1])

    return X_train, y_train, X_test, y_test

# Execute the functions
if __name__ == '__main__':
    # Load config data
    config_data = utils.config_load()

    # Run all functions
    read_data()
    split_input_output()
    split_train_test()

