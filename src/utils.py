import yaml
import joblib


CONFIG_DIR = 'config/config.yaml'


def config_load():
    """Function to load config files"""
    with open(CONFIG_DIR, 'r') as file:
        config = yaml.safe_load(file)    
    return config

def pickle_load(file_path):
    """Function to load pickle files"""
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """Function to dump data into pickle"""
    joblib.dump(data, file_path)