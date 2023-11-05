# Import library
import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scaling import predict_score
from pre_processing import transform_woe


# Class for receiving the input
class api_data(BaseModel):
    income : int
    age : int
    experience : int
    current_job_years : int
    current_house_years : int
    married : str
    house_ownership : str
    car_ownership : str
    profession : str
    city : str
    state : str 


# Load config data    
config_data = utils.config_load()

# Create app
app = FastAPI()

# Create an address to return something
@app.get('/')
def home():
    return "Hello world"

# Create an address to perform the prediction
@app.post('/predict')
def get_data(data: api_data):

    # Load columns list for the input
    columns_ = config_data['columns_']
    
    # Ingest the data input
    input_list = [
        data.income, data.age, data.experience, 
        data.current_job_years, data.current_house_years, data.married, 
        data.house_ownership, data.car_ownership, data.profession, 
        data.city, data.state
        ]
    
    # Transform the input to a dataframe
    input_table = pd.DataFrame({'0' : input_list},
                               index = columns_).T
    
    # Predict the credit score
    y_score = predict_score(raw_data = input_table,
                          config_data = config_data)
    
    # Predict the probability of good
    best_model = utils.pickle_load(config_data['best_model_path'])
    input_woe = transform_woe(raw_data = input_table,
                              config_data = config_data)
    y_prob = best_model.predict_proba(input_woe)[0][0]
    y_prob = round(y_prob, 2)

    # Define the recommendation (based on the credit score)
    cutoff_score = config_data['cutoff_score']
    if y_score > cutoff_score:
        y_status = "APPROVE"
    else:
        y_status = "REJECT"

    # Summarize the results of prediction
    results = {
        'Score' : y_score,
        'Proba' : y_prob,
        'Recommendation' : y_status
    }

    return results


if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)
