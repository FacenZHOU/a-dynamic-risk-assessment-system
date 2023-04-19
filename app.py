"""
This module is to setup API

Author: Facen
Date: 2023-04-14
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])

with open(os.path.join(output_model_path, 'trainedmodel.pkl'), "rb") as model:
    prediction_model = pickle.load(model)


# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # call the prediction function created in Step 3
    logging.info('Prediction Endpoint')
    filepath = request.json.get('filepath')
    pred, y_test = model_predictions(file_path=filepath)
    return str(pred)

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    # check the score of the deployed model
    logging.info('Scoring Endpoint')
    sorce = score_model()
    return str(sorce)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    # check means, medians, and modes for each column
    logging.info('Summary Statistics Endpoint')
    stats = dataframe_summary()
    return str(stats)

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def disgnostics():        
    # check timing and percent NA values
    logging.info('Diagnostics Endpoint')
    time = execution_time()
    na = missing_data()
    outdated = outdated_packages_list()
    return str(f"execution_time: {time} \n missing_values: {na} \n outdated_packages: {outdated}")


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
