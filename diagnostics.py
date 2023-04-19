"""
This module acomplishes diagnostic tests on the model and data.

Author: Facen
Date: 2023-04-14
"""

import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])

finaldata = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))

 
def model_predictions(dir_path = test_data_path, file_path = 'testdata.csv'):
    """
    read the deployed model and a test dataset, calculate predictions
    """
    logging.info("calculate model predictions")
    # Load model
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), 'rb') as model:
        model = pickle.load(model)

    # Load data
    test_data = pd.read_csv(os.path.join(dir_path, file_path))
    X_test = test_data.iloc[:, 1:]
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()

    pred = model.predict(X_test.values)

    return pred, y_test

 
def dataframe_summary():
    """
    calculate summary statistics:  means, medians, and standard deviations
    """
    logging.info("calculate statistics on the data")

    numeric = finaldata.select_dtypes(include='int64')
    stats = numeric.iloc[:, :-1].agg(['mean', 'median', 'std'])
    
    return stats


def missing_data():
    """
    calculate percentage of missing data
    """
    logging.info("calculate percentage of missing data")
    nas = list(finaldata.isna().sum())
    napercents = [nas[i] / len(finaldata.index) for i in range(len(nas))]

    return napercents


def execution_time():
    """
    calculate timing of training.py and ingestion.py
    """
    logging.info("calculate execution time")
    timings = []
    scripts = ['ingestion.py', 'training.py']
    for process in scripts:
        starttime = timeit.default_timer()
        # runs the process file
        os.system(f'python3 {process}')
        timing = timeit.default_timer() - starttime
        timings.append(timing)

    return timings

 
def outdated_packages_list():
    """
    check for version of packages
    """
    logging.info("check for version of packages")
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    #print(outdated)
    return str(outdated)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
