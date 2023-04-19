"""
This module  copies the model artifacts to a production deployment directory

Author: Facen
Date: 2023-04-14
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
from shutil import copy2

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


# function for deployment
def store_model_into_pickle(model):
    """
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    # copy the files in the model directory to the deployment directory
    for filename in os.listdir(model):
        copy2(os.path.join(model_path, filename), 
              os.path.join(prod_deployment_path, filename))

    # copy the files in the dataset directory to the deployment directory
    copy2(os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
        os.path.join(prod_deployment_path, 'ingestedfiles.txt'))

    logging.info("Artifacts copied to deployment directory")


if __name__ == '__main__':
    store_model_into_pickle(model_path)
        
        
        

