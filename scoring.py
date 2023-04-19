"""
This module accomplishes model scoring

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

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


def score_model():
    """
    this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    """
    #load the trained model from the trainedmodel.pkl file
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    
    #load the test data from the testdata.csv file
    df_test = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X_test = df_test.iloc[:, 1:]
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()
    
    #use the model to predict the test data
    y_pred = model.predict(X_test.values)
    
    #calculate the F1 score
    f1 = metrics.f1_score(y_test, y_pred)
    
    #write the F1 score to the latestscore.txt file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(str(f1))

    logging.info(f"Scoring: F1={f1:.2f}")

    return f1


if __name__ == '__main__':
    score_model()
