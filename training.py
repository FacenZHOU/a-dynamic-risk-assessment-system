"""
This module trains a logistic regression model on the data in the finaldata.csv file
and saves the model to a file called trainedmodel.pkl

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
model_path = os.path.join(config['output_model_path']) 


def train_model():
    """
    Function for training the model
    """
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #load the data from the finaldata.csv file
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = df.iloc[:, 1:]
    y = X.pop('exited').values.reshape(-1, 1).ravel()

    #fit the logistic regression to your data
    model.fit(X.values, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train_model()
    logging.info("Logistic Regression Model successfully trained and saved")