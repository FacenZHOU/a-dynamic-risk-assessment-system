"""
This module is used to generate a confusion matrix using the test data and the deployed model

Author: Facen
Date: 2023-04-14
"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

y_pred, y_test = model_predictions()

 
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))

    logging.info("Saved confusion matrix")


if __name__ == '__main__':
    score_model()
