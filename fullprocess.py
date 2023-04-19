"""
This module calls functions from other modules to run the full process

Author: Facen
Date: 2023-04-14
"""

import deployment
import logging
import json
import os
import ingestion
from diagnostics import model_predictions
from sklearn.metrics import f1_score
from training import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

input_path = os.path.join(config['input_folder_path'])
output_path = os.path.join(config['output_folder_path'])
prod_dir = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

filenames = os.listdir(prod_dir)

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files = []
with open(os.path.join(prod_dir, 'ingestedfiles.txt')) as file:
    ingested_files = file.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = os.listdir(input_path)

is_new_data = False
for file in source_files:
    if file not in ingested_files:
        is_new_data = True
        break

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not is_new_data:
    logging.info("No new data found. Ending process.")
    exit(0)

ingestion.merge_multiple_dataframe()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_dir, 'latestscore.txt')) as file:
    old_f1 = float(file.read())

pred, y_test = model_predictions(output_path, 'finaldata.csv')
new_f1 = float(f1_score(y_test, pred))

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 >= old_f1:
    logging.info("No model drift found. Ending process.")
    exit(0)

logging.info("retraining and reploying model")
train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle(model_path)

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python3 reporting.py")
os.system("python3 apicalls.py")







