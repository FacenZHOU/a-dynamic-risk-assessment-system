"""
This module reads a collection of csv files from a datafolder specified in config.json 
and de-dupe the pandas dataframe and save the output to a csv file in the output folder

Author: Facen
Date: 2023-04-14
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    """
    Function for data ingestion
    """
    logging.info("starting data ingestion process")
    # check for datasets, compile them together, and write to an output file
    current_path = os.getcwd()
    all_files = []
    filenames = os.listdir(input_folder_path)
    
    # Add all files in input folder to a list
    for filename in filenames:
        if filename.endswith(".csv"):
            all_files.append(os.path.join(current_path, input_folder_path, filename))
    
    # Define a dataframe to hold all the data
    df = pd.DataFrame(
        columns=[
            "corporation",
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
            "exited",
        ]
    )
    
    # Loop through all files and append to the dataframe
    for file in all_files:
        df_temp = pd.read_csv(file)
        df = pd.concat([df, df_temp])
    
    # Remove duplicates
    clean_df = df.drop_duplicates()
    
    # Write to output file
    clean_df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)
    print("Data ingestion completed at {}".format(datetime.now()))
    
    # Save the record
    with open(f"{output_folder_path}/ingestedfiles.txt", "w") as f:
        for filename in filenames:
            if filename.endswith(".csv"):
                f.write(filename)
                f.write("\n")

    logging.info(f"record of ingestion saved in {output_folder_path}")


if __name__ == '__main__':
    merge_multiple_dataframe()
