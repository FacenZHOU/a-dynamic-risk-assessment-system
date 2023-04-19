"""
This module calls API endpoints

Author: Facen
Date: 2023-04-14
"""

import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

#Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json={"filepath": "testdata.csv"}).text
response2 = requests.get(f"{URL}/scoring").text
response3 = requests.get(f"{URL}/summarystats").text
response4 = requests.get(f"{URL}/diagnostics").text

#combine all API responses
responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

#write the responses to your workspace
with open('config.json', 'r') as file:
    config = json.load(file)
    output_model_path = os.path.join(config['output_model_path'])

with open(os.path.join(output_model_path, "apireturns.txt"), "w") as file:
    file.write(responses)



