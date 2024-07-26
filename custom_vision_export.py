'''
This file is for exporting the model from Azure's custom vision. 
This code is taken from the documentation but still would not work!
'''

from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
import requests
import time

ENDPOINT = "https://facialexpressiondetection.cognitiveservices.azure.com/"
training_key = "277174145c94441ca71bf9610f931a06"

credentials = AzureKeyCredential(training_key)
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

project_id = "bcf06869-03db-4073-aeb8-4e6fae048c36"
iteration_id = "01d08dfd-224a-4240-941c-c98bda55d574"
platform = "TensorFlow"
flavor = "TensorFlowNormal"
export = trainer.export_iteration(project_id, iteration_id, platform, flavor, raw=False)

while (export.status == "Exporting"):
    print ("Waiting 10 seconds...")
    time.sleep(10)
    exports = trainer.get_exports(project_id, iteration_id)
    for e in exports:
        if e.platform == export.platform and e.flavor == export.flavor:
            export = e
            break
    print("Export status is: ", export.status)

if export.status == "Done":
    # Success, now we can download it
    export_file = requests.get(export.download_uri)
    with open("export.zip", "wb") as file:
        file.write(export_file.content)
