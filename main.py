from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
import pandas as pd
import numpy as np
from array import array
import os
from PIL import Image
import sys
import time

data = pd.DataFrame({
    'ImageDescription': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'Confidence': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'Prediction': ['True', 'True', 'True', 'True', 'True', 'False', 'False', 'True', 'True', 'True']
    
                    })

subscription_key = "538da4419e5e45748026b3f29a6b4679"
endpoint = "https://azure-cognitive-services-4494.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
""" images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images") """
folder = '/Users/cankus/Downloads/images'
files = os.listdir(folder)
k = 0
for file in files:
    file_path = os.path.join(folder, file)
    with open(file_path, mode='rb') as image:
        print("===== Tag an image - remote =====")
        description_results = computervision_client.describe_image_in_stream(image)
        print('Description of remote image: ')
        if (len(description_results.captions) == 0):
            print('No descriotion detected')
        else:
            for caption in description_results.captions:
                print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
                data.loc[k, 'ImageDescription'] = caption.text
                data.loc[k, 'Confidence'] = round((caption.confidence*100), 2)

        k += 1

        """ # Print results with confidence score
        print("Tags in the remote image: ")
        if (len(tags_result_remote.tags) == 0):
        print("No tags detected.")
        else:
        for tag in tags_result_remote.tags:
        print("'{}' with confidence {:.2f}%".format(
        tag.name, tag.confidence * 100))
        print()
        print("End of Computer Vision quickstart.") """

pd.set_option('display.max_columns', None)
print(data.head(10))


data.to_csv(r'/Users/cankus/Downloads/csv')
