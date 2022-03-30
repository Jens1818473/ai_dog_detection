from io import BytesIO
import os
import sys
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Selects Tensorflow CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disables Tensorflow logs

try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as error:
    print('ERROR -', error)
    input('Press ENTER to exit')
    sys.exit()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f"Using Tensorflow version: {tf.__version__}")


class dog_classification:
    size = (180, 180)  # target size of image for prediction
    model_name = 'undefined'  # name of the model (needs to be in same folder)
    current_directory = os.path.join(os.path.dirname(__file__))  # get file's folder path
    model_path = os.path.join(current_directory, model_name)
    if not os.path.isfile(model_path):
        print('ERROR - Model not found')
        input('Press ENTER to exit')
        sys.exit()

    model = keras.models.load_model(model_path)
    class_names = []

