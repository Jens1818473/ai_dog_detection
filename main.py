import matplotlib.pyplot as plt
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.python.data import AUTOTUNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f"Using Tensorflow version: {tf.__version__}")

data_dir = "Raw_data_dogs"

data_dir = pathlib.Path(data_dir)

img_height = 180
img_width = 180
batch_size = 32

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Enter the different types of dog
class_names = class_names = ['A01-15', 'A01-10', 'A01-100', 'A01-120', 'A01-130', 'A01-30',
               'A01-5', 'A01-50', 'A01-60', 'A01-70', 'A01-80', 'A01-90']

print(class_names)