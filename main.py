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
