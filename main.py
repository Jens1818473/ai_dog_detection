from io import BytesIO
import os
import sys
import requests
import pathlib
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.data import AUTOTUNE
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

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

data_path = "Raw_data_dogs"

data_dir = pathlib.Path(data_path)

img_height = 180
img_width = 180
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("on")

    plt.show()


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_dataset = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

chan_dim = -1


model = keras.models.load_model('model2803.h5')

# Predict every img in testset
dir_path = 'Dog_or_muffin'

correct = 0
wrong_predictions = []
data_set_size = 5

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size=(img_height, img_width, 3))