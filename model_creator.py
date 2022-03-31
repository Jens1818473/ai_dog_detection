import pathlib
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.data import AUTOTUNE
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import sys

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

model = Sequential([
    # layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(8, (5, 5), padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(axis=chan_dim),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(axis=chan_dim),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(axis=chan_dim),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(axis=chan_dim),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(axis=chan_dim),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=20,
  #callbacks=callbacks
)

model.save('dog_prediction_model_1.h5')