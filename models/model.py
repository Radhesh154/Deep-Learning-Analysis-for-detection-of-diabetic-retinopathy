import os
import cv2
import json
import math
import numpy as np
from PIL import Image

import tensorflow as tf

# Use ONLY tensorflow.keras imports
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
# ImageDataGenerator is NOT needed for inference – we remove it

np.random.seed(2020)
tf.random.set_seed(2020)

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)
    return im

def build_model():
    # Use absolute path to avoid confusion
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up to project root
    weights_path = os.path.join(base_dir, "models", "pretrained", "DenseNet-BC-121-32-no-top.h5")

    densenet = DenseNet121(
        weights=weights_path,
        include_top=False,
        input_shape=(224, 224, 3),
    )

    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.00005),
        metrics=["accuracy"]
    )
    return model

def classify_image(img):
    model = build_model()
    # Also use absolute path for model.h5
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
    weights_path = os.path.join(base_dir, "models", "pretrained", "model.h5")
    model.load_weights(weights_path)

    x_val = np.empty((1, 224, 224, 3), dtype=np.uint8)
    x_val[0, :, :, :] = preprocess_image(img)
    y_val_pred = model.predict(x_val)
    return np.argmax(np.squeeze(y_val_pred))