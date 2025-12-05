# replace original load block with this
import tensorflow as tf
from tensorflow.keras import layers, models
import os


def build_model(input_shape=(128,128,3)):
    base = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation='relu', name="dense"),
        layers.Dense(128, activation='relu', name="dense_1"),
        layers.Dense(32, activation='relu', name="dense_2"),
        layers.Dense(1, activation='sigmoid', name="dense_3")
    ])
    return model

MODEL_H5 = os.path.join(os.path.dirname(__file__), "saved_model.h5")
model = build_model((128,128,3))
# load weights by name (skip mismatches)
model.load_weights(MODEL_H5, by_name=True, skip_mismatch=True)
print(f"[INFO] Rebuilt model and loaded weights from {MODEL_H5}")
