import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("NumPy version:", np.__version__)
print("ALL PACKAGES WORKING! Ready for pneumonia CNN")
