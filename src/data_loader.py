import tensorflow as tf
import os

def load_data(data_path):
    # This function creates a "Dataset" object that TensorFlow understands
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels='inferred',     # Automatically uses folder names as labels
        label_mode='int',      # Converts labels to numbers (0, 1, 2...)
        color_mode='grayscale',# Vital: We are using Grayscale for binary data
        image_size=(224, 224), # Resizes all malware images to be the same size
        batch_size=32,
        shuffle=True          
    )
    return dataset
my_data = load_data('./malimg_dataset/test')