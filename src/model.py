from keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class GrayToRGB(layers.Layer):
    def __init__(self, **kwargs):
        super(GrayToRGB, self).__init__(**kwargs)

    def call(self, inputs):
        # Repeats the last dimension (1) three times to get (None, 224, 224, 3)
        return tf.repeat(inputs, 3, axis=-1)

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 1)), # Normalize pixels
        layers.Conv2D(32, 3, activation='relu'),            # Find edges
        layers.MaxPooling2D(),                              # Simplify
        layers.Conv2D(64, 3, activation='relu'),            # Find patterns
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')     # Final guess
    ])
    return model