import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import os

# 1. Load Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    './malimg_dataset/train',
    image_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    './malimg_dataset/test',
    image_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    shuffle=True
)

# 2. Define the Model Architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Start frozen

model = tf.keras.models.Sequential([
    # IMPORTANT: Force normalization here in case the loader missed it
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(26, activation='softmax')
])

# 3. Setup Callbacks and Weights
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

# Lowering Benign weight to 5.0 to prevent the "Benign-only" bias lock
class_weight = {i: 1.0 for i in range(26)}

# --- PHASE 1: WARM UP (Train the Head only) ---
print("\n[PHASE 1] Training the classification head...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Faster rate for the new layers
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=5, # Just enough to stabilize
    class_weight=class_weight,
    callbacks=[early_stopping]
)

# --- PHASE 2: FINE-TUNING (Unfreeze the Brain) ---
print("\n[PHASE 2] Unfreezing base model for global fine-tuning...")
base_model.trainable = True 

# CRITICAL: We use a microscopic learning rate here so we don't 
# destroy the ImageNet features we're trying to tweak.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=20, # The "real" learning happens here
    class_weight=class_weight,
    callbacks=[early_stopping]
)

# 4. Save the model
os.makedirs('./models/', exist_ok=True)
model.save('./models/malware_resnet50.h5')
print("\nModel saved successfully as 'malware_resnet50.h5'!")