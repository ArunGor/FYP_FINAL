import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from data_loader import load_data
from model import GrayToRGB

# Load Model and Data
model = tf.keras.models.load_model('./models/malware_resnet50.h5', 
                                   custom_objects={'GrayToRGB': GrayToRGB})
test_ds = load_data('./malimg_dataset/test')

# Get Predictions and True Labels
y_true = []
y_pred = []

for images, labels in test_ds:
    if images.shape[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))


# Create Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_ds.class_names, yticklabels=test_ds.class_names)
plt.title('Malware Classification Confusion Matrix')
plt.ylabel('Actual Family')
plt.xlabel('Predicted Family')
plt.savefig('./output/confusion_matrix.png')
print("Confusion Matrix saved to ./output/confusion_matrix.png")