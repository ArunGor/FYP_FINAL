import glob

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import shap
import matplotlib.pyplot as plt

def file_to_image(file_path, target_size=(224, 224)):
    # 1. Read the raw bytes
    with open(file_path, 'rb') as f:
        content = f.read()
    
    d = np.frombuffer(content, dtype=np.uint8)
    side = int(len(d)**0.5)
    if side == 0: return None
    d = d[:side*side] 
    img_array = d.reshape((side, side))
    
    # 2. Convert to RGB
    img = Image.fromarray(img_array).convert('RGB')
    
    # 3. Resize
    img = img.resize(target_size, Image.BILINEAR)
    
    # 4. PREPARE FOR MODEL (DO NOT DIVIDE BY 255 HERE!)
    # The Rescaling layer inside the model will do the division.
    return np.array(img).reshape(1, 224, 224, 3).astype('float32')

def test_file_to_image(file_path, target_size=(224, 224)):
    # 1. Is it a dataset PNG or a raw EXE?
    if file_path.lower().endswith('.png'):
        # DATASET FILES ARE ALREADY IMAGES. Open them properly!
        img = Image.open(file_path).convert('RGB')
    else:
        # RAW BINARIES NEED TO BE CONVERTED
        with open(file_path, 'rb') as f:
            content = f.read()
        d = np.frombuffer(content, dtype=np.uint8)
        
        # Use the fixed-width logic for binaries
        kb = len(d) / 1024
        if kb < 10: width = 32
        elif kb < 30: width = 64
        elif kb < 60: width = 128
        elif kb < 100: width = 256
        elif kb < 200: width = 384
        elif kb < 500: width = 512
        elif kb < 1000: width = 768
        else: width = 1024
        
        height = len(d) // width
        img_array = d[:width * height].reshape((height, width))
        img = Image.fromarray(img_array).convert('RGB')

    # 2. Resize to match the 99% accuracy model
    img = img.resize(target_size, Image.BILINEAR)
    
    # 3. DO NOT divide by 255. 
    # Your model has a 'layers.Rescaling(1./255)' layer at the top!
    return np.array(img).reshape(1, 224, 224, 3).astype('float32')

# --- CONFIGURATION ---
test_file = 'C:/Windows/System32/calc.exe' 
model_path = './models/malware_resnet50.h5'
output_path = './data/output/hallucination_test.png'

class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 
    'Benign', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
    'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 
    'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 
    'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]

# --- EXECUTION ---
input_data = file_to_image(test_file)


# Load Model (No custom_objects needed anymore!)
model = tf.keras.models.load_model(model_path)

# Predict
preds = model.predict(input_data)
class_idx = np.argmax(preds)
prob_val = np.max(preds) # Decimal probability (e.g., 0.99)
confidence_pct = prob_val * 100
label = class_names[class_idx]

# --- LOGGING ---
print(f"\nFile: {os.path.basename(test_file)}")
print(f"Classified as: {label}")
print(f"Confidence: {confidence_pct:.2f}%")

print(f"DEBUG: Model has {model.output_shape[-1]} output classes.")

print("\nRAW PROBABILITIES:")
for i, prob in enumerate(preds[0]):
    if prob > 0.001: # Only print significant probabilities
        print(f"{class_names[i]}: {prob:.4f}")

# --- DECISION LOGIC ---
THRESHOLD = 0.90 # Using 0.90 because prob_val is a decimal (0.0 to 1.0)

if prob_val < THRESHOLD:
    final_result = "⚠️ INCONCLUSIVE"
    reason = f"Confidence ({confidence_pct:.2f}%) is below the {THRESHOLD*100}% safety limit."
elif label == "Benign":
    final_result = "✅ SAFE"
    reason = f"File identified as Benign System Data with {confidence_pct:.2f}% confidence."
else:
    final_result = f"🚨 MALWARE DETECTED: {label}"
    reason = f"High-confidence match ({confidence_pct:.2f}%) for known malware family."

print("\n" + "="*40)
print(f"FINAL DECISION: {final_result}")
print(f"REASONING: {reason}")
print("="*40 + "\n")

# 1. Pick a specific family to test

target_family = 'Skintrim.N' 
test_folder_path = f'./malimg_dataset/test/{target_family}'

class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 
    'Benign', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
    'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 
    'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 
    'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]

# 2. Get all images in that folder
image_files = glob.glob(os.path.join(test_folder_path, "*.png"))

print(f"Testing {len(image_files)} files from {target_family}...\n")

correct_count = 0

for file_path in image_files: # Testing the first 20 for speed
    # Reuse your existing file_to_image function
    input_data = test_file_to_image(file_path)
    
    preds = model.predict(input_data, verbose=0)
    class_idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    predicted_label = class_names[class_idx]
    for i, prob in enumerate(preds[0]):
        if prob > 0.001: # Only print significant probabilities
            print(f"{class_names[i]}: {prob:.4f}")
    
    if predicted_label == target_family:
        correct_count += 1
        status = "✅ MATCH"
    else:
        status = "❌ MISMATCH"
        
    print(f"File: {os.path.basename(file_path)} | Predicted: {predicted_label} ({confidence:.2f}%) | {status}")

print(f"\nResults: {correct_count}/{min(len(image_files), 20)} correctly identified.")


# --- SHAP EXPLANATION ---
print("Generating SHAP plot (this may take a minute)...")
# Use a background of zeros for clear contrast in the explanation
background = np.zeros((1, 224, 224, 3)) 
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(input_data)

# SHAP returns a list of arrays (one per class). We want the one for our predicted class.
shap.image_plot(shap_values[class_idx], input_data, show=False)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"SHAP plot saved to {output_path}")
