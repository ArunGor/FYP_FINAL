# Deep Learning-Based Malware Classification Using Residual Neural Networks

**Author:** Arunachaleshwar Garimalla  
**Institution:** Queen Mary University of London (School of Electronic Engineering and Computer Science)  
**Supervisor:** Dr. Athen Ma  
**Module:** Final Year Undergraduate Project 2025/26  

## 📌 Project Overview
This repository contains the source code for a Deep Learning-based malware classification system. The system utilizes a fine-tuned ResNet-50 architecture to classify polymorphic malware binaries by converting their raw byte streams into 2D structural grayscale images. It includes a CustomTkinter Graphical User Interface (GUI) for real-time inference and integrates SHAP (SHapley Additive exPlanations) for Explainable AI (XAI) validation.

## 📂 Project Structure
Please ensure your directory contains the following files and matches this structure before execution:
```text
├── models/
│   └── malware_resnet50.h5       # The final, pre-trained ResNet-50 model weights
├── malimg_dataset/
│   └── test/                     # Directory containing testing data (benign/malware variants)
├── src/
│   ├── app.py                    # The main CustomTkinter Desktop Application
│   ├── build_benign_set.py       # Utility script for aggregating benign executables
│   ├── data_loader.py            # Logic for reading raw binaries and PNGs 
│   ├── model.py                  # Neural network architecture and custom layers (GrayToRGB)
│   ├── org_dataset.py            # Utility for dataset organization and validation splitting
│   ├── results.py                # Script to generate the Confusion Matrix
│   ├── test_external.py          # Script for testing inference on external/raw binaries
│   └── train_resnet.py           # The two-phase Transfer Learning training script
├── requirements.txt              # Project dependencies
└── README.md                     # Execution instructions (This file)
```
## Prerequisites & Installation
This project was developed and tested using Python 3.10. To ensure full compatibility and avoid dimensionality errors, use Python 3.10.x.

1. Clone or Extract the Repository:
Extract the provided ZIP file. Navigate to the root folder via your terminal.

2. Install Dependencies:
All required packages (including the specific TensorFlow version) can be installed via the provided requirements file:
```sh
pip install -r requirements.txt
```
## Execution Guide

1. Running the Desktop Application (GUI)
To launch the asynchronous frontend scanner:
```sh
python src/app.py
```
Click "Select File" to browse for a target binary (.exe) or dataset image (.png).

Click "Analyze File". The inference runs on a background daemon thread, keeping the UI responsive.

The results will display the predicted class and a confidence threshold decision (Malicious, Benign, or Inconclusive).

2. Generating the Confusion Matrix
To evaluate the model against the holdout validation set and generate the confusion matrix:
```sh
python src/results.py
```

## Generative AI Acknowledge

In accordance with Queen Mary University of London guidelines regarding academic integrity, I formally acknowledge the use of Generative AI during the preparation of this project.

The tool was strictly utilized as a collaborative peer for:

Debugging & Syntax: Troubleshooting specific TensorFlow/Keras dimensionality errors (e.g., tensor shape mismatches).

Formatting: Assisting in the structural layout of the final academic dissertation and markdown documentation.

Accountability Statement: All AI-assisted debugging solutions were manually reviewed, empirically verified, and integrated locally. I claim full responsibility and ownership over the final system implementation and the contents of this repository.
