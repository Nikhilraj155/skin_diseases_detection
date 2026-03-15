"""
Single Image Prediction Script
===============================
This script takes an image path as input and predicts the skin disease.

Usage:
    python predict_disease.py "path/to/image.jpg"
    python predict_disease.py                                # Opens file dialog
"""

import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Paths
    MODEL_PATH = Path("trained_model/best_model.h5")
    LABELS_PATH = Path("trained_model/labels.json")
    
    # Image settings (must match training)
    IMG_SIZE = (224, 224)
    CHANNELS = 3


def load_model_and_labels():
    """Load the trained model and label mappings"""
    print("Loading model and labels...")
    
    # Load model
    model = keras.models.load_model(str(Config.MODEL_PATH))
    print("[OK] Model loaded from:", Config.MODEL_PATH)
    
    # Load labels
    with open(Config.LABELS_PATH, 'r') as f:
        labels_data = json.load(f)
    
    classes = labels_data['classes']
    class_indices = labels_data['class_indices']
    idx_to_label = labels_data['idx_to_label']
    
    print("[OK] Labels loaded:", len(classes), "classes")
    
    return model, classes, class_indices, idx_to_label


def preprocess_image(image_path):
    """Load and preprocess a single image for model prediction"""
    # Load image
    img = Image.open(image_path)
    print(f"[OK] Image loaded: {image_path}")
    print(f"  Original size: {img.size}, Mode: {img.mode}")
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        print(f"  Converting from {img.mode} to RGB...")
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize(Config.IMG_SIZE)
    print(f"  Resized to: {img.size}")
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    # Apply MobileNetV2 preprocessing (scale to [-1, 1])
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_disease(image_path, model, idx_to_label):
    """Predict skin disease from image"""
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred = predictions[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(pred)[::-1][:3]
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    print("\nTop 3 Predictions:")
    print("-"*50)
    
    for rank, idx in enumerate(top_3_indices, 1):
        label = idx_to_label[str(idx)]
        confidence = pred[idx] * 100
        print(f"  {rank}. {label:25s}: {confidence:6.2f}%")
    
    # Get the top prediction
    top_idx = np.argmax(pred)
    top_label = idx_to_label[str(top_idx)]
    top_confidence = pred[top_idx] * 100
    
    print("\n" + "="*50)
    print(f"  PREDICTED DISEASE: {top_label}")
    print(f"  CONFIDENCE: {top_confidence:.2f}%")
    print("="*50)
    
    return top_label, top_confidence, pred


def main():
    """Main prediction function"""
    print("\n" + "="*50)
    print("SKIN DISEASE PREDICTION")
    print("="*50 + "\n")
    
    # Load model and labels first
    model, classes, class_indices, idx_to_label = load_model_and_labels()
    
    # Get image path from user
    print("Enter the path to your image file:")
    print("(or press Enter to use a sample image)")
    image_path = input("\nImage path: ").strip().strip('"')
    
    # Use sample image if no input
    if not image_path:
        image_path = "data2/3. Atopic Dermatitis - 1.25k/t-1IMG001.jpg"
        print(f"Using sample image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        # Try with quotes removed
        image_path2 = image_path.strip("'")
        if os.path.exists(image_path2):
            image_path = image_path2
        else:
            return
    
    # Make prediction
    predict_disease(image_path, model, idx_to_label)


if __name__ == "__main__":
    main()
