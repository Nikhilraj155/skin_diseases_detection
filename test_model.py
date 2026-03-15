"""
Skin Disease Model Testing Script
==================================
This script tests the trained skin disease classification model using images
from the data2 directory. It calculates accuracy and detailed metrics.

Usage:
    python test_model.py
"""

import os
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

# For metrics
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Paths
    MODEL_PATH = Path("trained_model/best_model.h5")
    LABELS_PATH = Path("trained_model/labels.json")
    TEST_DATA_DIR = Path("data2")
    
    # Image settings (must match training)
    IMG_SIZE = (224, 224)
    CHANNELS = 3


# Folder name to label mapping (from data2 folder names to model labels)
FOLDER_TO_LABEL = {
    '1. Eczema 1677': 'Eczema',
    '2. Melanoma 15.75k': 'Melanoma',
    '3. Atopic Dermatitis - 1.25k': 'Atopic_Dermatitis',
    '4. Basal Cell Carcinoma (BCC) 3323': 'Basal_Cell_Carcinoma',
    '5. Melanocytic Nevi (NV) - 7970': 'Melanocytic_Nevi',
    '6. Benign Keratosis-like Lesions (BKL) 2624': 'Benign_Keratosis',
    '7. Psoriasis pictures Lichen Planus and related diseases - 2k': 'Psoriasis',
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k': 'Seborrheic_Keratosis',
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 'Fungal_Infections',
    '10. Warts Molluscum and other Viral Infections - 2103': 'Viral_Infections'
}


def load_model_and_labels():
    """Load the trained model and label mappings"""
    print("\n" + "="*60)
    print("Loading Model and Labels...")
    print("="*60)
    
    # Load model
    model = keras.models.load_model(str(Config.MODEL_PATH))
    print(f"✓ Model loaded from: {Config.MODEL_PATH}")
    
    # Load labels
    with open(Config.LABELS_PATH, 'r') as f:
        labels_data = json.load(f)
    
    classes = labels_data['classes']
    class_indices = labels_data['class_indices']
    idx_to_label = labels_data['idx_to_label']
    
    print(f"✓ Labels loaded: {len(classes)} classes")
    print(f"  Classes: {classes}")
    
    return model, classes, class_indices, idx_to_label


def preprocess_image(image_path):
    """Load and preprocess a single image for model prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(Config.IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        
        # Apply MobileNetV2 preprocessing (scale to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, True
    except Exception as e:
        print(f"  Error loading {image_path}: {e}")
        return None, False


def collect_test_images():
    """Collect all test images from data2 directory with their true labels"""
    print("\n" + "="*60)
    print("Collecting Test Images...")
    print("="*60)
    
    test_data = []
    folder_stats = defaultdict(int)
    
    # Scan each folder in data2
    for folder_name in os.listdir(Config.TEST_DATA_DIR):
        folder_path = Config.TEST_DATA_DIR / folder_name
        
        if not folder_path.is_dir():
            continue
        
        # Get the label for this folder
        if folder_name not in FOLDER_TO_LABEL:
            print(f"  Warning: Unknown folder '{folder_name}', skipping...")
            continue
        
        label = FOLDER_TO_LABEL[folder_name]
        
        # Count images in this folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for img_file in os.listdir(folder_path):
            img_path = folder_path / img_file
            
            if img_path.suffix.lower() in image_extensions:
                test_data.append({
                    'path': str(img_path),
                    'true_label': label,
                    'filename': img_file,
                    'folder': folder_name
                })
                folder_stats[label] += 1
    
    print(f"✓ Found {len(test_data)} test images in {len(folder_stats)} categories")
    print("\n  Images per category:")
    for label, count in sorted(folder_stats.items()):
        print(f"    - {label}: {count}")
    
    return test_data


def run_predictions(model, test_data, classes, class_indices, idx_to_label):
    """Run predictions on all test images"""
    print("\n" + "="*60)
    print("Running Predictions...")
    print("="*60)
    
    results = []
    correct = 0
    total = len(test_data)
    
    # Process in batches for efficiency
    batch_size = 32
    num_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_data = test_data[start_idx:end_idx]
        
        # Preprocess batch
        batch_images = []
        batch_valid = []
        
        for item in batch_data:
            img_array, success = preprocess_image(item['path'])
            if success:
                batch_images.append(img_array[0])  # Remove batch dim
                batch_valid.append(item)
        
        if not batch_images:
            continue
        
        # Convert to numpy array
        batch_images = np.array(batch_images)
        
        # Get predictions
        predictions = model.predict(batch_images, verbose=0)
        
        # Process each prediction
        for i, pred in enumerate(predictions):
            pred_class_idx = np.argmax(pred)
            pred_label = idx_to_label[str(pred_class_idx)]
            confidence = pred[pred_class_idx]
            
            true_label = batch_valid[i]['true_label']
            is_correct = (pred_label == true_label)
            
            if is_correct:
                correct += 1
            
            results.append({
                'filename': batch_valid[i]['filename'],
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(confidence),
                'correct': is_correct
            })
        
        # Progress update
        progress = (batch_idx + 1) / num_batches * 100
        print(f"\r  Progress: {progress:.1f}% ({end_idx}/{total} images)", end="")
    
    print()  # New line after progress
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n✓ Completed {total} predictions")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    
    return results, accuracy


def calculate_detailed_metrics(results, classes):
    """Calculate and display detailed metrics"""
    print("\n" + "="*60)
    print("Detailed Metrics")
    print("="*60)
    
    # Per-class statistics
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': defaultdict(int)})
    
    for result in results:
        true_label = result['true_label']
        pred_label = result['predicted_label']
        
        class_stats[true_label]['total'] += 1
        if result['correct']:
            class_stats[true_label]['correct'] += 1
        class_stats[true_label]['predictions'][pred_label] += 1
    
    # Print per-class accuracy
    print("\n  Per-Class Accuracy:")
    print("  " + "-"*55)
    
    class_accuracies = []
    for label in sorted(classes):
        stats = class_stats[label]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
        else:
            accuracy = 0
        class_accuracies.append((label, accuracy, stats['total']))
        print(f"    {label:25s}: {accuracy:6.2f}% ({stats['total']:4d} samples)")
    
    # Confusion matrix style analysis
    print("\n  Prediction Distribution (True -> Predicted):")
    print("  " + "-"*55)
    
    for true_label in sorted(class_stats.keys()):
        stats = class_stats[true_label]
        if stats['total'] == 0:
            continue
        
        print(f"\n    {true_label}:")
        for pred_label in sorted(stats['predictions'].keys()):
            count = stats['predictions'][pred_label]
            if pred_label == true_label:
                print(f"      -> {pred_label:25s}: {count:4d} (correct)")
            else:
                print(f"      -> {pred_label:25s}: {count:4d}")
    
    return class_stats


def print_sample_predictions(results, num_samples=10):
    """Print sample predictions (correct and incorrect)"""
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    
    # Get correct and incorrect predictions
    correct_preds = [r for r in results if r['correct']]
    incorrect_preds = [r for r in results if not r['correct']]
    
    print(f"\n  Correct Predictions (showing up to {num_samples}):")
    print("  " + "-"*55)
    for r in correct_preds[:num_samples]:
        print(f"    ✓ {r['filename'][:40]:40s} | {r['true_label']:20s} | {r['confidence']*100:.1f}%")
    
    if len(correct_preds) > num_samples:
        print(f"    ... and {len(correct_preds) - num_samples} more correct predictions")
    
    print(f"\n  Incorrect Predictions (showing up to {num_samples}):")
    print("  " + "-"*55)
    for r in incorrect_preds[:num_samples]:
        print(f"    ✗ {r['filename'][:40]:40s}")
        print(f"      True: {r['true_label']:20s} | Predicted: {r['predicted_label']} ({r['confidence']*100:.1f}%)")
    
    if len(incorrect_preds) > num_samples:
        print(f"    ... and {len(incorrect_preds) - num_samples} more incorrect predictions")


def save_results(results, accuracy, output_file="test_results.json"):
    """Save test results to JSON file"""
    print(f"\n  Saving results to {output_file}...")
    
    output_data = {
        'accuracy': accuracy,
        'total_images': len(results),
        'correct_predictions': sum(1 for r in results if r['correct']),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  ✓ Results saved to {output_file}")


def main():
    """Main testing function"""
    print("\n" + "="*60)
    print("SKIN DISEASE MODEL TESTING")
    print("="*60)
    print(f"\n  Model: {Config.MODEL_PATH}")
    print(f"  Test Data: {Config.TEST_DATA_DIR}")
    print(f"  Image Size: {Config.IMG_SIZE}")
    
    # Load model and labels
    model, classes, class_indices, idx_to_label = load_model_and_labels()
    
    # Collect test images
    test_data = collect_test_images()
    
    if len(test_data) == 0:
        print("\n  ERROR: No test images found!")
        return
    
    # Run predictions
    results, accuracy = run_predictions(model, test_data, classes, class_indices, idx_to_label)
    
    # Calculate detailed metrics
    calculate_detailed_metrics(results, classes)
    
    # Print sample predictions
    print_sample_predictions(results)
    
    # Save results
    save_results(results, accuracy)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"\n  Final Accuracy: {accuracy:.2f}%")
    print(f"  Total Images Tested: {len(results)}")
    print()


if __name__ == "__main__":
    main()
