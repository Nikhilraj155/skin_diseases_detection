"""
Skin Disease Classification Model Training Script (Optimized Version)
======================================================================
This script trains a CNN model for skin disease classification using:
- HAM10000 dataset (data1)
- Kaggle Skin Disease Image Dataset (data2)

The script handles:
- Data loading from both datasets
- Label mapping between datasets
- Image preprocessing (224x224, normalization)
- Data augmentation
- Train/validation split
- CNN training with transfer learning (MobileNetV2/ResNet50)
- Early stopping, model checkpointing
- Model and label mapping save
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Paths
    DATA1_DIR = Path("data1")
    DATA2_DIR = Path("data2")
    HAM_METADATA = DATA1_DIR / "HAM10000_metadata.csv"
    HAM_IMAGES_PART1 = DATA1_DIR / "HAM10000_images_part_1"
    HAM_IMAGES_PART2 = DATA1_DIR / "HAM10000_images_part_2"
    
    # Output paths
    OUTPUT_DIR = Path("trained_model")
    MODEL_PATH = OUTPUT_DIR / "skin_disease_model.h5"
    LABELS_PATH = OUTPUT_DIR / "labels.json"
    
    # Image settings
    IMG_SIZE = (224, 224)
    CHANNELS = 3
    
    # Training settings - OPTIMIZED FOR FASTER TRAINING
    BATCH_SIZE = 64  # Increased for faster processing
    EPOCHS = 10  # Reduced for faster training
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model settings
    USE_TRANSFER_LEARNING = True
    BASE_MODEL = "MobileNetV2"


# ==============================================================================
# LABEL MAPPING
# ==============================================================================

# Unified label mapping - maps all sources to unified class names
UNIFIED_LABELS = {
    # HAM10000
    'nv': 'Melanocytic_Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign_Keratosis',
    'bcc': 'Basal_Cell_Carcinoma',
    'akiec': 'Fungal_Infections',
    'vasc': 'Vascular_Lesions',
    'df': 'Dermatofibroma',
    # Kaggle
    'Eczema': 'Eczema',
    'Melanoma': 'Melanoma',
    'Atopic_Dermatitis': 'Atopic_Dermatitis',
    'Basal_Cell_Carcinoma': 'Basal_Cell_Carcinoma',
    'Melanocytic_Nevi': 'Melanocytic_Nevi',
    'Benign_Keratosis': 'Benign_Keratosis',
    'Psoriasis': 'Psoriasis',
    'Seborrheic_Keratosis': 'Seborrheic_Keratosis',
    'Fungal_Infections': 'Fungal_Infections',
    'Viral_Infections': 'Viral_Infections'
}

# Kaggle folder names to standardized labels
KAGGLE_LABEL_MAP = {
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


def get_all_classes():
    """Returns all unique class names from both datasets"""
    return sorted(set(UNIFIED_LABELS.values()))


# ==============================================================================
# DATA LOADING
# ==============================================================================

def prepare_dataset_directory():
    """Prepare a temporary directory with the merged dataset in Keras-friendly format"""
    print("\n" + "="*60)
    print("Preparing Dataset Directory...")
    print("="*60)
    
    # Create temp directory
    temp_dir = Path("temp_dataset")
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "validation"
    
    # Clean up if exists
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    
    # Get all unique labels
    all_labels = sorted(set(UNIFIED_LABELS.values()))
    
    # Create label directories
    for label in all_labels:
        (train_dir / label).mkdir(exist_ok=True)
        (val_dir / label).mkdir(exist_ok=True)
    
    return temp_dir, train_dir, val_dir, all_labels


def load_ham10000_data():
    """Load HAM10000 dataset with metadata"""
    print("\n" + "="*60)
    print("Loading HAM10000 Dataset...")
    print("="*60)
    
    # Read metadata
    metadata = pd.read_csv(Config.HAM_METADATA)
    print(f"Metadata shape: {metadata.shape}")
    
    # Create image paths and labels list
    ham_data = []
    
    for _, row in metadata.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        
        # Try both image folders
        img_path_1 = Config.HAM_IMAGES_PART1 / f"{image_id}.jpg"
        img_path_2 = Config.HAM_IMAGES_PART2 / f"{image_id}.jpg"
        
        if img_path_1.exists():
            full_path = str(img_path_1)
        elif img_path_2.exists():
            full_path = str(img_path_2)
        else:
            continue
        
        # Map to unified label
        label = UNIFIED_LABELS.get(dx, dx)
        ham_data.append({'path': full_path, 'label': label})
    
    print(f"Loaded {len(ham_data)} HAM10000 images")
    return ham_data


def load_kaggle_data():
    """Load Kaggle dataset from folder names"""
    print("\n" + "="*60)
    print("Loading Kaggle Dataset...")
    print("="*60)
    
    kaggle_data = []
    
    if not Config.DATA2_DIR.exists():
        print("Warning: Kaggle dataset not found")
        return kaggle_data
    
    for folder_name in os.listdir(Config.DATA2_DIR):
        folder_path = Config.DATA2_DIR / folder_name
        
        if not folder_path.is_dir():
            continue
        
        label = KAGGLE_LABEL_MAP.get(folder_name, folder_name)
        unified_label = UNIFIED_LABELS.get(label, label)
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    img_path = os.path.join(root, file)
                    kaggle_data.append({
                        'path': img_path,
                        'label': unified_label
                    })
    
    print(f"Loaded {len(kaggle_data)} Kaggle images")
    return kaggle_data


def copy_images_to_dataset(all_data, train_dir, val_dir, all_labels):
    """Copy images to train/validation directories with label-based split"""
    print("\n" + "="*60)
    print("Copying Images to Dataset Directory...")
    print("="*60)
    
    from sklearn.model_selection import train_test_split
    
    # Group by label
    label_groups = {}
    for item in all_data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # Copy each label group
    total_train = 0
    total_val = 0
    
    for label in all_labels:
        if label not in label_groups:
            continue
            
        items = label_groups[label]
        train_items, val_items = train_test_split(
            items, 
            test_size=Config.VALIDATION_SPLIT, 
            random_state=42,
            shuffle=True
        )
        
        # Copy train images
        import shutil
        for i, item in enumerate(train_items):
            src = item['path']
            dst = train_dir / label / f"{label}_{i}.jpg"
            try:
                shutil.copy2(src, dst)
            except:
                pass
        
        # Copy validation images
        for i, item in enumerate(val_items):
            src = item['path']
            dst = val_dir / label / f"{label}_{i}.jpg"
            try:
                shutil.copy2(src, dst)
            except:
                pass
        
        total_train += len(train_items)
        total_val += len(val_items)
    
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_val}")
    return total_train, total_val


def create_data_generators(train_dir, val_dir):
    """Create training and validation data generators with augmentation"""
    print("\n" + "="*60)
    print("Creating Data Generators...")
    print("="*60)
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    
    print(f"\nTraining classes: {train_generator.num_classes}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator


# ==============================================================================
# MODEL BUILDING
# ==============================================================================

def build_model(num_classes):
    """Build CNN model with MobileNetV2 transfer learning"""
    print("\n" + "="*60)
    print("Building Model: MobileNetV2 (Transfer Learning)")
    print("="*60)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model, base_model


def fine_tune_model(model, base_model):
    """Unfreeze some layers for fine-tuning"""
    print("\n" + "="*60)
    print("Fine-tuning: Unfreezing top layers of base model")
    print("="*60)
    
    # Unfreeze the last 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model recompiled with lower learning rate for fine-tuning")
    
    return model


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(model, train_generator, val_generator, epochs=Config.EPOCHS):
    """Train the model with callbacks"""
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(Config.OUTPUT_DIR / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Phase 1: Train only top layers
    print("\n--- Phase 1: Training top layers ---")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n--- Phase 2: Fine-tuning base model ---")
    model = fine_tune_model(model, model.layers[0])
    
    history_fine = model.fit(
        train_generator,
        epochs=epochs // 2,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    for key in history_fine.history:
        if key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    return model, history


# ==============================================================================
# SAVE MODEL AND LABELS
# ==============================================================================

def save_model_and_labels(model, train_generator):
    """Save the trained model and label mappings"""
    print("\n" + "="*60)
    print("Saving Model and Labels...")
    print("="*60)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save(str(Config.MODEL_PATH))
    print(f"Model saved to: {Config.MODEL_PATH}")
    
    # Save label mapping as JSON
    class_indices = train_generator.class_indices
    idx_to_label = {v: k for k, v in class_indices.items()}
    
    labels_dict = {
        'class_indices': class_indices,
        'idx_to_label': idx_to_label,
        'num_classes': len(class_indices),
        'classes': list(class_indices.keys())
    }
    
    with open(Config.LABELS_PATH, 'w') as f:
        json.dump(labels_dict, f, indent=4)
    print(f"Labels saved to: {Config.LABELS_PATH}")
    
    return Config.MODEL_PATH, Config.LABELS_PATH


def cleanup_temp_directory():
    """Clean up temporary dataset directory"""
    import shutil
    temp_dir = Path("temp_dataset")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("\nTemporary dataset directory cleaned up")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("SKIN DISEASE CLASSIFICATION MODEL TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Image size: {Config.IMG_SIZE}")
    print(f"  - Batch size: {Config.BATCH_SIZE}")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Learning rate: {Config.LEARNING_RATE}")
    print(f"  - Validation split: {Config.VALIDATION_SPLIT}")
    print(f"  - Transfer learning: {Config.USE_TRANSFER_LEARNING}")
    
    try:
        # Step 1: Prepare dataset directory
        temp_dir, train_dir, val_dir, all_labels = prepare_dataset_directory()
        
        # Step 2: Load data
        ham_data = load_ham10000_data()
        kaggle_data = load_kaggle_data()
        
        # Step 3: Merge datasets
        all_data = ham_data + kaggle_data
        print(f"\nTotal images: {len(all_data)}")
        
        # Show class distribution
        df = pd.DataFrame(all_data)
        print("\nCombined Class Distribution:")
        print(df['label'].value_counts())
        
        # Step 4: Copy images to train/val directories
        copy_images_to_dataset(all_data, train_dir, val_dir, all_labels)
        
        # Step 5: Create data generators
        train_generator, val_generator = create_data_generators(train_dir, val_dir)
        
        # Step 6: Build model
        num_classes = train_generator.num_classes
        model, base_model = build_model(num_classes)
        
        # Step 7: Train model
        model, history = train_model(model, train_generator, val_generator)
        
        # Step 8: Evaluate
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        val_loss, val_acc = model.evaluate(val_generator, verbose=1)
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        if 'accuracy' in history.history:
            train_acc = history.history['accuracy'][-1]
            print(f"Training Accuracy: {train_acc:.4f}")
        
        # Step 9: Save model and labels
        model_path, labels_path = save_model_and_labels(model, train_generator)
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nResults:")
        print(f"  - Final Training Accuracy: {train_acc:.4f}")
        print(f"  - Final Validation Accuracy: {val_acc:.4f}")
        print(f"\nSaved files:")
        print(f"  - Model: {model_path}")
        print(f"  - Labels: {labels_path}")
        
    finally:
        # Clean up temp directory
        cleanup_temp_directory()
    
    return model, history


if __name__ == "__main__":
    main()