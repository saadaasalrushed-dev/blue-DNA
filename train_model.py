"""
Blue DNA - AI Beach Guardian
AI Model Training Script
Trains MobileNetV2 on 50 images (20 plastic, 20 clean, 10 oil)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAINING_DATA_DIR = 'training_data'
MODEL_SAVE_PATH = 'models/pollution_classifier.h5'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50  # More epochs for better learning
LEARNING_RATE = 0.0001

# Class names (must match folder names)
CLASSES = ['clean', 'oil', 'plastic']


def create_model():
    """
    Create MobileNetV2 model with custom classification head
    Uses transfer learning from ImageNet weights
    """
    print("Creating MobileNetV2 model with transfer learning...")
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers (don't train them initially)
    base_model.trainable = False
    
    # Build custom classification head
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Preprocess input for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = Dropout(0.2)(x)
    
    # Dense layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer (3 classes: Clean, Oil, Plastic)
    outputs = Dense(3, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created successfully!")
    return model, base_model


def unfreeze_base_model(base_model, model):
    """
    Unfreeze some layers of base model for fine-tuning
    """
    print("Unfreezing top layers of base model for fine-tuning...")
    
    # Unfreeze top layers
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Fine-tuning from layer {fine_tune_at}")


def create_data_generators():
    """
    Create data generators with augmentation for training and validation
    """
    print("Creating data generators with augmentation...")
    
    # Data augmentation for training
    # Use smaller validation split (10%) to get more training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.1  # 90/10 split (more training data!)
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1  # Match the split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        TRAINING_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Found {train_generator.samples} training images")
    print(f"Found {val_generator.samples} validation images")
    print(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator


def train_model():
    """
    Main training function
    """
    print("=" * 60)
    print("Blue DNA - AI Beach Guardian Model Training")
    print("=" * 60)
    
    # Check if training data exists
    if not os.path.exists(TRAINING_DATA_DIR):
        print(f"ERROR: Training data directory '{TRAINING_DATA_DIR}' not found!")
        print("Please create the following structure:")
        print(f"  {TRAINING_DATA_DIR}/")
        print(f"    clean/ (20 images)")
        print(f"    oil/ (10 images)")
        print(f"    plastic/ (20 images)")
        return
    
    # Check each class folder
    for class_name in CLASSES:
        class_path = os.path.join(TRAINING_DATA_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"WARNING: Class folder '{class_path}' not found!")
        else:
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Found {count} images in {class_name}/")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create model
    model, base_model = create_model()
    
    # Print model summary
    print("\nModel Summary:")
    print("-" * 60)
    model.summary()
    
    # Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Callbacks
    callbacks = [
        # Early stopping - stop if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # More patience (wait longer before stopping)
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate if stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train top layers only (frozen base)
    print("\n" + "=" * 60)
    print("PHASE 1: Training top layers (base model frozen)")
    print("=" * 60)
    
    history1 = model.fit(
        train_generator,
        epochs=20,  # More epochs in phase 1
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen layers
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    
    unfreeze_base_model(base_model, model)
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS - 20,  # Adjust for new phase 1 epochs
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Evaluating final model...")
    print("=" * 60)
    
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
    
    print(f"\nFinal Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    if val_accuracy >= 0.85:
        print("✅ Target accuracy (85%+) achieved!")
    else:
        print(f"⚠️  Accuracy below target. Consider training with more data or more epochs.")
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("Training complete!")


if __name__ == '__main__':
    # Set memory growth to avoid GPU memory issues
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    train_model()

