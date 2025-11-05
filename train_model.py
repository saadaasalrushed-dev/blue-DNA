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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Optional sklearn for detailed metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: sklearn not available. Detailed metrics will be skipped.")

# Configuration
TRAINING_DATA_DIR = 'training_data'
MODEL_SAVE_PATH = 'models/pollution_classifier.h5'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4  # Even smaller batch for better gradient updates
EPOCHS = 150  # More epochs for better learning
LEARNING_RATE = 0.002  # Higher initial learning rate for faster convergence
VAL_SPLIT = 0.12  # Use 88% for training, 12% for validation

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
    
    # Base model (preprocessing will be done in ImageDataGenerator)
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Batch normalization for better training
    x = keras.layers.BatchNormalization()(x)
    
    # Dropout for regularization
    x = Dropout(0.3)(x)
    
    # First dense layer with more neurons
    x = Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second dense layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer (3 classes: Clean, Oil, Plastic)
    outputs = Dense(3, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile model with focal loss for better handling of class imbalance
    # Focal loss focuses learning on hard examples
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
            focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss)
        return focal_loss_fixed
    
    # Use focal loss for better class imbalance handling
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
        loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal loss for imbalanced data
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
    
    # Recompile with lower learning rate and focal loss
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
            focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss)
        return focal_loss_fixed
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 20, beta_1=0.9, beta_2=0.999),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    print(f"Fine-tuning from layer {fine_tune_at}")


def create_data_generators():
    """
    Create data generators with augmentation for training and validation
    CRITICAL FIX: Use MobileNetV2 preprocess_input instead of rescale
    """
    print("Creating data generators with augmentation...")
    
    # More aggressive data augmentation for training
    # Use MobileNetV2 preprocessing function (normalizes to [-1, 1])
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing
        rotation_range=40,  # Increased rotation
        width_shift_range=0.3,  # More shift
        height_shift_range=0.3,
        shear_range=0.3,  # More shear
        zoom_range=0.3,  # More zoom
        horizontal_flip=True,
        vertical_flip=False,  # Keep vertical flip off for beach images
        brightness_range=[0.7, 1.3],  # Wider brightness range
        channel_shift_range=20,  # Color channel shifts
        fill_mode='nearest',
        validation_split=VAL_SPLIT
    )
    
    # No augmentation for validation, but use MobileNetV2 preprocessing
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing
        validation_split=VAL_SPLIT
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
    
    # Calculate class weights to handle imbalance
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    print(f"Class weights (to handle imbalance): {class_weights}")
    
    return train_generator, val_generator, class_weights


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
    train_generator, val_generator, class_weights = create_data_generators()
    
    # Enhanced callbacks for better training
    callbacks = [
        # Early stopping - stop if validation loss doesn't improve
        EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=20,  # Much more patience
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        # Save best model based on validation accuracy
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        # Reduce learning rate if stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=5,
            min_lr=1e-8,
            verbose=1,
            mode='min'
        )
    ]
    
    # Phase 1: Train top layers only (frozen base)
    print("\n" + "=" * 60)
    print("PHASE 1: Training top layers (base model frozen)")
    print("=" * 60)
    print(f"Training for up to 50 epochs...")
    
    history1 = model.fit(
        train_generator,
        epochs=50,  # More epochs in phase 1
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,  # Handle class imbalance
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen layers
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    print(f"Fine-tuning for up to {EPOCHS - 50} epochs...")
    
    unfreeze_base_model(base_model, model)
    
    # Reset callbacks for phase 2 (new learning rate)
    callbacks_phase2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=25,  # Even more patience for fine-tuning
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-8,
            verbose=1,
            mode='min'
        )
    ]
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS - 50,  # Adjust for new phase 1 epochs
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        class_weight=class_weights,  # Handle class imbalance
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Evaluating final model...")
    print("=" * 60)
    
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS:")
    print(f"{'=' * 60}")
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Per-class predictions for diagnostics
    if HAS_SKLEARN:
        print(f"\nPer-class predictions (validation set):")
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = val_generator.classes[:len(predicted_classes)]
        
        # Get class names from generator
        class_names = list(val_generator.class_indices.keys())
        class_indices = {v: k for k, v in val_generator.class_indices.items()}
        
        print(classification_report(true_classes, predicted_classes, 
                                    target_names=[class_indices[i] for i in range(len(class_names))]))
    else:
        print("\n(Install sklearn for detailed per-class metrics)")
    
    if val_accuracy >= 0.85:
        print("✅ Target accuracy (85%+) achieved!")
    elif val_accuracy >= 0.70:
        print("⚠️  Accuracy is acceptable but could be improved.")
        print("   Consider: adding more training data, especially for the 'oil' class")
    else:
        print("⚠️  Accuracy below target. Recommendations:")
        print("   1. Add more training images (especially 'oil' class)")
        print("   2. Check data quality and ensure correct labels")
        print("   3. Try training for more epochs")
    
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

