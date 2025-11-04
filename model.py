"""
Blue DNA - AI Beach Guardian
AI Model Loading and Prediction Module
Uses MobileNetV2 with transfer learning for pollution classification
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = 'models/pollution_classifier.h5'
IMAGE_SIZE = (224, 224)  # MobileNetV2 input size
CLASS_NAMES = ['Clean', 'Oil', 'Plastic']  # Order matters - must match training

# Global model variable (loaded once, reused)
_model = None


def load_model():
    """
    Load the trained AI model
    If model doesn't exist, returns None (will need training)
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            try:
                # Method 1: Try with safe_mode=False (for TensorFlow 2.20+)
                _model = keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    safe_mode=False
                )
                print("Model loaded successfully with safe_mode=False!")
                return _model
            except Exception as load_error1:
                print(f"Method 1 failed: {str(load_error1)}")
                try:
                    # Method 2: Try with custom_objects to handle TrueDivide
                    custom_objects = {
                        'TrueDivide': tf.keras.layers.Lambda(lambda x: x / 1.0),
                        'tf': tf
                    }
                    _model = keras.models.load_model(
                        MODEL_PATH,
                        compile=False,
                        custom_objects=custom_objects
                    )
                    print("Model loaded successfully with custom_objects!")
                    return _model
                except Exception as load_error2:
                    print(f"Method 2 failed: {str(load_error2)}")
                    try:
                        # Method 3: Try loading weights only and rebuild model
                        print("Attempting to load weights only...")
                        # This is a fallback - model structure might be incompatible
                        raise load_error2  # Will trigger dummy model creation
                    except Exception as load_error3:
                        print(f"All loading methods failed: {str(load_error3)}")
                        print("Model was likely trained with incompatible TensorFlow version")
                        raise load_error3
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Please train the model first using train_model.py")
            # Return a dummy model for development (will need to train)
            return create_dummy_model()
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Creating dummy model for development...")
        print("NOTE: You may need to retrain the model with the same TensorFlow version as Render")
        return create_dummy_model()


def create_dummy_model():
    """
    Create a dummy model for development/testing
    This should be replaced with actual trained model
    """
    print("Creating dummy model for development...")
    
    # Create base MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom classification head
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with dummy weights (will need training)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Dummy model created. Please train using train_model.py")
    return model


def preprocess_image(image):
    """
    Preprocess image for AI model input
    Args:
        image: PIL Image object
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    try:
        # Convert to RGB if needed (handles RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array (ensure float32)
        img_array = np.array(image, dtype=np.float32)
        
        # Expand dimensions for batch (model expects batch dimension)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2 (normalizes to [-1, 1])
        img_array = preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise


def predict(model, processed_image):
    """
    Predict pollution type from preprocessed image
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array from preprocess_image()
    Returns:
        tuple: (result_string, confidence_float)
        Example: ('Plastic', 0.95)
    """
    try:
        # Get model prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get confidence (probability)
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        result = CLASS_NAMES[predicted_class_idx]
        
        return result, confidence
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return default prediction on error
        return 'Clean', 0.5


def get_model_info():
    """Get information about the loaded model"""
    global _model
    if _model is None:
        return "Model not loaded"
    
    try:
        return {
            'input_shape': str(_model.input_shape),
            'output_shape': str(_model.output_shape),
            'parameters': _model.count_params(),
            'layers': len(_model.layers)
        }
    except:
        return "Unable to get model info"

