"""
Deep Learning Models for Medical Image Classification

This module implements multiple CNN architectures:
1. Baseline CNN (custom architecture)
2. Transfer Learning with pre-trained models (ResNet50, EfficientNetB0, VGG16)
3. Fine-tuning strategies
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, EfficientNetB0, MobileNetV2, DenseNet121
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np


class BaselineCNN:
    """
    Custom CNN Architecture - Good for learning fundamentals
    
    Architecture Explanation:
    - Convolutional Layers: Extract hierarchical features (edges -> textures -> patterns)
    - Batch Normalization: Stabilizes training, allows higher learning rates
    - MaxPooling: Reduces spatial dimensions, provides translation invariance
    - Dropout: Prevents overfitting by randomly deactivating neurons
    - Global Average Pooling: Better than Flatten for reducing parameters
    """
    
    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=1):
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid')  # Use 'softmax' for multi-class
        ])
        
        return model


class TransferLearningModel:
    """
    Transfer Learning Implementation
    
    Why Transfer Learning?
    - Pre-trained models learned general features from ImageNet (1.4M images)
    - Saves training time and computational resources
    - Often achieves better performance, especially with limited data
    - Works because low-level features (edges, textures) are universal
    
    Fine-tuning Strategy:
    1. Freeze pre-trained layers initially
    2. Train only the top layers
    3. Optionally unfreeze some layers later for fine-tuning
    """
    
    @staticmethod
    def build_model(base_model_name='ResNet50', input_shape=(224, 224, 3), 
                   num_classes=1, freeze_base=True, trainable_layers=0):
        """
        Build transfer learning model
        
        Args:
            base_model_name: Name of pre-trained model
            input_shape: Input image dimensions
            num_classes: Number of output classes
            freeze_base: Whether to freeze base model weights
            trainable_layers: Number of top layers to make trainable (if freeze_base=True)
        """
        
        # Load pre-trained model without top classification layer
        base_models = {
            'ResNet50': ResNet50,
            'VGG16': VGG16,
            'EfficientNetB0': EfficientNetB0,
            'MobileNetV2': MobileNetV2,
            'DenseNet121': DenseNet121
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Model {base_model_name} not supported")
        
        base_model = base_models[base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model if specified
        base_model.trainable = not freeze_base
        
        # If partially unfreezing, make last N layers trainable
        if freeze_base and trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build complete model
        inputs = layers.Input(shape=input_shape)
        
        # Preprocessing for specific models
        if base_model_name == 'ResNet50':
            x = tf.keras.applications.resnet50.preprocess_input(inputs)
        elif base_model_name == 'VGG16':
            x = tf.keras.applications.vgg16.preprocess_input(inputs)
        elif base_model_name == 'EfficientNetB0':
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        elif base_model_name == 'MobileNetV2':
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        elif base_model_name == 'DenseNet121':
            x = tf.keras.applications.densenet.preprocess_input(inputs)
        else:
            x = inputs
        
        # Base model
        x = base_model(x, training=False)  # Set to False for initial training
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        return model, base_model


class ModelCompiler:
    """
    Compile models with appropriate loss functions and metrics
    
    Deep Learning Concepts:
    - Loss Function: Measures how wrong predictions are
        * Binary Crossentropy: For 2-class problems
        * Categorical Crossentropy: For multi-class problems
    - Optimizer: Algorithm to update weights (Adam is most popular)
    - Metrics: Human-readable performance measures
    """
    
    @staticmethod
    def compile_model(model, learning_rate=0.001, loss='binary_crossentropy'):
        """
        Compile model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            loss: Loss function ('binary_crossentropy' or 'categorical_crossentropy')
        """
        
        # Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=learning_rate)
        
        # Metrics to track
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    @staticmethod
    def get_callbacks(model_name, patience=7):
        """
        Create training callbacks
        
        Callbacks Explained:
        - ModelCheckpoint: Saves best model during training
        - EarlyStopping: Stops training if no improvement (prevents overfitting)
        - ReduceLROnPlateau: Reduces learning rate when stuck (helps convergence)
        """
        
        callbacks = [
            ModelCheckpoint(
                filepath=f'models/{model_name}_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks


def create_ensemble_model(models_list, input_shape=(224, 224, 3)):
    """
    Create an ensemble of multiple models (Advanced technique)
    
    Ensemble Learning: Combines predictions from multiple models
    - Often achieves better performance than single models
    - Reduces variance and overfitting
    - Different models capture different patterns
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # Get predictions from each model
    outputs = []
    for model in models_list:
        # Remove last layer and get features
        x = model.layers[-2].output
        outputs.append(x)
    
    # Average predictions
    averaged = layers.Average()(outputs) if len(outputs) > 1 else outputs[0]
    
    # Final classification
    output = layers.Dense(1, activation='sigmoid')(averaged)
    
    ensemble = models.Model(inputs=inputs, outputs=output)
    
    return ensemble


def print_model_summary(model, model_name):
    """Print detailed model information"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    model.summary()
    
    # Count parameters
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = sum([np.prod(v.shape) for v in model.non_trainable_weights])
    
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")


if __name__ == "__main__":
    # Example usage
    import os
    os.makedirs('models', exist_ok=True)
    
    print("Building Baseline CNN...")
    baseline = BaselineCNN.build_model()
    baseline = ModelCompiler.compile_model(baseline)
    print_model_summary(baseline, "Baseline CNN")
    
    print("\n\nBuilding Transfer Learning Model (ResNet50)...")
    resnet_model, base = TransferLearningModel.build_model('ResNet50')
    resnet_model = ModelCompiler.compile_model(resnet_model)
    print_model_summary(resnet_model, "ResNet50 Transfer Learning")
    
    print("\n✓ Models created successfully!")