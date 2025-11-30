"""
Deep Learning Models for Medical Image Classification
Compatible with train.py expecting:
TransferLearningModel(model_name=..., input_shape=...).get()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, EfficientNetB0, MobileNetV2, DenseNet121,
)
from tensorflow.keras.optimizers import Adam
import numpy as np


# ============================================================
# Baseline CNN
# ============================================================
class BaselineCNN:
    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=1):
        model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.4),

            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid'),
        ])
        return model


# ============================================================
# UPDATED TransferLearningModel 
# ============================================================
class TransferLearningModel:
    """
    Now works as a normal class:
    model = TransferLearningModel("resnet50", (224,224,3))
    model = model.get()
    """

    def __init__(self, model_name="resnet50", input_shape=(224,224,3),
                 num_classes=1, freeze_base=True, trainable_layers=0):

        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.trainable_layers = trainable_layers

        self.model = self._build()

    # ------------------------------------------
    # Build model internally
    # ------------------------------------------
    def _build(self):
        base_models = {
            "resnet50": ResNet50,
            "vgg16": VGG16,
            "efficientnetb0": EfficientNetB0,
            "mobilenetv2": MobileNetV2,
            "densenet121": DenseNet121,
        }

        if self.model_name not in base_models:
            raise ValueError(f"Unsupported model: {self.model_name}")

        base_model = base_models[self.model_name](
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )

        # Freeze logic
        base_model.trainable = not self.freeze_base
        if self.freeze_base and self.trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-self.trainable_layers]:
                layer.trainable = False

        inputs = layers.Input(shape=self.input_shape)

        # Preprocessing choice
        preprocess = {
            "resnet50": tf.keras.applications.resnet.preprocess_input,
            "vgg16": tf.keras.applications.vgg16.preprocess_input,
            "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
            "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
            "densenet121": tf.keras.applications.densenet.preprocess_input,
        }[self.model_name]

        x = preprocess(inputs)
        x = base_model(x, training=False)

        # Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        if self.num_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return models.Model(inputs, outputs)

    def get(self):
        """Return the final Keras model"""
        return self.model


# ============================================================
# Compiler
# ============================================================
class ModelCompiler:
    @staticmethod
    def compile_model(model, learning_rate=1e-4, loss="binary_crossentropy"):
        optimizer = Adam(learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        return model


# ============================================================
# Ensemble (unchanged)
# ============================================================
def create_ensemble_model(models_list, input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    outputs = [m.layers[-2].output for m in models_list]

    if len(outputs) > 1:
        avg = layers.Average()(outputs)
    else:
        avg = outputs[0]

    final = layers.Dense(1, activation="sigmoid")(avg)

    return models.Model(inputs, final)
