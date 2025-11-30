import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

from models import BaselineCNN, TransferLearningModel, ModelCompiler


# -------------------------------------------------------------------
# DATASET LOADING FUNCTION
# -------------------------------------------------------------------
def load_data(data_dir, img_size=(224, 224), batch_size=32):

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # STRONG DATA AUGMENTATION (fix overfitting)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size
    )

    val_gen = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# -------------------------------------------------------------------
# CLASS WEIGHTS TO FIX IMBALANCE
# -------------------------------------------------------------------
def compute_class_weights(train_gen):
    labels = train_gen.classes
    class_counts = np.bincount(labels)

    weight_for_0 = len(labels) / (2 * class_counts[0])
    weight_for_1 = len(labels) / (2 * class_counts[1])

    return {0: weight_for_0, 1: weight_for_1}


# -------------------------------------------------------------------
# TRAINING PLOT FUNCTION
# -------------------------------------------------------------------
def plot_training(history, model_name):
    plt.figure(figsize=(15, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training Loss", "Validation Loss"])

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training Accuracy", "Validation Accuracy"])

    # Precision & Recall
    plt.subplot(2, 2, 3)
    if "precision" in history.history:
        plt.plot(history.history["precision"])
    if "val_precision" in history.history:
        plt.plot(history.history["val_precision"])
    if "recall" in history.history:
        plt.plot(history.history["recall"])
    if "val_recall" in history.history:
        plt.plot(history.history["val_recall"])
    plt.title("Precision & Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend(["Train Precision", "Val Precision", "Train Recall", "Val Recall"])

    # AUC
    plt.subplot(2, 2, 4)
    if "auc" in history.history:
        plt.plot(history.history["auc"])
    if "val_auc" in history.history:
        plt.plot(history.history["val_auc"])
    plt.title("AUC (ROC)")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.legend(["Train AUC", "Val AUC"])

    plt.suptitle(f"Training History - {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------------------------
def train_model(args):

    # Load data
    train_gen, val_gen, test_gen = load_data(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    # Class weights (important!)
    class_weights = compute_class_weights(train_gen)
    print("\nClass weights:", class_weights)

    # Choose model
    if args.model == "baseline":
        model = BaselineCNN(input_shape=(args.img_size, args.img_size, 3))
    else:
        model = TransferLearningModel(
            model_name=args.model,
            input_shape=(args.img_size, args.img_size, 3)
        )

    model = ModelCompiler.compile_model(
        model,
        learning_rate=args.learning_rate,
        loss='binary_crossentropy'
    )

    callbacks = ModelCompiler.get_callbacks(model_name)

    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Plot training history
    plot_training(history, args.model)

    # Evaluate on test set
    print("\n Evaluating on TEST SET...")
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_gen)
    print(f"\n TEST ACCURACY: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, AUC: {test_auc:.4f}")

    print("\nTraining Complete!")


# -------------------------------------------------------------------
# CLI ARGUMENTS
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/raw/chest_xray")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["baseline", "resnet50", "vgg16", "mobilenet", "efficientnet"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    train_model(args)
