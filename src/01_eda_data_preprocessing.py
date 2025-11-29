"""
Exploratory Data Analysis and Preprocessing for Medical Image Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MedicalImageEDA:
    def __init__(self, data_dir):
        """
        Initialize EDA class
        Args:
            data_dir: Path to data directory containing train/test/val folders
        """
        self.data_dir = Path(data_dir)
        self.image_paths = []
        self.labels = []
        
    def load_dataset_info(self):
        """Load dataset structure and basic statistics"""
        stats = {}
        
        for split in ['train', 'test', 'val']:
            split_path = self.data_dir / split
            if not split_path.exists():
                continue
                
            for class_name in os.listdir(split_path):
                class_path = split_path / class_name
                if class_path.is_dir():
                    images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
                    stats[f"{split}_{class_name}"] = len(images)
                    
                    # Store paths and labels
                    for img_path in images:
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_name)
        
        return stats
    
    def visualize_class_distribution(self, stats):
        """Visualize class distribution across splits"""
        df = pd.DataFrame(list(stats.items()), columns=['Category', 'Count'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Category', y='Count', palette='viridis')
        plt.xticks(rotation=45)
        plt.title('Class Distribution Across Splits', fontsize=16, fontweight='bold')
        plt.xlabel('Split_Class')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nDataset Statistics:")
        print(df.to_string(index=False))
        
    def analyze_image_properties(self, sample_size=100):
        """Analyze image dimensions, color channels, and pixel distributions"""
        sample_paths = np.random.choice(self.image_paths, min(sample_size, len(self.image_paths)), replace=False)
        
        dimensions = []
        aspect_ratios = []
        mean_intensities = []
        
        for img_path in sample_paths:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                dimensions.append((h, w))
                aspect_ratios.append(w / h)
                mean_intensities.append(img.mean())
        
        # Plot distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Image dimensions
        heights, widths = zip(*dimensions)
        axes[0, 0].scatter(widths, heights, alpha=0.5)
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        axes[0, 0].set_title('Image Dimensions Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Aspect ratios
        axes[0, 1].hist(aspect_ratios, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Aspect Ratio (W/H)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Aspect Ratio Distribution')
        axes[0, 1].axvline(np.median(aspect_ratios), color='red', linestyle='--', label='Median')
        axes[0, 1].legend()
        
        # Mean intensities
        axes[1, 0].hist(mean_intensities, bins=30, edgecolor='black', color='green')
        axes[1, 0].set_xlabel('Mean Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Mean Intensity Distribution')
        
        # Summary stats
        stats_text = f"""
        Dimensions Statistics:
        Mean Width: {np.mean(widths):.1f}
        Mean Height: {np.mean(heights):.1f}
        
        Aspect Ratio:
        Mean: {np.mean(aspect_ratios):.3f}
        Median: {np.median(aspect_ratios):.3f}
        
        Intensity:
        Mean: {np.mean(mean_intensities):.1f}
        Std: {np.std(mean_intensities):.1f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/image_properties.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'mean_aspect_ratio': np.mean(aspect_ratios)
        }
    
    def visualize_sample_images(self, n_samples=8):
        """Display sample images from each class"""
        class_names = list(set(self.labels))
        
        fig, axes = plt.subplots(len(class_names), n_samples // len(class_names), 
                                figsize=(15, 8))
        
        for i, class_name in enumerate(class_names):
            class_paths = [p for p, l in zip(self.image_paths, self.labels) if l == class_name]
            sample_paths = np.random.choice(class_paths, min(n_samples // len(class_names), len(class_paths)), replace=False)
            
            for j, img_path in enumerate(sample_paths):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if len(class_names) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j] if len(class_names) > 1 else axes[j]
                    
                ax.imshow(img)
                ax.axis('off')
                if j == 0:
                    ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Create data generators with augmentation for training
    
    Deep Learning Concepts Applied:
    - Data Augmentation: Improves generalization by artificially expanding dataset
    - Normalization: Scales pixel values to [0,1] for stable training
    - Batch Processing: Efficient memory usage and gradient estimation
    """
    
    # Training data augmentation (IMPORTANT for medical images)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,          # Random rotation
        width_shift_range=0.1,       # Horizontal shift
        height_shift_range=0.1,      # Vertical shift
        shear_range=0.1,             # Shear transformation
        zoom_range=0.1,              # Random zoom
        horizontal_flip=True,        # Mirror images
        fill_mode='nearest',         # Fill strategy for new pixels
        brightness_range=[0.8, 1.2]  # Brightness adjustment
    )
    
    # Validation and test only need rescaling (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Use 'categorical' for multi-class
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def visualize_augmentations(data_dir, class_name='NORMAL', n_examples=5):
    """Visualize the effect of data augmentation"""
    # Get a sample image
    class_path = Path(data_dir) / 'train' / class_name
    sample_img_path = list(class_path.glob('*.jpeg'))[0]
    
    img = cv2.imread(str(sample_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    
    # Create augmented versions
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    fig, axes = plt.subplots(2, n_examples, figsize=(15, 6))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Augmented versions
    img_array = img.reshape((1,) + img.shape)
    
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
        if i >= n_examples - 1:
            break
        axes[0, i+1].imshow(batch[0])
        axes[0, i+1].set_title(f'Augmented {i+1}')
        axes[0, i+1].axis('off')
    
    # Second row
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
        if i >= n_examples:
            break
        axes[1, i].imshow(batch[0])
        axes[1, i].set_title(f'Augmented {n_examples+i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize EDA
    data_dir = "data/raw/chest_xray"  
    eda = MedicalImageEDA(data_dir)
    
    # Run analyses
    print("Loading dataset information...")
    stats = eda.load_dataset_info()
    
    print("\nVisualizing class distribution...")
    eda.visualize_class_distribution(stats)
    
    print("\nAnalyzing image properties...")
    img_props = eda.analyze_image_properties(sample_size=100)
    
    print("\nVisualizing sample images...")
    eda.visualize_sample_images(n_samples=8)
    
    print("\nVisualizing augmentation effects...")
    visualize_augmentations(data_dir)
    
    print("\n✓ EDA Complete! Check the 'results' folder for visualizations.")