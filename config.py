"""
Project Configuration File
CNN+SE Attention for Handwritten Math Expression Recognition
"""

import os
import torch

class Config:
    """Project configuration settings"""

    # Project paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')

    # Dataset parameters
    NUM_CLASSES = 46431  # Actual unique classes in training set (89% have only 1 sample)
    IMAGE_SIZE = 32    # LeNet-5 standard input size
    INPUT_CHANNELS = 1  # Grayscale images

    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    DROPOUT_RATE = 0.3

    # Learning rate scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # Data augmentation
    ENABLE_AUGMENTATION = True

    # Device configuration (automatic detection)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    SE_REDUCTION_RATIO_1 = 2  # For first SE block (6 channels)
    SE_REDUCTION_RATIO_2 = 4  # For second SE block (16 channels)

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Number of workers for data loading
    NUM_WORKERS = 4

    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist"""
        dirs = [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            os.path.join(cls.DATA_RAW_DIR, 'train_images'),
            os.path.join(cls.DATA_RAW_DIR, 'test_images'),
            os.path.join(cls.DATA_RAW_DIR, 'val_images'),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def print_info(cls):
        """Print configuration information"""
        print("=" * 50)
        print("Project Configuration:")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Number of classes: {cls.NUM_CLASSES}")
        print(f"Image size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Number of epochs: {cls.NUM_EPOCHS}")
        print(f"Dropout rate: {cls.DROPOUT_RATE}")
        print(f"Data augmentation: {cls.ENABLE_AUGMENTATION}")
        print("=" * 50)


# Create a global config instance
config = Config()
