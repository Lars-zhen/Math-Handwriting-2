"""
Preprocessing Package
Image processing and data augmentation for handwritten math expressions
"""

from .image_processor import (
    preprocess_image,
    preprocess_image_batch,
    denoise_image,
    binarize_image,
    remove_border,
    resize_with_padding,
    apply_morphology,
    visualize_preprocessing_steps
)

from .augmentation import (
    get_train_augmentation,
    get_val_augmentation,
    apply_augmentation,
    RandomAugmenter,
    create_light_augmentation,
    create_strong_augmentation,
    augment_batch
)

from .dataset import (
    MathExpressionDataset,
    SyntheticDataset,
    CSVGenerator
)

__all__ = [
    # Image processing
    'preprocess_image',
    'preprocess_image_batch',
    'denoise_image',
    'binarize_image',
    'remove_border',
    'resize_with_padding',
    'apply_morphology',
    'visualize_preprocessing_steps',
    # Augmentation
    'get_train_augmentation',
    'get_val_augmentation',
    'apply_augmentation',
    'RandomAugmenter',
    'create_light_augmentation',
    'create_strong_augmentation',
    'augment_batch',
    # Dataset
    'MathExpressionDataset',
    'SyntheticDataset',
    'CSVGenerator'
]
