"""
Data Augmentation Module
Uses albumentations library for image augmentation
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_augmentation(
    image_size: int = 32,
    rotation_limit: int = 15,
    noise_var_limit: float = 0.02
) -> A.Compose:
    """
    Get training augmentation pipeline

    Augmentations include:
        - Random rotation
        - Random scale and translation
        - Gaussian noise
        - Elastic deformation (for stroke variation)
        - Random erasing (optional dropout)

    Args:
        image_size (int): Input image size (default: 32)
        rotation_limit (int): Maximum rotation angle in degrees (default: 15)
        noise_var_limit (float): Maximum noise variance (default: 0.02)

    Returns:
        A.Compose: Albumentations composition
    """
    return A.Compose([
        # Random rotation
        A.Rotate(
            limit=rotation_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),

        # Random affine transformations (scale and translation)
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),

        # Gaussian noise
        A.GaussNoise(
            var_limit=(0, noise_var_limit * 255 * 255),
            mean=0,
            per_channel=False,
            p=0.3
        ),

        # Elastic deformation for stroke variation
        A.ElasticTransform(
            alpha=1.0,
            sigma=50,
            alpha_affine=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.2
        ),

        # Random brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            p=0.3
        ),

        # Blur
        A.GaussianBlur(
            blur_limit=(3, 5),
            sigma_limit=0,
            p=0.2
        ),
    ])


def get_val_augmentation() -> A.Compose:
    """
    Get validation augmentation (typically identity/no augmentation)

    Returns:
        A.Compose: Albumentations composition (identity transform)
    """
    return A.Compose([
        # No augmentation for validation
    ])


def apply_augmentation(
    image: np.ndarray,
    transform: A.Compose
) -> np.ndarray:
    """
    Apply augmentation to a single-channel image

    Args:
        image (np.ndarray): Input image of shape (1, H, W) or (H, W)
        transform (A.Compose): Augmentation pipeline

    Returns:
        np.ndarray: Augmented image with same shape as input
    """
    # Remove channel dimension if present
    if len(image.shape) == 3 and image.shape[0] == 1:
        image_2d = image.squeeze(0)  # (1, H, W) -> (H, W)
    else:
        image_2d = image

    # Ensure uint8 for albumentations
    if image_2d.dtype != np.uint8:
        image_2d = (image_2d * 255).astype(np.uint8)

    # Apply transform
    transformed = transform(image=image_2d)

    # Get augmented image
    augmented = transformed['image']

    # Convert back to float32 in [0, 1]
    if augmented.dtype != np.float32:
        augmented = augmented.astype(np.float32) / 255.0

    # Add back channel dimension
    if len(image.shape) == 3 and image.shape[0] == 1:
        augmented = np.expand_dims(augmented, axis=0)

    return augmented


class RandomAugmenter:
    """
    Wrapper class for random augmentation with configurable probability
    """

    def __init__(
        self,
        image_size: int = 32,
        augmentation_prob: float = 0.5,
        rotation_limit: int = 15
    ):
        """
        Initialize random augmenter

        Args:
            image_size (int): Image size
            augmentation_prob (float): Probability of applying augmentation
            rotation_limit (int): Maximum rotation angle
        """
        self.augmentation_prob = augmentation_prob
        self.transform = get_train_augmentation(
            image_size=image_size,
            rotation_limit=rotation_limit
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation with probability

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Augmented image
        """
        if np.random.random() < self.augmentation_prob:
            return apply_augmentation(image, self.transform)
        return image


def create_light_augmentation() -> A.Compose:
    """
    Create light augmentation for small datasets

    Returns:
        A.Compose: Light augmentation pipeline
    """
    return A.Compose([
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
    ])


def create_strong_augmentation() -> A.Compose:
    """
    Create strong augmentation for large datasets

    Returns:
        A.Compose: Strong augmentation pipeline
    """
    return A.Compose([
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.15, 0.15),
            rotate=(-15, 15),
            shear=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.ElasticTransform(alpha=30, sigma=30, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    ])


def augment_batch(
    images: np.ndarray,
    transform: A.Compose,
    labels: np.ndarray = None
) -> tuple:
    """
    Apply augmentation to a batch of images

    Args:
        images (np.ndarray): Batch of images (B, 1, H, W) or (B, H, W)
        transform (A.Compose): Augmentation pipeline
        labels (np.ndarray, optional): Labels (unchanged)

    Returns:
        tuple: (augmented_images, labels)
    """
    augmented_images = []
    batch_size = images.shape[0]

    for i in range(batch_size):
        aug_img = apply_augmentation(images[i], transform)
        augmented_images.append(aug_img)

    augmented_batch = np.stack(augmented_images, axis=0)

    if labels is not None:
        return augmented_batch, labels
    return augmented_batch


if __name__ == "__main__":
    # Test augmentation
    print("Data augmentation module loaded successfully")
    print("Available functions:")
    print("  - get_train_augmentation: Get training augmentation pipeline")
    print("  - get_val_augmentation: Get validation augmentation (no change)")
    print("  - apply_augmentation: Apply augmentation to single image")
    print("  - create_light_augmentation: Light augmentation")
    print("  - create_strong_augmentation: Strong augmentation")

    # Quick test
    test_transform = get_train_augmentation()
    test_image = np.random.rand(32, 32).astype(np.float32)
    result = apply_augmentation(test_image, test_transform)
    print(f"Test passed: input shape {test_image.shape}, output shape {result.shape}")
