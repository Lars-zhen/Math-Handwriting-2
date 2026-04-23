"""
Image Preprocessing Module
Handles image loading, denoising, binarization and normalization
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (32, 32),
    invert: bool = True
) -> np.ndarray:
    """
    Preprocess a handwritten math expression image

    Processing pipeline:
        1. Load image using OpenCV
        2. Convert to grayscale
        3. Apply median blur for denoising
        4. Apply adaptive threshold for binarization
        5. Morphological closing to connect broken strokes
        6. Resize to target dimensions
        7. Normalize to [0, 1] range
        8. Add channel dimension

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target (width, height), default (32, 32)
        invert (bool): Whether to invert colors (default: True for white strokes)

    Returns:
        np.ndarray: Preprocessed image of shape (1, H, W)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Apply median blur for noise reduction
    # Kernel size 3 is effective for salt-and-pepper noise
    denoised = cv2.medianBlur(image, ksize=3)

    # Adaptive thresholding for binarization
    # Better than global thresholding for varying lighting conditions
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Morphological closing to connect broken strokes
    # Rectangle kernel 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Resize to target dimensions
    resized = cv2.resize(closed, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] range
    normalized = resized.astype(np.float32) / 255.0

    # Invert if needed (white strokes on black background -> black strokes on white)
    if invert:
        normalized = 1.0 - normalized

    # Add channel dimension: (H, W) -> (1, H, W)
    result = np.expand_dims(normalized, axis=0)

    return result


def preprocess_image_batch(
    image_paths: list,
    target_size: Tuple[int, int] = (32, 32)
) -> np.ndarray:
    """
    Preprocess a batch of images

    Args:
        image_paths (list): List of image paths
        target_size (tuple): Target (width, height)

    Returns:
        np.ndarray: Batch of preprocessed images of shape (B, 1, H, W)
    """
    processed_images = []
    for path in image_paths:
        img = preprocess_image(path, target_size)
        processed_images.append(img)

    return np.stack(processed_images, axis=0)


def denoise_image(image: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Apply denoising to an image

    Args:
        image (np.ndarray): Input image (grayscale)
        method (str): Denoising method ('median', 'gaussian', 'bilateral')

    Returns:
        np.ndarray: Denoised image
    """
    if method == 'median':
        return cv2.medianBlur(image, ksize=3)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def binarize_image(
    image: np.ndarray,
    method: str = 'adaptive',
    threshold: int = 127
) -> np.ndarray:
    """
    Binarize an image

    Args:
        image (np.ndarray): Input grayscale image
        method (str): Binarization method ('adaptive', 'otsu', 'global')
        threshold (int): Threshold value for global method

    Returns:
        np.ndarray: Binarized image
    """
    if method == 'adaptive':
        return cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
    elif method == 'otsu':
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    elif method == 'global':
        _, binary = cv2.threshold(
            image, threshold, 255, cv2.THRESH_BINARY_INV
        )
        return binary
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def remove_border(
    image: np.ndarray,
    border_size: int = 2
) -> np.ndarray:
    """
    Remove borders from image and crop to content

    Args:
        image (np.ndarray): Input image
        border_size (int): Border size to remove

    Returns:
        np.ndarray: Cropped image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Find non-zero pixels
    coords = cv2.findNonZero(255 - gray)
    x, y, w, h = cv2.boundingRect(coords)

    # Add margin
    x = max(0, x - border_size)
    y = max(0, y - border_size)
    w = min(image.shape[1] - x, w + 2 * border_size)
    h = min(image.shape[0] - y, h + 2 * border_size)

    return image[y:y+h, x:x+w]


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fill_value: int = 255
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio using padding

    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target (width, height)
        fill_value (int): Padding value

    Returns:
        np.ndarray: Resized and padded image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded image
    padded = np.full((target_h, target_w), fill_value, dtype=image.dtype)

    # Calculate padding offsets
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # Place resized image in center
    padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    return padded


def apply_morphology(
    image: np.ndarray,
    operation: str = 'close',
    kernel_size: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Apply morphological operations

    Args:
        image (np.ndarray): Input binary image
        operation (str): Operation type ('open', 'close', 'erode', 'dilate')
        kernel_size (tuple): Kernel size

    Returns:
        np.ndarray: Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    if operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        return cv2.erode(image, kernel)
    elif operation == 'dilate':
        return cv2.dilate(image, kernel)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


def visualize_preprocessing_steps(
    image_path: str,
    target_size: Tuple[int, int] = (32, 32)
) -> dict:
    """
    Visualize all preprocessing steps for debugging

    Args:
        image_path (str): Path to input image
        target_size (tuple): Target size

    Returns:
        dict: Dictionary containing all intermediate results
    """
    results = {}

    # Load
    original = cv2.imread(image_path)
    results['original'] = original
    results['grayscale'] = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Denoise
    results['denoised'] = cv2.medianBlur(results['grayscale'], ksize=3)

    # Binarize
    results['binary'] = cv2.adaptiveThreshold(
        results['denoised'],
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    results['morphed'] = cv2.morphologyEx(results['binary'], cv2.MORPH_CLOSE, kernel)

    # Resize
    resized = cv2.resize(results['morphed'], target_size)
    results['resized'] = resized

    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    results['normalized'] = normalized

    # Invert
    results['final'] = 1.0 - normalized

    return results


if __name__ == "__main__":
    # Test preprocessing
    print("Image preprocessing module loaded successfully")
    print("Available functions:")
    print("  - preprocess_image: Main preprocessing function")
    print("  - preprocess_image_batch: Batch preprocessing")
    print("  - visualize_preprocessing_steps: Debug visualization")
