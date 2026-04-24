"""
Dataset Loader Module
PyTorch Dataset for handwritten math expression recognition
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .image_processor import preprocess_image
from .augmentation import get_train_augmentation, apply_augmentation


class MathExpressionDataset(Dataset):
    """
    Dataset class for handwritten math expressions

    Expected CSV format:
        filename,label
        image001.png,0
        image002.png,1
        ...

    Args:
        csv_file (str): Path to CSV file containing filenames and labels
        root_dir (str): Root directory containing images
        transform (bool): Whether to apply data augmentation (default: True)
        is_training (bool): Whether this is training dataset (affects augmentation)
        target_size (tuple): Target image size (default: (32, 32))
        augment (bool): Enable data augmentation (default: True)
        label_mapping (dict, optional): Pre-defined label to index mapping
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: bool = True,
        is_training: bool = True,
        target_size: tuple = (32, 32),
        augment: bool = True,
        label_mapping: dict = None
    ):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.is_training = is_training
        self.target_size = target_size
        self.augment = augment and is_training

        # Load CSV file
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.data_frame = pd.read_csv(csv_file)

        # Validate CSV columns
        if 'filename' not in self.data_frame.columns or 'label' not in self.data_frame.columns:
            raise ValueError("CSV must contain 'filename' and 'label' columns")

        # Set up augmentation
        if self.augment:
            self.augmentation = get_train_augmentation(
                image_size=target_size[0],
                rotation_limit=15
            )
        else:
            self.augmentation = None

        # Use provided label mapping or create one from this dataset
        if label_mapping is not None:
            self.label_mapping = label_mapping
            self.inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        else:
            # Get class information and create mapping
            self.classes = sorted(self.data_frame['label'].unique())
            self.label_mapping = {cls: idx for idx, cls in enumerate(self.classes)}
            self.inverse_label_mapping = {idx: cls for cls, idx in self.label_mapping.items()}

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset

        Args:
            idx (int): Sample index

        Returns:
            tuple: (image_tensor, label) where image_tensor has shape (1, H, W)
        """
        # Get filename and label
        row = self.data_frame.iloc[idx]
        img_name = row['filename']
        original_label = int(row['label'])
        
        # Map original label to contiguous index
        label = self.label_mapping.get(original_label, original_label)

        # Construct full image path
        img_path = os.path.join(self.root_dir, img_name)

        # Load and preprocess image
        try:
            # Use custom preprocessing
            image = preprocess_image(img_path, target_size=self.target_size)
        except Exception as e:
            # Fallback to PIL + manual preprocessing
            image = self._load_and_preprocess_fallback(img_path)

        # Apply augmentation if enabled
        if self.augment and self.augmentation is not None:
            image = apply_augmentation(image, self.augmentation)

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image).float()

        return image_tensor, label

    def _load_and_preprocess_fallback(self, img_path: str) -> np.ndarray:
        """
        Fallback method using PIL for image loading

        Args:
            img_path (str): Image path

        Returns:
            np.ndarray: Preprocessed image of shape (1, H, W)
        """
        # Load with PIL
        img = Image.open(img_path).convert('L')

        # Resize
        img = img.resize(self.target_size, Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Invert if needed (white on black -> black on white)
        img_array = 1.0 - img_array

        # Add channel dimension
        return np.expand_dims(img_array, axis=0)

    def get_class_counts(self) -> dict:
        """
        Get the count of samples per class

        Returns:
            dict: Dictionary mapping class to sample count
        """
        return self.data_frame['label'].value_counts().to_dict()

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of samples per class (normalized)

        Returns:
            dict: Dictionary mapping class to proportion
        """
        counts = self.get_class_counts()
        total = sum(counts.values())
        return {cls: count / total for cls, count in counts.items()}

    def get_sample_by_class(self, label: int, num_samples: int = 4) -> list:
        """
        Get random samples from a specific class

        Args:
            label (int): Class label
            num_samples (int): Number of samples to retrieve

        Returns:
            list: List of (image_tensor, label) tuples
        """
        class_samples = self.data_frame[self.data_frame['label'] == label]
        indices = class_samples.index.tolist()

        if len(indices) < num_samples:
            num_samples = len(indices)

        selected_indices = np.random.choice(indices, num_samples, replace=False)

        samples = []
        for idx in selected_indices:
            sample = self[idx]
            samples.append(sample)

        return samples


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and debugging
    Generates random math symbol images
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 249,
        image_size: tuple = (32, 32)
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        # Generate random image (simulating stroke patterns)
        image = np.random.rand(1, *self.image_size).astype(np.float32)

        # Generate random label
        label = np.random.randint(0, self.num_classes)

        return torch.from_numpy(image), label


class CSVGenerator:
    """
    Utility class to generate CSV files for the dataset
    """

    @staticmethod
    def create_csv_from_folder(
        folder_path: str,
        output_csv: str,
        file_extension: str = '.png'
    ):
        """
        Create CSV file from folder of images

        Assumes folder structure:
            folder_path/
                class_0/
                    img1.png
                    img2.png
                class_1/
                    img3.png
                    ...

        Args:
            folder_path (str): Root folder containing class subfolders
            output_csv (str): Output CSV file path
            file_extension (str): Image file extension to include
        """
        import glob

        rows = []
        for class_folder in sorted(glob.glob(os.path.join(folder_path, '*'))):
            if not os.path.isdir(class_folder):
                continue

            class_name = os.path.basename(class_folder)
            class_label = int(class_name) if class_name.isdigit() else class_name

            for img_path in glob.glob(os.path.join(class_folder, f'*{file_extension}')):
                img_name = os.path.basename(img_path)
                rows.append({'filename': img_name, 'label': class_label})

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Created CSV with {len(df)} entries: {output_csv}")

    @staticmethod
    def split_dataset(
        csv_file: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Split existing dataset into train/val/test

        Args:
            csv_file (str): Input CSV file
            output_dir (str): Output directory for split CSV files
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            test_ratio (float): Test set ratio
            seed (int): Random seed
        """
        df = pd.read_csv(csv_file)

        # Shuffle
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        # Save
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        print(f"Dataset split complete:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")


if __name__ == "__main__":
    print("Dataset loader module loaded successfully")
    print("Classes available:")
    print("  - MathExpressionDataset: Main dataset class")
    print("  - SyntheticDataset: For testing")
    print("  - CSVGenerator: Utility for creating CSV files")
