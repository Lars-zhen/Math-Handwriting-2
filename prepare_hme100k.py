"""
HME100K Dataset Preparation Script
Converts HME100K dataset to the format required by our CNN+SE model
"""

import os
import json
import csv
import shutil
from collections import defaultdict
import random


# Paths
HME100K_DIR = r"E:\Math Hand\HME100K\HME100K"
TRAIN_LABELS = os.path.join(HME100K_DIR, "train", "train_labels.txt")
TRAIN_IMAGES = os.path.join(HME100K_DIR, "train", "train_images")
TEST_LABELS = os.path.join(HME100K_DIR, "test", "test_labels.txt")
TEST_IMAGES = os.path.join(HME100K_DIR, "test", "test_images")

# Output paths
OUTPUT_DIR = r"E:\Math Hand\cnn_se_math\data\raw"
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "train_images")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "val_images")
TEST_IMG_DIR = os.path.join(OUTPUT_DIR, "test_images")


def load_labels(labels_file):
    """
    Load labels from HME100K format (tab-separated: filename, expression)

    Returns:
        dict: {filename: expression}
    """
    labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    filename, expression = parts
                    labels[filename] = expression
    return labels


def create_class_mapping(train_labels, test_labels):
    """
    Create class mapping by assigning integer IDs to unique expressions

    Returns:
        dict: {expression: class_id}
        list: List of unique expressions
    """
    # Collect all unique expressions
    all_expressions = set(train_labels.values())
    all_expressions.update(test_labels.values())

    # Sort for reproducibility
    sorted_expressions = sorted(list(all_expressions))

    # Create mapping
    class_to_idx = {expr: idx for idx, expr in enumerate(sorted_expressions)}

    return class_to_idx, sorted_expressions


def parse_expression_to_label(expression):
    """
    Parse LaTeX expression to create a simplified label
    For HME100K, we use the full expression as the class
    """
    return expression


def prepare_dataset():
    """Main function to prepare the dataset"""
    print("=" * 60)
    print("HME100K Dataset Preparation")
    print("=" * 60)

    # Create output directories
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)

    # Load labels
    print("\n1. Loading labels...")
    train_labels = load_labels(TRAIN_LABELS)
    test_labels = load_labels(TEST_LABELS)
    print(f"   Train samples: {len(train_labels)}")
    print(f"   Test samples: {len(test_labels)}")

    # Create class mapping
    print("\n2. Creating class mapping...")
    class_to_idx, all_classes = create_class_mapping(train_labels, test_labels)
    num_classes = len(all_classes)
    print(f"   Total unique classes: {num_classes}")

    # Save class mapping
    class_mapping_file = os.path.join(OUTPUT_DIR, "class_mapping.json")
    with open(class_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print(f"   Class mapping saved to: {class_mapping_file}")

    # Prepare train data
    print("\n3. Preparing train data...")
    train_data = []
    for filename, expression in train_labels.items():
        src_path = os.path.join(TRAIN_IMAGES, filename)
        if os.path.exists(src_path):
            class_id = class_to_idx[expression]
            train_data.append((filename, class_id))

    print(f"   Found {len(train_data)} train images")

    # Split train into train and val (85:15 ratio)
    print("\n4. Splitting train into train/val (85:15)...")
    random.seed(42)
    random.shuffle(train_data)

    val_size = int(len(train_data) * 0.15)
    val_data = train_data[:val_size]
    train_split = train_data[val_size:]

    print(f"   Train split: {len(train_split)} samples")
    print(f"   Val split: {len(val_data)} samples")

    # Prepare test data
    print("\n5. Preparing test data...")
    test_data = []
    for filename, expression in test_labels.items():
        src_path = os.path.join(TEST_IMAGES, filename)
        if os.path.exists(src_path):
            class_id = class_to_idx[expression]
            test_data.append((filename, class_id))

    print(f"   Found {len(test_data)} test images")

    # Copy images and write CSV files
    print("\n6. Copying images and writing CSV files...")

    # Train
    train_csv_rows = []
    for filename, class_id in train_split:
        src = os.path.join(TRAIN_IMAGES, filename)
        dst = os.path.join(TRAIN_IMG_DIR, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        train_csv_rows.append((filename, class_id))

    with open(TRAIN_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(train_csv_rows)
    print(f"   Train CSV: {TRAIN_CSV} ({len(train_csv_rows)} rows)")

    # Val
    val_csv_rows = []
    for filename, class_id in val_data:
        src = os.path.join(TRAIN_IMAGES, filename)
        dst = os.path.join(VAL_IMG_DIR, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        val_csv_rows.append((filename, class_id))

    with open(VAL_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(val_csv_rows)
    print(f"   Val CSV: {VAL_CSV} ({len(val_csv_rows)} rows)")

    # Test
    test_csv_rows = []
    for filename, class_id in test_data:
        src = os.path.join(TEST_IMAGES, filename)
        dst = os.path.join(TEST_IMG_DIR, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        test_csv_rows.append((filename, class_id))

    with open(TEST_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(test_csv_rows)
    print(f"   Test CSV: {TEST_CSV} ({len(test_csv_rows)} rows)")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Total classes: {num_classes}")
    print(f"  Train samples: {len(train_split)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"  - {TRAIN_CSV}")
    print(f"  - {VAL_CSV}")
    print(f"  - {TEST_CSV}")
    print(f"  - {class_mapping_file}")
    print(f"  - train_images/, val_images/, test_images/")

    return num_classes


if __name__ == "__main__":
    num_classes = prepare_dataset()

    # Update config.py with the correct number of classes
    config_file = r"E:\Math Hand\cnn_se_math\config.py"
    print(f"\nUpdating config.py with NUM_CLASSES = {num_classes}...")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # Replace NUM_CLASSES
    if "NUM_CLASSES = 249" in config_content:
        config_content = config_content.replace("NUM_CLASSES = 249", f"NUM_CLASSES = {num_classes}")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("config.py updated successfully!")
    else:
        print("NUM_CLASSES already set or not found. Please update manually if needed.")
