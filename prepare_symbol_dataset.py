"""
Symbol Recognition Dataset Converter
Extracts symbol classes with sufficient samples for training
"""

import os
import pandas as pd
from collections import Counter
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Configuration
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples per symbol class
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_symbol_dataset():
    """Create a balanced symbol recognition dataset"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_raw_dir = os.path.join(base_dir, 'data', 'raw')
    symbol_data_dir = os.path.join(base_dir, 'data', 'symbols')
    
    # Load original data
    train_df = pd.read_csv(os.path.join(data_raw_dir, 'train.csv'))
    
    print(f"Original dataset: {len(train_df)} samples, {train_df['label'].nunique()} unique labels")
    
    # Count samples per label
    label_counts = train_df['label'].value_counts()
    
    # Filter labels with sufficient samples
    sufficient_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    print(f"Labels with >= {MIN_SAMPLES_PER_CLASS} samples: {len(sufficient_labels)}")
    
    # Filter dataframe to only include sufficient labels
    filtered_df = train_df[train_df['label'].isin(sufficient_labels)]
    print(f"Filtered dataset: {len(filtered_df)} samples")
    
    # Create output directories
    os.makedirs(os.path.join(symbol_data_dir, 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(symbol_data_dir, 'val_images'), exist_ok=True)
    os.makedirs(os.path.join(symbol_data_dir, 'test_images'), exist_ok=True)
    
    # Split by class to ensure all splits have all classes
    train_records = []
    val_records = []
    test_records = []
    
    # Create label mapping: original_label -> new_contiguous_label
    label_mapping = {label: idx for idx, label in enumerate(sorted(sufficient_labels))}
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    
    for original_label in sufficient_labels:
        class_samples = filtered_df[filtered_df['label'] == original_label].to_dict('records')
        random.shuffle(class_samples)
        
        n_total = len(class_samples)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        # Split samples
        train_samples = class_samples[:n_train]
        val_samples = class_samples[n_train:n_train + n_val]
        test_samples = class_samples[n_train + n_val:]
        
        new_label = label_mapping[original_label]
        
        for sample in train_samples:
            train_records.append({
                'filename': sample['filename'],
                'label': new_label,
                'original_label': sample['label']
            })
        
        for sample in val_samples:
            val_records.append({
                'filename': sample['filename'],
                'label': new_label,
                'original_label': sample['label']
            })
        
        for sample in test_samples:
            test_records.append({
                'filename': sample['filename'],
                'label': new_label,
                'original_label': sample['label']
            })
    
    # Create DataFrames
    train_df_new = pd.DataFrame(train_records)
    val_df_new = pd.DataFrame(val_records)
    test_df_new = pd.DataFrame(test_records)
    
    # Copy images to new directories
    print("\nCopying images...")
    
    for df, dest_dir in [(train_df_new, 'train_images'), (val_df_new, 'val_images'), (test_df_new, 'test_images')]:
        for _, row in df.iterrows():
            src = os.path.join(data_raw_dir, dest_dir.replace('_images', '_images'), row['filename'])
            dst = os.path.join(symbol_data_dir, dest_dir, row['filename'])
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    # Save CSV files
    train_df_new[['filename', 'label']].to_csv(
        os.path.join(symbol_data_dir, 'train.csv'), index=False
    )
    val_df_new[['filename', 'label']].to_csv(
        os.path.join(symbol_data_dir, 'val.csv'), index=False
    )
    test_df_new[['filename', 'label']].to_csv(
        os.path.join(symbol_data_dir, 'test.csv'), index=False
    )
    
    # Save label mapping for reference
    mapping_df = pd.DataFrame([
        {'new_label': idx, 'original_label': orig, 'count': label_counts[orig]}
        for idx, orig in idx_to_label.items()
    ])
    mapping_df.to_csv(os.path.join(symbol_data_dir, 'label_mapping.csv'), index=False)
    
    print("\n" + "=" * 50)
    print("Symbol Dataset Created!")
    print("=" * 50)
    print(f"Number of classes: {len(sufficient_labels)}")
    print(f"Train samples: {len(train_df_new)}")
    print(f"Val samples: {len(val_df_new)}")
    print(f"Test samples: {len(test_df_new)}")
    print(f"\nDataset saved to: {symbol_data_dir}")
    
    return {
        'num_classes': len(sufficient_labels),
        'train_samples': len(train_df_new),
        'val_samples': len(val_df_new),
        'test_samples': len(test_df_new),
        'label_mapping': label_mapping
    }

if __name__ == '__main__':
    create_symbol_dataset()
