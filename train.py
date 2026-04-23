"""
Main Training Script
Trains the LeNet-5 + SE Attention model on handwritten math expressions
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, Config
from models import LeNet5WithSE, count_parameters
from preprocessing import MathExpressionDataset
from train import train_with_early_stopping


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(config):
    """
    Create data loaders for train, validation, and test sets

    Args:
        config: Configuration object

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Paths to CSV files and image directories
    train_csv = os.path.join(config.DATA_RAW_DIR, 'train.csv')
    val_csv = os.path.join(config.DATA_RAW_DIR, 'val.csv')
    test_csv = os.path.join(config.DATA_RAW_DIR, 'test.csv')

    train_img_dir = os.path.join(config.DATA_RAW_DIR, 'train_images')
    val_img_dir = os.path.join(config.DATA_RAW_DIR, 'val_images')
    test_img_dir = os.path.join(config.DATA_RAW_DIR, 'test_images')

    # Check if data files exist
    if not os.path.exists(train_csv):
        print(f"Warning: Training CSV not found at {train_csv}")
        print("Please prepare your data according to the README instructions.")
        # Create synthetic data for testing
        print("\nCreating synthetic dataset for testing...")
        from preprocessing import SyntheticDataset
        train_dataset = SyntheticDataset(num_samples=1000, num_classes=config.NUM_CLASSES)
        val_dataset = SyntheticDataset(num_samples=200, num_classes=config.NUM_CLASSES)
        test_dataset = SyntheticDataset(num_samples=200, num_classes=config.NUM_CLASSES)
    else:
        # Load all CSV files to create unified label mapping
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        
        # Get all unique labels from all datasets
        all_labels = sorted(set(train_df['label'].unique()) | set(val_df['label'].unique()) | set(test_df['label'].unique()))
        # Create unified mapping: label -> contiguous index
        unified_label_mapping = {label: idx for idx, label in enumerate(all_labels)}
        num_classes = len(all_labels)
        
        print(f"Unified label mapping created: {num_classes} unique classes")
        
        # Update config
        config.NUM_CLASSES = num_classes
        
        # Create datasets with unified mapping
        train_dataset = MathExpressionDataset(
            csv_file=train_csv,
            root_dir=train_img_dir,
            transform=config.ENABLE_AUGMENTATION,
            is_training=True,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            augment=config.ENABLE_AUGMENTATION,
            label_mapping=unified_label_mapping
        )

        val_dataset = MathExpressionDataset(
            csv_file=val_csv,
            root_dir=val_img_dir,
            transform=False,
            is_training=False,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            augment=False,
            label_mapping=unified_label_mapping
        )

        test_dataset = MathExpressionDataset(
            csv_file=test_csv,
            root_dir=test_img_dir,
            transform=False,
            is_training=False,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            augment=False,
            label_mapping=unified_label_mapping
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_model(config):
    """
    Create and initialize the model

    Args:
        config: Configuration object

    Returns:
        nn.Module: Initialized model
    """
    model = LeNet5WithSE(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        se_reduction_1=config.SE_REDUCTION_RATIO_1,
        se_reduction_2=config.SE_REDUCTION_RATIO_2
    )

    # Count parameters
    num_params = count_parameters(model)
    print(f"\nModel created: LeNet5WithSE")
    print(f"  Parameters: {num_params:,}")

    return model


def main():
    """Main training function"""
    # Create necessary directories
    Config.create_dirs()

    # Print initial configuration
    Config.print_info()

    # Set random seed
    set_seed(Config.RANDOM_SEED)

    # Create data loaders
    print("\n" + "=" * 50)
    print("Loading Dataset")
    print("=" * 50)
    train_loader, val_loader, test_loader = create_dataloaders(Config)
    
    # Print updated class count
    print(f"\nActual number of classes (after unified mapping): {Config.NUM_CLASSES}")

    # Create model
    print("\n" + "=" * 50)
    print("Creating Model")
    print("=" * 50)
    model = create_model(Config)
    model = model.to(Config.DEVICE)
    print(f"Model moved to: {Config.DEVICE}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    print(f"\nLoss function: CrossEntropyLoss")

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    print(f"Optimizer: Adam (lr={Config.LEARNING_RATE}, wd={Config.WEIGHT_DECAY})")

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True
    )
    print(f"LR Scheduler: ReduceLROnPlateau (factor={Config.SCHEDULER_FACTOR}, patience={Config.SCHEDULER_PATIENCE})")

    # Train model (TensorBoard logging is now handled automatically by trainer)
    print(f"\nTensorBoard log directory: {Config.LOG_DIR}")
    print("To view training progress, run: tensorboard --logdir=logs")
    print("\n" + "=" * 50)

    # Train model
    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        log_dir=Config.LOG_DIR
    )

    # Save training history to file
    history_path = os.path.join(Config.CHECKPOINT_DIR, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training History\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}{'LR':<15}\n")
        f.write("-" * 70 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8}{history['train_loss'][i]:<12.4f}{history['train_acc'][i]:<12.2f}{history['val_loss'][i]:<12.4f}{history['val_acc'][i]:<12.2f}{history['lr'][i]:<15.8f}\n")
    print(f"\nTraining history saved to: {history_path}")

    # Load best model and evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on Test Set")
    print("=" * 50)

    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

        # Evaluate on test set
        from train import validate
        test_metrics = validate(model, test_loader, criterion, Config.DEVICE, epoch=0)

        print("\n" + "=" * 50)
        print("Final Test Results")
        print("=" * 50)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Correct predictions: {test_metrics['correct']}/{test_metrics['total']}")

        # Save test results
        results_path = os.path.join(Config.CHECKPOINT_DIR, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("Test Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.2f}%\n")
            f.write(f"Correct predictions: {test_metrics['correct']}/{test_metrics['total']}\n")
            f.write("\nTraining History:\n")
            for i, (train_loss, train_acc, val_loss, val_acc) in enumerate(
                zip(history['train_loss'], history['train_acc'],
                    history['val_loss'], history['val_acc'])
            ):
                f.write(f"Epoch {i+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%\n")
        print(f"\nResults saved to: {results_path}")
    else:
        print("No best model checkpoint found.")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
