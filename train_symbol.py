"""
Symbol Recognition Training Script
Uses ResNet-18 for improved symbol classification
"""

import os
import sys
import random
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from preprocessing import MathExpressionDataset


class TrainingLogger:
    """Enhanced logger for tracking training metrics"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, 'symbol_training_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'learning_rate', 'epoch_time', 'best_val_acc'
        ])

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict,
                  learning_rate: float, epoch_time: float, best_val_acc: float):
        self.csv_writer.writerow([
            epoch + 1,
            f"{train_metrics['loss']:.6f}",
            f"{train_metrics['accuracy']:.4f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['accuracy']:.4f}",
            f"{learning_rate:.8f}",
            f"{epoch_time:.2f}",
            f"{best_val_acc:.4f}"
        ])
        self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ResNet18SymbolClassifier(nn.Module):
    """ResNet-18 based symbol classifier"""
    
    def __init__(self, num_classes, input_size=64, pretrained=False):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first conv layer for grayscale input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(config, data_dir):
    """Create data loaders for symbol recognition"""
    
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    
    train_img_dir = os.path.join(data_dir, 'train_images')
    val_img_dir = os.path.join(data_dir, 'val_images')
    test_img_dir = os.path.join(data_dir, 'test_images')
    
    # Create datasets with unified label mapping
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Get all unique labels
    all_labels = sorted(set(train_df['label'].unique()) | 
                        set(val_df['label'].unique()) | 
                        set(test_df['label'].unique()))
    label_mapping = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(all_labels)
    
    print(f"Number of symbol classes: {num_classes}")
    
    train_dataset = MathExpressionDataset(
        csv_file=train_csv,
        root_dir=train_img_dir,
        transform=True,
        is_training=True,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        augment=config.ENABLE_AUGMENTATION,
        label_mapping=label_mapping
    )
    
    val_dataset = MathExpressionDataset(
        csv_file=val_csv,
        root_dir=val_img_dir,
        transform=False,
        is_training=False,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        augment=False,
        label_mapping=label_mapping
    )
    
    test_dataset = MathExpressionDataset(
        csv_file=test_csv,
        root_dir=test_img_dir,
        transform=False,
        is_training=False,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        augment=False,
        label_mapping=label_mapping
    )
    
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
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, num_classes


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]', unit='batch', leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return {'loss': running_loss / total, 'accuracy': 100. * correct / total, 'correct': correct, 'total': total}


def validate(model, dataloader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]', unit='batch', leave=False)
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return {'loss': running_loss / total, 'accuracy': 100. * correct / total, 'correct': correct, 'total': total}


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, 
                               scheduler, device, num_epochs, patience, checkpoint_dir, log_dir):
    """Train with early stopping"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = TrainingLogger(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_metrics = validate(model, val_loader, criterion, device, epoch)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['lr'].append(current_lr)

            tb_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            tb_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            tb_writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            tb_writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            tb_writer.add_scalar('Learning_Rate', current_lr, epoch)

            logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time, best_val_acc)

            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss']
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_symbol_model.pth'))
                print(f"  -> Saved best model (Val Acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best epoch: {best_epoch} with Val Acc: {best_val_acc:.2f}%")
                break

    finally:
        logger.close()
        tb_writer.close()

    print("\n" + "=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 60)

    return best_val_acc, best_epoch, history


def main():
    """Main training function"""
    
    # Symbol dataset configuration
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'symbols')
    IMAGE_SIZE = 64  # Larger input for better recognition
    BATCH_SIZE = 32  # Smaller batch for ResNet-18
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    
    print("=" * 60)
    print("Symbol Recognition Training (ResNet-18)")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Set seed
    set_seed(Config.RANDOM_SEED)
    
    # Create dataloaders
    print("\n" + "=" * 50)
    print("Loading Dataset")
    print("=" * 50)
    
    # Temporarily update config for this run
    original_size = Config.IMAGE_SIZE
    original_batch = Config.BATCH_SIZE
    Config.IMAGE_SIZE = IMAGE_SIZE
    Config.BATCH_SIZE = BATCH_SIZE
    
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(Config, DATA_DIR)
    
    # Restore config
    Config.IMAGE_SIZE = original_size
    Config.BATCH_SIZE = original_batch
    
    # Create model
    print("\n" + "=" * 50)
    print("Creating Model")
    print("=" * 50)
    
    model = ResNet18SymbolClassifier(num_classes=num_classes, input_size=IMAGE_SIZE, pretrained=True)
    model = model.to(Config.DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 Symbol Classifier")
    print(f"Parameters: {num_params:,}")
    print(f"Number of classes: {num_classes}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Train
    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'symbols')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'symbols')

    print(f"TensorBoard log directory: {log_dir}")
    print("To view training progress, run: tensorboard --logdir=logs")

    best_val_acc, best_epoch, history = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        Config.DEVICE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, checkpoint_dir, log_dir
    )

    # Save training history
    history_path = os.path.join(checkpoint_dir, 'symbol_training_history.txt')
    with open(history_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Symbol Recognition Training History\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}{'LR':<15}\n")
        f.write("-" * 70 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8}{history['train_loss'][i]:<12.4f}{history['train_acc'][i]:<12.2f}{history['val_loss'][i]:<12.4f}{history['val_acc'][i]:<12.2f}{history['lr'][i]:<15.8f}\n")
    print(f"\nTraining history saved to: {history_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on Test Set")
    print("=" * 50)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_symbol_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        test_metrics = validate(model, test_loader, criterion, Config.DEVICE, 0)
        
        print(f"\nTest Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Correct predictions: {test_metrics['correct']}/{test_metrics['total']}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
