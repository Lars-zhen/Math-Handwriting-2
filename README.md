# CNN+SE Attention Mechanism for Handwritten Math Expression Recognition

A lightweight handwritten math expression recognition system based on LeNet-5 convolutional neural network and Squeeze-and-Excitation (SE) attention module.

## Project Overview

This project implements a lightweight handwritten math expression recognition model, primarily used for judging the correctness of handwritten algebra answers by middle school students. Key features:

- **LeNet-5 Architecture**: Classic convolutional neural network structure, suitable for small-scale image classification tasks
- **SE Attention Mechanism**: Enhances key features (error symbols, calculation steps) through channel attention, suppresses irrelevant features (illegible strokes)
- **Lightweight Design**: Avoids deep model design to maintain efficient inference speed
- **Data Augmentation**: Multiple augmentation strategies to improve model generalization

## Project Structure

```
cnn_se_math/
├── config.py                 # Configuration file
├── train.py                  # Main training script (expression level)
├── train_symbol.py           # Symbol level training script
├── prepare_hme100k.py        # HME100K dataset preparation script
├── prepare_symbol_dataset.py # Symbol dataset preparation script
├── test_model.py             # Model structure test
├── requirements.txt          # Dependencies
├── README.md                  # Project documentation
├── models/                    # Model definitions
│   ├── __init__.py
│   ├── se_block.py           # SE attention module
│   └── lenet5_se.py          # LeNet-5 + SE model
├── preprocessing/             # Data preprocessing
│   ├── __init__.py
│   ├── image_processor.py    # Image processing
│   ├── augmentation.py       # Data augmentation
│   └── dataset.py            # Dataset loading
├── train/                     # Training module
│   ├── __init__.py
│   └── trainer.py            # Training and validation functions
├── evaluate/                  # Evaluation module
│   ├── __init__.py
│   └── eval.py               # Evaluation script
├── data/                      # Data directory
│   ├── raw/                  # Raw data
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── train_images/
│   │   ├── val_images/
│   │   └── test_images/
│   └── processed/            # Processed data
├── checkpoints/              # Model checkpoints
└── logs/                     # Training logs
```

---

## Environment Setup

### Prerequisites

- Python 3.11 or 3.12 (3.11 recommended for best compatibility)
- NVIDIA GPU (optional, for GPU acceleration)
- CUDA 12.4 or higher (optional)

### Method 1: Recommended - Using Virtual Environment

#### Step 1: Create Virtual Environment

```powershell
# Navigate to project directory
cd e:\Math Hand\cnn_se_math

# Create virtual environment with Python 3.11
py -3.11 -m venv venv
```

#### Step 2: Activate Virtual Environment

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# If you encounter execution policy error, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

#### Step 3: Install PyTorch (CUDA Version)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Install Other Dependencies

```powershell
pip install opencv-python pandas matplotlib scikit-learn albumentations tensorboard tqdm Pillow
```

#### Step 5: Verify Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

### Method 2: Using requirements.txt

```powershell
# Create virtual environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

**Note**: The PyTorch in requirements.txt is CPU version by default. For GPU acceleration, install CUDA version separately:
```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Method 3: System-wide Installation (Not Recommended)

If you don't want to use virtual environment:

```powershell
# Install Python 3.11
# Download: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

# Install PyTorch CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install opencv-python pandas matplotlib scikit-learn albumentations tensorboard tqdm Pillow
```

---

## Data Preparation

### 1. Dataset Source

- [HME100K](http://www.cs.rit.edu/~dprl/HC/HME100K.html) - Handwritten Math Expression Dataset

### 2. Data Format Requirements

Required data format:

```
data/raw/
├── train.csv              # Training set label file
├── val.csv                # Validation set label file
├── test.csv               # Test set label file
├── train_images/           # Training images directory
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── val_images/             # Validation images directory
│   └── ...
└── test_images/            # Test images directory
    └── ...
```

### 3. CSV File Format

CSV files must contain `filename` and `label` columns:

```csv
filename,label
image001.png,0
image002.png,1
image003.png,2
...
```

**Note**:
- `filename` is the image file name
- `label` is an integer class ID (starting from 0)
- HME100K dataset has 249 classes

### 4. Using HME100K Dataset Preparation Script

```powershell
# Run dataset preparation script
python prepare_hme100k.py
```

The script will automatically:
1. Download HME100K dataset
2. Parse expressions and extract images
3. Generate training, validation, and test CSV files
4. Organize data into the required format

### 5. Custom Dataset Preparation

If your images are organized by class folders, use this tool to generate CSV:

```python
from preprocessing import CSVGenerator

# Generate CSV from folder
CSVGenerator.create_csv_from_folder(
    folder_path='path/to/images',
    output_csv='data/raw/train.csv',
    file_extension='.png'
)

# Split dataset
CSVGenerator.split_dataset(
    csv_file='path/to/all_data.csv',
    output_dir='data/raw',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

---

## Training Model

### Basic Training (Expression Level)

```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Run training
python train.py
```

### Symbol Level Training

To train a symbol-level recognition model:

```powershell
python train_symbol.py
```

### Custom Configuration

Modify configuration parameters in `config.py`:

```python
# Training hyperparameters
BATCH_SIZE = 64           # Batch size
LEARNING_RATE = 0.001     # Learning rate
NUM_EPOCHS = 50           # Number of training epochs
DROPOUT_RATE = 0.3        # Dropout ratio

# Dataset parameters
NUM_CLASSES = 249         # Number of classes
IMAGE_SIZE = 32           # Image size

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
```

### Monitor Training

Monitor training process using TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser

---

## Test Model Structure

After installing dependencies, test if the model structure is correct:

```bash
python test_model.py
```

Expected output:
```
====================================================
 CNN+SE MODEL STRUCTURE TEST SUITE
====================================================

==================================================
Testing SE Block...
==================================================
  Input shape: torch.Size([2, 6, 32, 32])
  Output shape: torch.Size([2, 6, 32, 32])
  Channels preserved: ✓
SE Block test: PASSED

...

✓ All tests passed! Model structure is correct.

Model can now be trained with: python train.py
```

---

## Evaluate Model

After training, evaluate model performance:

```bash
python evaluate/eval.py
```

Evaluation results include:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## Model Architecture

### SE Attention Module

```
Input (C×H×W)
    ↓
Global Average Pooling → C
    ↓
FC (C → C/r) → ReLU
    ↓
FC (C/r → C) → Sigmoid
    ↓
Channel Weights (C×1×1)
    ↓
Scale: Input × Weights
    ↓
Output (C×H×W)
```

### LeNet-5 + SE Architecture

```
Input (1×32×32)
    ↓
Conv1: 1→6, 5×5, padding=2
    ↓
ReLU + AvgPool (2×2)
    ↓
SE Block 1 (6 channels, reduction=2)
    ↓
Conv2: 6→16, 5×5
    ↓
ReLU + AvgPool (2×2)
    ↓
SE Block 2 (16 channels, reduction=4)
    ↓
Flatten (16×5×5)
    ↓
FC: 400 → 120 + Dropout(0.3)
    ↓
FC: 120 → 84 + Dropout(0.3)
    ↓
FC: 84 → 249 (num_classes)
    ↓
Output (249)
```

---

## SE Attention Optimization

Special optimizations for handwritten math problems:

1. **Enhance Key Features**:
   - Error symbol recognition (e.g., +/-, ×/÷ confusion)
   - Calculation step recognition (key calculation nodes)
   - Number structure recognition (prevent 0/6, 1/7 confusion)

2. **Suppress Irrelevant Features**:
   - Illegible handwriting strokes
   - Uneven writing ink
   - Paper background noise

3. **Channel Weight Learning**:
   - SE module automatically learns which channels are more important for classification
   - Enhance channel weights related to error symbols
   - Suppress channel weights for background noise

---

## Dependencies

- Python 3.11 or 3.12 (3.11 recommended)
- PyTorch >= 2.0.0 (CUDA version)
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Albumentations >= 1.3.0
- scikit-learn >= 1.0.0
- TensorBoard >= 2.10.0
- Pillow >= 9.0.0
- pandas >= 1.5.0
- matplotlib >= 3.6.0
- tqdm >= 4.64.0

---

## GPU Acceleration

The model automatically detects and uses GPU if available:

```python
# config.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Important**: For GPU acceleration, ensure:
1. NVIDIA GPU driver is installed
2. CUDA Toolkit 12.4+ is installed
3. PyTorch CUDA version is installed (not CPU version)

Check if GPU is available:
```python
import torch
print(torch.cuda.is_available())  # True means available
print(torch.cuda.get_device_name(0))  # Shows GPU model
```

---

## FAQ

### Q: What if GPU memory is insufficient during training?

A: Reduce `BATCH_SIZE`, default is 64, try 32 or 16.

### Q: How to handle input images of different sizes?

A: Currently the model uses fixed 32×32 input. Modify `IMAGE_SIZE` in `config.py`, or use `resize_with_padding` function for preprocessing.

### Q: How to add new symbol classes?

A: Modify `NUM_CLASSES` in `config.py` and use corresponding label IDs during data preparation.

### Q: Python 3.14 cannot install PyTorch CUDA version?

A: Python 3.14 is too new, PyTorch doesn't support it yet. Use Python 3.11 or 3.12 to create virtual environment.

### Q: Execution policy error when activating virtual environment?

A: Run in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: How to use symbol level training?

A: Use `train_symbol.py` script for symbol level training:
```powershell
python train_symbol.py
```

### Q: HME100K dataset download failed?

A: Manually download the dataset and extract to `data/raw` directory, then run `prepare_hme100k.py` for processing.

---

## Quick Start Checklist

- [ ] Install Python 3.11
- [ ] Create virtual environment (`py -3.11 -m venv venv`)
- [ ] Activate virtual environment (`.\venv\Scripts\Activate.ps1`)
- [ ] Install PyTorch CUDA version
- [ ] Install other dependencies
- [ ] Verify environment (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Prepare dataset (`python prepare_hme100k.py`)
- [ ] Test model structure (`python test_model.py`)
- [ ] Start training (`python train.py`)
- [ ] Monitor training (TensorBoard)

---

## Related Paper

This project is based on the following paper:
- **Handwritten Algebra Answer Judgement for Middle School Students Based on CNN-SE Attention Mechanism**

The paper covers:
- Research background and significance
- Related technology introduction (LeNet-5, SE attention mechanism)
- System design and implementation
- Experiment and analysis
- Conclusion and future work

---

## References

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) - SE Attention Original Paper
- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - LeNet-5 Original Paper
- [CROHME Dataset](https://www.isical.ac.in/~crochme/) - Handwritten Math Expression Dataset
- [HME100K Dataset](http://www.cs.rit.edu/~dprl/HC/HME100K.html) - Large-scale Handwritten Math Expression Dataset

## License

MIT License

---

# CNN+SE 注意力机制手写数学表达式识别

基于 LeNet-5 卷积神经网络和 Squeeze-and-Excitation (SE) 注意力模块的手写代数题识别系统。

## 项目简介

本项目实现了一个轻量级的手写数学表达式识别模型，主要用于判断中学生手写代数题的答题正确性。项目特点：

- **LeNet-5 架构**：经典的卷积神经网络结构，适合小型图像分类任务
- **SE 注意力机制**：通过通道注意力强化关键特征（错误符号、计算步骤），抑制无关特征（字迹潦草笔画）
- **轻量化设计**：不做过深模型设计，保持高效推理速度
- **数据增强**：多种增强策略提升模型泛化能力

## 项目结构

```
cnn_se_math/
├── config.py                 # 配置文件
├── train.py                  # 主训练脚本（表达式级别）
├── train_symbol.py           # 符号级别训练脚本
├── prepare_hme100k.py        # HME100K 数据集准备脚本
├── prepare_symbol_dataset.py # 符号数据集准备脚本
├── test_model.py             # 模型结构测试
├── requirements.txt          # 依赖包
├── README.md                  # 项目说明
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── se_block.py           # SE注意力模块
│   └── lenet5_se.py          # LeNet-5 + SE模型
├── preprocessing/            # 数据预处理
│   ├── __init__.py
│   ├── image_processor.py     # 图像处理
│   ├── augmentation.py       # 数据增强
│   └── dataset.py            # 数据集加载
├── train/                     # 训练模块
│   ├── __init__.py
│   └── trainer.py             # 训练和验证函数
├── evaluate/                  # 评估模块
│   ├── __init__.py
│   └── eval.py               # 评估脚本
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── train_images/
│   │   ├── val_images/
│   │   └── test_images/
│   └── processed/            # 处理后数据
├── checkpoints/              # 模型检查点
└── logs/                     # 训练日志
```

---

## 环境安装（详细步骤）

### 前置要求

- Python 3.11 或 3.12（推荐 3.11，兼容性最好）
- NVIDIA 显卡（可选，用于 GPU 加速）
- CUDA 12.4 或更高版本（可选）

### 方法一：推荐 - 使用虚拟环境

#### Step 1: 创建虚拟环境

```powershell
# 进入项目目录
cd e:\Math Hand\cnn_se_math

# 使用 Python 3.11 创建虚拟环境
py -3.11 -m venv venv
```

#### Step 2: 激活虚拟环境

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

#### Step 3: 安装 PyTorch（CUDA 版本）

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: 安装其他依赖

```powershell
pip install opencv-python pandas matplotlib scikit-learn albumentations tensorboard tqdm Pillow
```

#### Step 5: 验证安装

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期输出：
```
PyTorch: 2.6.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

### 方法二：使用 requirements.txt

```powershell
# 创建虚拟环境
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# 安装所有依赖
pip install -r requirements.txt
```

**注意**：requirements.txt 中的 PyTorch 默认是 CPU 版本。如果需要 GPU 加速，请单独安装 CUDA 版本：
```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 方法三：系统级安装（不推荐）

如果不想使用虚拟环境：

```powershell
# 安装 Python 3.11
# 下载地址: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

# 安装 PyTorch CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install opencv-python pandas matplotlib scikit-learn albumentations tensorboard tqdm Pillow
```

---

## 数据准备

### 1. 数据集来源

- [HME100K](http://www.cs.rit.edu/~dprl/HC/HME100K.html) - 手写数学表达式数据集

### 2. 数据格式要求

项目要求的数据格式：

```
data/raw/
├── train.csv              # 训练集标签文件
├── val.csv                # 验证集标签文件
├── test.csv               # 测试集标签文件
├── train_images/          # 训练图像目录
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── val_images/            # 验证图像目录
│   └── ...
└── test_images/           # 测试图像目录
    └── ...
```

### 3. CSV 文件格式

CSV 文件必须包含 `filename` 和 `label` 两列：

```csv
filename,label
image001.png,0
image002.png,1
image003.png,2
...
```

**注意**：
- `filename` 为图像文件名
- `label` 为整数类别 ID（从 0 开始）
- HME100K 数据集有 249 个类别

### 4. 使用 HME100K 数据集准备脚本

```powershell
# 运行数据集准备脚本
python prepare_hme100k.py
```

脚本会自动：
1. 下载 HME100K 数据集
2. 解析表达式并提取图像
3. 生成训练、验证、测试 CSV 文件
4. 将数据组织成项目所需格式

### 5. 自定义数据集准备

如果你的图像按照类别文件夹组织，可以使用以下工具生成 CSV：

```python
from preprocessing import CSVGenerator

# 从文件夹生成 CSV
CSVGenerator.create_csv_from_folder(
    folder_path='path/to/images',
    output_csv='data/raw/train.csv',
    file_extension='.png'
)

# 分割数据集
CSVGenerator.split_dataset(
    csv_file='path/to/all_data.csv',
    output_dir='data/raw',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

---

## 训练模型

### 基本训练（表达式级别）

```powershell
# 确保虚拟环境已激活
.\venv\Scripts\Activate.ps1

# 运行训练
python train.py
```

### 符号级别训练

如果需要训练符号级别的识别模型：

```powershell
python train_symbol.py
```

### 自定义配置

在 `config.py` 中修改配置参数：

```python
# 训练超参数
BATCH_SIZE = 64           # 批次大小
LEARNING_RATE = 0.001     # 学习率
NUM_EPOCHS = 50           # 训练轮数
DROPOUT_RATE = 0.3        # Dropout比率

# 数据集参数
NUM_CLASSES = 249         # 类别数量
IMAGE_SIZE = 32           # 图像尺寸

# 早停设置
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
```

### 监控训练

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开 http://localhost:6006

---

## 测试模型结构

在安装依赖后，可以测试模型结构是否正确：

```bash
python test_model.py
```

预期输出：
```
====================================================
 CNN+SE MODEL STRUCTURE TEST SUITE
====================================================

==================================================
Testing SE Block...
==================================================
  Input shape: torch.Size([2, 6, 32, 32])
  Output shape: torch.Size([2, 6, 32, 32])
  Channels preserved: ✓
SE Block test: PASSED

...

✓ All tests passed! Model structure is correct.

Model can now be trained with: python train.py
```

---

## 评估模型

训练完成后，评估模型性能：

```bash
python evaluate/eval.py
```

评估结果包括：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数 (F1 Score)
- 分类报告
- 混淆矩阵

---

## 模型架构

### SE 注意力模块

```
Input (C×H×W)
    ↓
Global Average Pooling → C
    ↓
FC (C → C/r) → ReLU
    ↓
FC (C/r → C) → Sigmoid
    ↓
Channel Weights (C×1×1)
    ↓
Scale: Input × Weights
    ↓
Output (C×H×W)
```

### LeNet-5 + SE 架构

```
Input (1×32×32)
    ↓
Conv1: 1→6, 5×5, padding=2
    ↓
ReLU + AvgPool (2×2)
    ↓
SE Block 1 (6通道, reduction=2)
    ↓
Conv2: 6→16, 5×5
    ↓
ReLU + AvgPool (2×2)
    ↓
SE Block 2 (16通道, reduction=4)
    ↓
Flatten (16×5×5)
    ↓
FC: 400 → 120 + Dropout(0.3)
    ↓
FC: 120 → 84 + Dropout(0.3)
    ↓
FC: 84 → 249 (类别数)
    ↓
Output (249)
```

---

## SE 注意力优化点

针对手写数学题的特殊优化：

1. **强化关键特征**：
   - 错误符号识别（如 +/-, ×/÷ 混淆）
   - 计算步骤识别（关键计算节点）
   - 数字结构识别（防止 0/6, 1/7 混淆）

2. **抑制无关特征**：
   - 字迹潦草笔画
   - 书写墨迹不均匀
   - 纸张背景噪声

3. **通道权重学习**：
   - SE 模块自动学习哪些通道对分类更重要
   - 错误符号相关通道权重增强
   - 背景噪声通道权重抑制

---

## 依赖环境

- Python 3.11 或 3.12（推荐 3.11）
- PyTorch >= 2.0.0 (CUDA 版本)
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Albumentations >= 1.3.0
- scikit-learn >= 1.0.0
- TensorBoard >= 2.10.0
- Pillow >= 9.0.0
- pandas >= 1.5.0
- matplotlib >= 3.6.0
- tqdm >= 4.64.0

---

## GPU 加速

模型会自动检测并使用 GPU（如果有）：

```python
# config.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**重要**：如果需要 GPU 加速，请确保：
1. 已安装 NVIDIA 显卡驱动
2. 已安装 CUDA Toolkit 12.4+
3. 已安装 PyTorch CUDA 版本（不是 CPU 版本）

检查 GPU 是否可用：
```python
import torch
print(torch.cuda.is_available())  # True 表示可用
print(torch.cuda.get_device_name(0))  # 显示 GPU 型号
```

---

## 常见问题

### Q: 训练时显存不足怎么办？

A: 减小 `BATCH_SIZE`，当前默认 64，可尝试 32 或 16。

### Q: 如何处理不同尺寸的输入图像？

A: 当前模型固定输入 32×32，可在 `config.py` 中修改 `IMAGE_SIZE`，或使用 `resize_with_padding` 函数预处理。

### Q: 如何添加新的符号类别？

A: 修改 `config.py` 中的 `NUM_CLASSES`，并在数据准备时使用对应的标签 ID。

### Q: Python 3.14 无法安装 PyTorch CUDA 版本？

A: Python 3.14 太新，PyTorch 尚未支持。请使用 Python 3.11 或 3.12 创建虚拟环境。

### Q: 激活虚拟环境时遇到执行策略错误？

A: 在 PowerShell 中运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: 如何使用符号级别训练？

A: 使用 `train_symbol.py` 脚本进行符号级别训练：
```powershell
python train_symbol.py
```

### Q: HME100K 数据集下载失败怎么办？

A: 手动下载数据集并解压到 `data/raw` 目录，然后运行 `prepare_hme100k.py` 进行处理。

---

## 快速开始清单

- [ ] 安装 Python 3.11
- [ ] 创建虚拟环境 (`py -3.11 -m venv venv`)
- [ ] 激活虚拟环境 (`.\venv\Scripts\Activate.ps1`)
- [ ] 安装 PyTorch CUDA 版本
- [ ] 安装其他依赖
- [ ] 验证环境 (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] 准备数据集 (`python prepare_hme100k.py`)
- [ ] 测试模型结构 (`python test_model.py`)
- [ ] 开始训练 (`python train.py`)
- [ ] 监控训练 (TensorBoard)

---

## 项目论文

本项目基于以下论文：
- **基于CNN_SE注意力机制的中学生代数题手写答题判断**

论文详细说明了：
- 研究背景与意义
- 相关技术介绍（LeNet-5、SE注意力机制）
- 系统设计与实现
- 实验与分析
- 总结与展望

---

## 参考资料

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) - SE 注意力机制原论文
- [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - LeNet-5 原论文
- [CROHME Dataset](https://www.isical.ac.in/~crochme/) - 手写数学表达式数据集
- [HME100K Dataset](http://www.cs.rit.edu/~dprl/HC/HME100K.html) - 大规模手写数学表达式数据集

## License

MIT License
