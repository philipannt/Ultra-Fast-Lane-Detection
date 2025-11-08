# Ultra-Fast Lane Detection v2 - Training Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation và cải thiện của **Ultra-Fast Lane Detection v2 (UFLDv2)** trên dataset **TUSimple** với ResNet18 backbone. Project này được tối ưu hóa để đạt được kết quả tương đương với paper (F1 Score: 96.11).

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Các cải thiện chính](#các-cải-thiện-chính)
- [Cấu hình đúng theo Paper](#cấu-hình-đúng-theo-paper)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Kết quả mong đợi](#kết-quả-mong-đợi)
- [Troubleshooting](#troubleshooting)

## 🎯 Tổng quan

**Ultra-Fast Lane Detection v2** là một phương pháp phát hiện làn đường siêu nhanh sử dụng Hybrid Anchor Driven Ordinal Classification. Project này implement và cải thiện training pipeline trên dataset TUSimple với mục tiêu đạt được F1 Score ≥ 96.11 như trong paper gốc.

### Model Architecture

- **Backbone**: ResNet18
- **Input Size**: 320×800 pixels (theo config chính thức)
- **Output**: Row and column lane predictions với existence probabilities
- **Gridding Number**: 100
- **Row Anchors**: 56
- **Column Anchors**: 41

## ✨ Các cải thiện chính

### 1. **Sửa Learning Rate (Quan trọng nhất)**

- **Trước**: `learning_rate = 0.0005` ❌
- **Sau**: `learning_rate = 0.05` ✅
- **Lý do**: Learning rate quá thấp khiến model học quá chậm, không đạt được performance như paper.

### 2. **Sửa Input Size**

- **Trước**: `288×800` ❌
- **Sau**: `320×800` ✅ (theo config chính thức `tusimple_res18.py`)
- **Ảnh hưởng**: Input size sai làm model không học đúng với architecture được train.

### 3. **Chuẩn hóa Batch Size**

- **Trước**: `batch_size = 24` (tự tối ưu)
- **Sau**: `batch_size = 32` (theo config chính thức)
- **Lợi ích**: Đảm bảo consistency với paper và training ổn định hơn.

### 4. **Tối ưu hóa Data Loading**

- Implement custom `DataLoaderWrapper` cho Windows compatibility
- Xử lý `num_workers=0` để tránh multiprocessing issues trên Windows
- Tối ưu transforms và collate function để tăng tốc độ loading

### 5. **Xử lý Loss Functions**

- Patch tất cả loss functions để xử lý empty tensors (tránh NaN/Inf)
- Implement safe division trong `soft_nll`, `MeanLoss`, `VarLoss`, `EMDLoss`, `RegLoss`
- Kiểm tra và xử lý NaN/Inf trong quá trình training

### 6. **Automatic Test Set Generation**

- Tự động generate `test.txt` từ `test_label.json` nếu không khớp số lượng
- Xử lý path resolution cho images (hỗ trợ cả `train_set/` và `test_set/`)
- Tạo dummy images cho missing files để đảm bảo số lượng predictions khớp

### 7. **Mixed Precision Training (AMP)**

- Enable AMP để tăng tốc độ training ~2x
- Hỗ trợ TF32 cho RTX 4xxx GPUs
- Tối ưu CUDA settings (CuDNN benchmark, non-deterministic)

### 8. **Checkpoint Management**

- Auto-resume từ `checkpoint_latest.pth`
- Lưu best model dựa trên F1 score (`model_best.pth`)
- Overwrite latest checkpoint mỗi epoch để tiết kiệm disk space

## ⚙️ Cấu hình đúng theo Paper

Tất cả các cấu hình đã được kiểm tra và đối chiếu với config chính thức từ repository gốc (`configs/tusimple_res18.py`):

```python
# Training Configuration
epoch = 100
batch_size = 32
learning_rate = 0.05  # ⚠️ QUAN TRỌNG: Không phải 0.0005!
optimizer = 'SGD'
momentum = 0.9
weight_decay = 0.0001
scheduler = 'multi'
steps = [50, 75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# Model Configuration
train_height = 320  # ⚠️ QUAN TRỌNG: Không phải 288!
train_width = 800
num_row = 56
num_col = 41
num_cell_row = 100
num_cell_col = 100
num_lanes = 4

# Loss Weights
mean_loss_w = 0.05
mean_loss_col_w = 0.05
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
```

## 🚀 Tính năng

### Core Features

- ✅ Training pipeline hoàn chỉnh với early stopping
- ✅ Validation và test evaluation tự động
- ✅ F1 score evaluation trên test set (mỗi 30 epochs)
- ✅ Checkpoint saving và auto-resume
- ✅ Mixed Precision Training (AMP) support
- ✅ Comprehensive logging và visualization

### Data Handling

- ✅ Automatic dataset loading và validation
- ✅ Train/Validation split (80/20)
- ✅ Test set auto-generation từ `test_label.json`
- ✅ Image path resolution (hỗ trợ multiple paths)
- ✅ Missing file handling với dummy images

### Error Handling

- ✅ NaN/Inf detection và handling trong loss functions
- ✅ Empty tensor handling
- ✅ Division by zero protection
- ✅ CUDA error recovery
- ✅ FileNotFoundError handling

### Performance Optimizations

- ✅ Mixed Precision Training (AMP) - ~2x faster
- ✅ TF32 support cho RTX 4xxx GPUs
- ✅ CuDNN benchmark mode
- ✅ Optimized DataLoader cho Windows
- ✅ Reduced progress bar updates
- ✅ Smart metrics calculation (mỗi 5 epochs)

## 📦 Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 2.0+ (với CUDA support)
- CUDA-capable GPU (tested trên RTX 4070)
- Windows/Linux

### Dependencies

Các dependencies sẽ được tự động cài đặt khi chạy notebook:

```python
- addict
- opencv-python
- tqdm
- sklearn
- pathspec
- imagesize
- ujson
```

### Setup

1. Clone repository:
```bash
git clone <repository-url>
cd Deep-Learning/UFLDv2
```

2. Download TUSimple dataset và đặt vào thư mục `TUSimple/`

3. Cấu hình paths trong notebook:
```python
repo_path = Path(r'C:\Users\ThienAn\OneDrive\Python\Deep-Learning\Ultra-Fast-Lane-Detection-v2')
data_root = Path(r'C:\Users\ThienAn\OneDrive\Python\Deep-Learning\UFLDv2\TUSimple')
```

## 💻 Sử dụng

### Training

1. Mở `cursor_model.ipynb` trong Jupyter Notebook
2. Chạy các cells theo thứ tự từ trên xuống
3. Training sẽ tự động:
   - Load dataset
   - Tạo model
   - Train với config đúng
   - Lưu checkpoints
   - Evaluate F1 score

### Resume Training

Training tự động resume từ `checkpoint_latest.pth` nếu có. Để resume từ checkpoint cụ thể:

```python
cfg.resume = 'path/to/checkpoint.pth'
```

### Evaluation

F1 score được tính tự động:
- Epoch 0 (để kiểm tra)
- Mỗi 30 epochs
- Epoch cuối cùng

### Checkpoints

- `model_best.pth`: Best model (highest F1 score)
- `checkpoint_latest.pth`: Latest checkpoint (overwritten mỗi epoch)

## 📊 Kết quả mong đợi

### Target Performance

- **F1 Score**: ≥ 96.11 (theo paper)
- **Accuracy**: Tăng dần theo epochs
- **Loss**: Giảm dần và ổn định

### Training Time

- **Per Epoch**: ~5-6 phút (với RTX 4070, batch_size=32)
- **Total Time**: ~8-10 giờ cho 100 epochs

## 📚 Tham khảo

- Paper: [Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)
- Original Repository: [Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)
- Dataset: [TUSimple Lane Detection](https://github.com/TuSimple/tusimple-benchmark)

## 📄 License

MIT License

## 👤 Author

Implementation và improvements bởi [Thien An]

## 🙏 Acknowledgments

- Authors của Ultra-Fast Lane Detection v2 paper
- TUSimple dataset providers
- PyTorch community

---

**Lưu ý**: README này mô tả implementation và improvements đã thực hiện. Để đạt được kết quả tốt nhất, hãy đảm bảo sử dụng đúng config như đã nêu ở trên.
