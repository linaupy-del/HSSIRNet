# HSSIRNet

## 中文说明

### 1. 项目简介

本仓库提供 HSSIRNet 的三阶段训练入口、网络结构、损失函数及数据读取代码，用于包含脑肿瘤的脑 MRI 图像配准研究。

当前代码涉及：

- **第一阶段：** 基于图像相似性和组织标签约束的配准。
- **第二阶段：** 联合配准与图像合成，在健康区域计算相关监督项。
- **第三阶段：** 在线图谱配准、病灶区域修复及最终配准。

### 2. 文件结构

```text
HSSIRNet/
├── Networks/
│   ├── only_reg_net_exe2.py      # 第一阶段网络
│   ├── pre_reg_syn_net_ex2.py    # 第二阶段网络
│   └── inpainting_reg_syn_net.py # 第三阶段网络
├── generators_ex2_1.py          # 第一、二阶段数据读取
├── generators4.py               # 第三阶段数据读取
├── losses.py                   # 损失函数
├── step1.py                    # 第一阶段训练入口
├── step2.py                    # 第二阶段训练入口
├── step3.py                    # 第三阶段训练入口
├── train_test_manifest.txt      # 数据文件清单与配对说明
├── write_excel.py              # Excel读写辅助函数
└── README.md
```

### 3. 运行依赖

代码使用 Python 和 PyTorch，直接导入的第三方依赖包括：

- PyTorch
- torchvision
- NumPy
- SimpleITK
- einops
- xlrd
- xlwt
- xlutils
- ...

### 4. 数据说明

根据 `train_test_manifest.txt`：

| 用途 | 固定图像 | 移动图像 | 数量 |
|---|---|---|---|
| 训练 | BraTS2020 | OASIS-1 | 366(365+1)个固定/移动图像配对 |
| Table 1：个体间配准测试 | BraTS2020 | OASIS-1 | 50 × 50，共2500对 |
| Table 2：个体内配准测试 | BraTS-Reg基线图像 | BraTS-Reg随访图像 | 140对 |

每个训练图像池包含365个非图谱图像条目及1个图谱副本(MNI152)。

BraTS-Reg 在本研究的数据清单中仅用于测试。

---

## English Documentation

### 1. Overview

This repository provides the three-stage training scripts, network architectures, loss functions, and data loaders for HSSIRNet, supporting research on brain MRI registration involving tumor-bearing images.

The implementation includes:

- **Stage 1:** Registration with image-similarity and tissue-label constraints.
- **Stage 2:** Joint registration and image synthesis, with relevant supervision terms evaluated in healthy regions.
- **Stage 3:** Online atlas registration, lesion-region inpainting, and final registration.

### 2. Repository Structure

```text
HSSIRNet/
├── Networks/
│   ├── only_reg_net_exe2.py      # Stage 1 network
│   ├── pre_reg_syn_net_ex2.py    # Stage 2 network
│   └── inpainting_reg_syn_net.py # Stage 3 network
├── generators_ex2_1.py          # Data loading for Stages 1 and 2
├── generators4.py               # Data loading for Stage 3
├── losses.py                   # Loss functions
├── step1.py                    # Stage 1 training script
├── step2.py                    # Stage 2 training script
├── step3.py                    # Stage 3 training script
├── train_test_manifest.txt      # Image manifest and pairing definitions
├── write_excel.py              # Excel utilities
└── README.md
```

### 3. Dependencies

The code uses Python and PyTorch. Third-party packages directly imported by the source include:

- PyTorch
- torchvision
- NumPy
- SimpleITK
- einops
- xlrd
- xlwt
- xlutils
- ...

### 4. Data

According to `train_test_manifest.txt`:

| Purpose | Fixed images | Moving images | Number of pairs |
|---|---|---|---|
| Training | BraTS2020 | OASIS-1 | 366(365+1) fixed/moving image pairs |
| Table 1: Inter-subject registration testing | BraTS2020 | OASIS-1 | 50 × 50 = 2500 pairs |
| Table 2: Intra-subject registration testing | BraTS-Reg baseline images | BraTS-Reg follow-up images | 140 pairs |

Each training image pool contains 365 non-atlas image entries and one atlas(MNI152) copy.

BraTS-Reg is used exclusively for testing in the study manifest.

