# Cross‑Subject EEG Emotion Recognition Library (X‑EER)

[English](#english-version) | [中文](#中文版本)

---

## 中文版本

### 📌 简介

一个 **简洁、自用** 的跨受试者脑电情感识别库，方便快速实验和原型验证。

- **跨受试者**：针对不同被试之间的分布差异。
- **易扩展**：清晰的模块划分，便于添加实验代码。
- **纯 PyTorch**：依赖最小化，开箱即用。

### 📁 目录结构

```text
├── datasets/       # 数据加载
├── loss_funcs/     # 损失函数实现（DANN、PairLoss 等）
├── models/         # 模型
├── trainers/       # 训练与评估逻辑
├── examples/       # 运行示例脚本
├── config.py       # 全局超参数配置
```

### 🚀 当前已实现方法

1. **DANN** 

   论文名：*Domain-Adversarial Training of Neural Networks* 

   基于 DANN 的跨受试者脑电情感识别，主要包含：`models/base`, `trainers/BaseTrainer`

   运行示例： 
   `python examples/run_base.py` 

2. **PR-PL** 

   论文名：*PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals* 

   原仓库：https://github.com/KAZABANA/PR-PL  

   对 PRPL 进行了简单的重构，主要包含：`loss_funcs/pairloss`, `models/prpl`, `trainers/PrplTrainer`

   运行示例： 
   `python examples/run_prpl.py` 

---

### ⚡ Quick Start

```bash
# 1. 克隆仓库
$ git clone https://github.com/qwangwl/TransferEEG.git && cd TransferEEG

# 2. 运行 DANN 示例
$ python examples/run_base.py  # train
```

## English Version

[中文](#中文版本)

### 📌 Introduction
A *lightweight, personal* library for **cross‑subject EEG emotion recognition**.

- **Cross‑subject focus**: tackles distribution shift across individuals.
- **Modular & extensible**: clearly separated folders for painless prototyping.
- **Pure PyTorch**: minimal dependencies, plug‑and‑play.

### 📁 Directory

```text
├── datasets/       # Data loaders
├── loss_funcs/     # Loss functions (DANN, PairLoss, ...)
├── models/         # models
├── trainers/       # Training / evaluation logic
├── examples/       # Example 
├── config.py       # Global hyper‑parameters
```

### 🚀 Currently Implemented

1. **DANN** 

   Paper: *Domain-Adversarial Training of Neural Networks* 

   Cross-subject EEG emotion recognition based on DANN, includes: `models/base`, `trainers/BaseTrainer`

   Run examples: 
   `python examples/run_base.py`  

2. **PR-PL** 

   Paper: *PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals* 

   Original repository: https://github.com/KAZABANA/PR-PL  

   We refactored PRPL slightly, covering: `loss_funcs/pairloss`, `models/prpl`, `trainers/PrplTrainer`

   Run examples: 
   `python examples/run_prpl.py`  

### ⚡ Quick Start

```bash
# 1. Clone repo
$ git clone https://github.com/qwangwl/TransferEEG.git && cd TransferEEG

# 2. Run DANN example
$ python examples/run_base.py  # train
```