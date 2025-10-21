# LeapHand Planner V1: U-Net Diffusion Model

## 概述

LeapHand Planner V1是基于U-Net扩散模型的轨迹规划系统，使用条件扩散过程生成机械手轨迹。

## 核心特性

- **U-Net扩散架构**: 基于Conv1D的U-Net结构进行去噪
- **条件生成**: 支持姿态、点云、触觉等多模态条件输入
- **MPC优化**: 集成模型预测控制进行轨迹优化
- **简单高效**: 轻量级架构，适合快速原型开发

## 架构

```python
DiffusionPlanner(
    cond_dim=6144+100+3,  # 点云 + 触觉 + 姿态
    seq_len=63,           # LeapHand关节维度
    in_channels=1,        # 单通道输入
    hidden_dim=256        # 隐藏层维度
)
```

## 使用方法

```python
from leap_hand_planner_v1 import DiffusionPlanner

# 创建模型
model = DiffusionPlanner()

# 训练
# python leap_hand_planner_v1/train.py

# 推理
trajectory = model.generate_trajectory(condition)
```

## 性能指标

- **架构复杂度**: ~50K参数
- **推理速度**: ~100ms/轨迹
- **训练时间**: ~2-4小时 (1000 epochs)

## 文件结构

```
leap_hand_planner_v1/
├── __init__.py
├── diffusion_planner.py    # 核心扩散模型
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── inference.py           # 推理脚本
└── diffusion.md           # 技术文档
```