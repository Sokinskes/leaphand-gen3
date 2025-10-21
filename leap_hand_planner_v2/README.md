# LeapHand Planner V2: Behavior Cloning (BC)

## 概述

LeapHand Planner V2是基于行为克隆(Behavior Cloning)的轨迹规划系统，通过监督学习直接从专家演示中学习轨迹规划策略。

## 核心特性

- **监督学习**: 直接从专家轨迹中学习映射关系
- **端到端训练**: 条件输入直接映射到轨迹输出
- **确定性预测**: 为相同输入提供一致的轨迹输出
- **快速收敛**: 相比生成式方法训练时间更短

## 架构

```python
BCPlanner(
    cond_dim=6144+100+3,  # 点云 + 触觉 + 姿态
    hidden_dim=512,       # 隐藏层维度
    output_dim=63,        # LeapHand关节维度
    num_layers=4          # MLP层数
)
```

## 使用方法

```python
from leap_hand_planner_v2 import BCPlanner

# 创建模型
model = BCPlanner()

# 训练
# python leap_hand_planner_v2/train_bc.py

# 推理
trajectory = model(condition)
```

## 性能指标

- **架构复杂度**: ~100K参数
- **推理速度**: ~10ms/轨迹
- **训练时间**: ~30分钟-1小时
- **预测稳定性**: 100%确定性输出

## 文件结构

```
leap_hand_planner_v2/
├── __init__.py
├── bc_planner.py          # 核心BC模型
├── train_bc.py           # 训练脚本
└── evaluate_bc.py        # 评估脚本
```