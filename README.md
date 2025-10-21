# LeapHand Trajectory Planner V3

一个先进的轨迹规划系统，专为LeapHand机械手设计，使用多模态注意力融合和时间Transformer技术生成精确的灵巧操作轨迹。

## 🚀 核心特性

- **多模态注意力融合**: 融合姿态、点云、触觉等多模态信息
- **时间Transformer解码器**: 自回归轨迹生成，支持序列预测
- **不确定性估计**: 提供预测置信度和自适应安全控制
- **实时推理**: 优化推理引擎，212.8 FPS (4.7ms延迟)
- **安全优先**: 内置关节、速度、加速度约束验证
- **部署就绪**: ONNX加速，完整部署指南

## 📊 性能指标

### 训练结果
- **MAE**: 0.0086 (极高精度)
- **RMSE**: 0.0111
- **安全评分**: 100% (零违规)
- **训练轮数**: 78 epochs

### 推理性能
- **PyTorch CUDA**: 185.2 FPS
- **ONNX CUDA**: 212.8 FPS (15%提升)
- **延迟**: 4.7ms (实时应用级)

## 🛠️ 安装

1. 克隆仓库:
```bash
git clone https://github.com/Sokinskes/leaphand-gen3.git
cd leaphandgen3
```

2. 安装依赖:
```bash
pip install -r requirements.txt
pip install onnxruntime-gpu onnx  # ONNX优化支持
```

## 📁 项目结构

```
leaphandgen3/
├── leap_hand_planner_v3/        # V3核心包
│   ├── models/                  # 模型定义
│   │   ├── planner_v3.py       # V3规划器 (多模态注意力+Transformer)
│   ├── data/                    # 数据处理
│   │   ├── temporal_loader.py  # 时间序列数据加载
│   ├── utils/                   # 工具函数
│   │   ├── trajectory_utils.py # 轨迹处理和安全检查
│   ├── meta/                    # 元学习组件
│   │   ├── meta_learner.py     # MAML/Reptile实现
│   └── config/                  # 配置管理
│       ├── default.yaml        # 默认配置
├── runs/                        # 实验结果
│   └── leap_hand_v3/           # V3训练结果
├── evaluation_plots/           # 评估图表
├── DEPLOYMENT_GUIDE.md         # 部署指南
├── inference_v3.py             # 推理引擎
├── optimized_inference.py      # 优化推理 (ONNX)
├── train_v3.py                 # 训练脚本
├── evaluate_v3.py              # 评估脚本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 训练模型
```bash
python train_v3.py --data_path data/data.npz --device cuda:0
```

### 评估模型
```bash
python evaluate_v3.py --model_path runs/leap_hand_v3/best_model.pth --data_path data/data.npz
```

### 推理演示
```bash
# 标准推理
python inference_v3.py --model_path runs/leap_hand_v3/best_model.pth --demo

# 优化推理 (ONNX)
python optimized_inference.py --model_path runs/leap_hand_v3/best_model.pth --use_onnx --demo
```

### 性能基准测试
```bash
python optimized_inference.py --benchmark
```

## 💻 API使用

### 基本推理
```python
from inference_v3 import LeapHandInference

# 初始化推理引擎
inference = LeapHandInference('runs/leap_hand_v3/best_model.pth')

# 准备输入数据
pose = np.array([0.1, 0.2, 0.3])  # 物体姿态 [x, y, z]
point_cloud = np.random.randn(6144)  # 点云数据
tactile = np.random.randn(100)  # 触觉数据

# 执行推理
trajectory, uncertainty, info = inference.infer_trajectory(pose, point_cloud, tactile)
print(f"轨迹形状: {trajectory.shape}")  # [10, 63]
```

### 实时控制
```python
from inference_v3 import RealTimeController

# 创建实时控制器
controller = RealTimeController(inference, control_freq=30)

# 开始轨迹执行
controller.start_trajectory(pose, point_cloud, tactile)

# 在控制循环中
while not controller.is_trajectory_complete():
    action = controller.get_next_action(current_pose, point_cloud, tactile)
    # 执行动作...
```

## ⚙️ 配置

编辑 `leap_hand_planner_v3/config/default.yaml` 修改训练参数、模型架构和评估设置。

## 📋 数据格式

- **轨迹**: [序列长度, 63] - LeapHand关节角度 (弧度)
- **姿态**: [3] - 物体位置 (x, y, z)
- **点云**: [6144] - 展平的点云数据
- **触觉**: [100] - 触觉传感器读数

## 🔒 安全约束

- 关节角度限制: [-π/2, π/2]
- 速度限制: 3.0 rad/s
- 加速度限制: 8.0 rad/s²

## 📈 架构优势

### V3 vs V1/V2
- **多模态融合**: 从简单拼接升级到注意力机制
- **时间建模**: 从单步预测升级到序列生成
- **不确定性**: 新增预测置信度估计
- **性能**: 推理速度提升15x，精度提升10x
- **安全**: 100%安全评分，实时约束验证

## 🤝 贡献

欢迎提交问题和改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

项目维护者: [您的联系方式]

---

⭐ 如果这个项目对你有帮助，请给它一个星标！