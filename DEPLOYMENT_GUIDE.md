# LeapHand Planner V3 - 部署指南

## 项目概述

LeapHand Planner V3是一个先进的轨迹规划系统，专为LeapHand机械手设计。该系统使用多模态注意力融合、时间Transformer解码器和不确定性估计等先进技术。

## 性能指标

### 训练结果
- **MAE**: 0.0086
- **RMSE**: 0.0111
- **安全评分**: 100%
- **训练轮数**: 78 epochs
- **最终损失**: 0.0004

### 推理性能
- **PyTorch CUDA**: 185.2 FPS (5.4ms)
- **ONNX CUDA**: 212.8 FPS (4.7ms)
- **安全违规率**: 0%

## 部署选项

### 1. PyTorch推理 (推荐用于开发)
```bash
python inference_v3.py --model_path runs/leap_hand_v3/best_model.pth --demo
```

### 2. 优化ONNX推理 (推荐用于生产)
```bash
python optimized_inference.py --model_path runs/leap_hand_v3/best_model.pth --use_onnx --device cuda
```

### 3. 批量推理
```bash
python optimized_inference.py --performance_test 1000 --optimize_for latency
```

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA 11.8+ (推荐RTX 30系列或更高)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 软件要求
- **Python**: 3.8-3.11
- **PyTorch**: 2.0+
- **CUDA**: 11.8+
- **ONNX Runtime**: 1.15+

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd leaphandgen3
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install onnxruntime-gpu onnx  # 用于ONNX优化
```

### 3. 下载训练好的模型
```bash
# 模型文件位置: runs/leap_hand_v3/best_model.pth
# ONNX文件会自动生成: runs/leap_hand_v3/best_model.onnx
```

## API使用

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
print(f"推理时间: {info['inference_time']:.4f}s")
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

### 优化推理
```python
from optimized_inference import OptimizedLeapHandInference

# 使用ONNX加速
inference = OptimizedLeapHandInference(
    'runs/leap_hand_v3/best_model.pth',
    use_onnx=True,
    device='cuda',
    optimize_for='latency'
)

trajectory, info = inference.infer_trajectory(pose, point_cloud, tactile)
```

## 配置选项

### 模型配置 (leap_hand_planner_v3/config/default.yaml)
```yaml
model:
  action_dim: 63          # LeapHand动作维度
  pose_dim: 3            # 物体姿态维度
  pc_dim: 6144           # 点云维度
  tactile_dim: 100       # 触觉维度
  seq_len: 10            # 轨迹序列长度
  hidden_dim: 512        # 隐藏层维度
  num_heads: 8           # 注意力头数
  num_layers: 4          # Transformer层数

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  use_ema: true          # 使用EMA模型

inference:
  use_postprocessing: true
  use_safety_check: true
  adaptive_safety: true
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用更小的模型 (减少hidden_dim)

2. **推理速度慢**
   - 使用ONNX优化版本
   - 启用optimize_for='latency'

3. **模型加载失败**
   - 检查模型文件路径
   - 确保PyTorch版本兼容

### 性能优化建议

1. **内存优化**
   - 使用梯度累积
   - 启用混合精度训练

2. **速度优化**
   - 使用ONNX Runtime
   - 启用TensorRT (如果可用)

3. **部署优化**
   - 模型量化 (8-bit)
   - 模型剪枝
   - 使用边缘设备优化

## 技术规格

### 模型架构
- **多模态注意力融合**: 融合姿态、点云、触觉信息
- **时间Transformer解码器**: 自回归轨迹生成
- **不确定性估计**: 提供预测置信度
- **安全检查**: 实时约束验证

### 数据格式
- **轨迹**: [序列长度, 63] - LeapHand关节角度
- **姿态**: [3] - 物体位置 (x, y, z)
- **点云**: [6144] - 展平的点云数据
- **触觉**: [100] - 触觉传感器读数

### 安全约束
- 关节角度限制: [-π/2, π/2]
- 速度限制: 3.0 rad/s
- 加速度限制: 8.0 rad/s²

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交问题和改进建议！

## 联系方式

项目维护者: [您的联系方式]