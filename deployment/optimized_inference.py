"""Optimized inference script for LeapHand Planner V3 with ONNX acceleration."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
import onnxruntime as ort
from leap_hand_planner_v3.models.planner_v3 import LeapHandPlannerV3
from leap_hand_planner_v3.config import config


class OptimizedLeapHandInference:
    """Optimized inference engine using ONNX Runtime for acceleration."""

    def __init__(
        self,
        model_path: str,
        config_path: str = 'leap_hand_planner_v3/config/default.yaml',
        device: str = 'cuda',
        use_onnx: bool = True,
        optimize_for: str = 'latency'  # 'latency' or 'throughput'
    ):
        self.device = device
        self.use_onnx = use_onnx
        self.optimize_for = optimize_for

        # Load configuration
        if config_path != 'leap_hand_planner_v3/config/default.yaml':
            config.__init__(config_path)

        if use_onnx:
            self.session = self._load_onnx_model(model_path)
        else:
            self.model = self._load_pytorch_model(model_path)
            self.model.eval()

        # Performance tracking
        self.inference_times = []

    def _load_pytorch_model(self, model_path: str) -> nn.Module:
        """Load PyTorch model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = LeapHandPlannerV3(
            action_dim=config.get('model.action_dim', 63),
            pose_dim=config.get('model.pose_dim', 3),
            pc_dim=config.get('model.pc_dim', 6144),
            tactile_dim=config.get('model.tactile_dim', 100),
            seq_len=config.get('model.seq_len', 10),
            hidden_dim=config.get('model.hidden_dim', 512),
            num_heads=config.get('model.num_heads', 8),
            num_layers=config.get('model.num_layers', 4)
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _load_onnx_model(self, model_path: str):
        """Load ONNX model with optimized settings."""
        # Convert .pth to .onnx if needed
        onnx_path = model_path.replace('.pth', '.onnx')
        if not Path(onnx_path).exists():
            print("Converting PyTorch model to ONNX...")
            self._convert_to_onnx(model_path, onnx_path)

        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()

        # Enable optimization based on target
        if self.optimize_for == 'latency':
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:  # throughput
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 4
            sess_options.intra_op_num_threads = 4

        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        return session

    def _convert_to_onnx(self, pytorch_path: str, onnx_path: str):
        """Convert PyTorch model to ONNX format."""
        # Load PyTorch model
        pytorch_model = self._load_pytorch_model(pytorch_path)

        # Create dummy input
        dummy_pose = torch.randn(1, 3).to(self.device)
        dummy_pc = torch.randn(1, 6144).to(self.device)
        dummy_tactile = torch.randn(1, 100).to(self.device)

        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            (dummy_pose, dummy_pc, dummy_tactile),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['pose', 'point_cloud', 'tactile'],
            output_names=['trajectory'],
            dynamic_axes={
                'pose': {0: 'batch_size'},
                'point_cloud': {0: 'batch_size'},
                'tactile': {0: 'batch_size'},
                'trajectory': {0: 'batch_size'}
            }
        )

        print(f"ONNX model saved to {onnx_path}")

    def infer_trajectory(
        self,
        pose: np.ndarray,
        point_cloud: np.ndarray,
        tactile: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform optimized trajectory inference.

        Args:
            pose: Object pose [3]
            point_cloud: Point cloud [6144]
            tactile: Tactile readings [100]

        Returns:
            trajectory: Predicted trajectory [seq_len, action_dim]
            info: Inference information
        """
        start_time = time.time()

        # Prepare inputs
        pose_tensor = np.asarray(pose).astype(np.float32).reshape(1, -1)
        pc_tensor = np.asarray(point_cloud).astype(np.float32).reshape(1, -1)
        tactile_tensor = np.asarray(tactile).astype(np.float32).reshape(1, -1)

        if self.use_onnx:
            # ONNX inference
            ort_inputs = {
                'pose': pose_tensor,
                'point_cloud': pc_tensor,
                'tactile': tactile_tensor
            }
            ort_outputs = self.session.run(None, ort_inputs)
            trajectory = ort_outputs[0][0]  # Remove batch dimension
        else:
            # PyTorch inference
            pose_t = torch.from_numpy(pose_tensor).to(self.device)
            pc_t = torch.from_numpy(pc_tensor).to(self.device)
            tactile_t = torch.from_numpy(tactile_tensor).to(self.device)

            with torch.no_grad():
                trajectory = self.model(pose_t, pc_t, tactile_t).squeeze(0).cpu().numpy()

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        info = {
            'inference_time': inference_time,
            'backend': 'onnx' if self.use_onnx else 'pytorch',
            'optimized_for': self.optimize_for
        }

        return trajectory, info

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)

        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'median_inference_time': np.median(times),
            'total_inferences': len(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0.0,
            'backend': 'onnx' if self.use_onnx else 'pytorch'
        }


def benchmark_inference():
    """Benchmark different inference backends."""
    print("Benchmarking LeapHand Planner V3 Inference Backends")
    print("=" * 60)

    model_path = 'runs/leap_hand_v3/best_model.pth'

    # Test data
    pose = np.random.randn(3).astype(np.float32)
    point_cloud = np.random.randn(6144).astype(np.float32)
    tactile = np.random.randn(100).astype(np.float32)

    backends = [
        ('PyTorch CPU', OptimizedLeapHandInference(model_path, device='cpu', use_onnx=False)),
        ('PyTorch CUDA', OptimizedLeapHandInference(model_path, device='cuda', use_onnx=False)),
        ('ONNX CPU', OptimizedLeapHandInference(model_path, device='cpu', use_onnx=True)),
        ('ONNX CUDA', OptimizedLeapHandInference(model_path, device='cuda', use_onnx=True)),
        ('ONNX CUDA Latency', OptimizedLeapHandInference(model_path, device='cuda', use_onnx=True, optimize_for='latency')),
        ('ONNX CUDA Throughput', OptimizedLeapHandInference(model_path, device='cuda', use_onnx=True, optimize_for='throughput')),
    ]

    results = []

    for name, engine in backends:
        print(f"\nTesting {name}...")

        try:
            # Warm up
            for _ in range(5):
                engine.infer_trajectory(pose, point_cloud, tactile)

            # Benchmark
            num_tests = 100
            for _ in range(num_tests):
                trajectory, info = engine.infer_trajectory(pose, point_cloud, tactile)

            stats = engine.get_performance_stats()

            print(".4f")
            print(".1f")
            print(f"  Trajectory shape: {trajectory.shape}")

            results.append((name, stats))

        except Exception as e:
            print(f"  Error: {e}")
            results.append((name, None))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, stats in results:
        if stats:
            print("30")
        else:
            print("30")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized LeapHand Planner V3 Inference')
    parser.add_argument('--model_path', type=str, default='runs/leap_hand_v3/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='leap_hand_planner_v3/config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--use_onnx', action='store_true', default=True,
                       help='Use ONNX for inference')
    parser.add_argument('--optimize_for', type=str, default='latency', choices=['latency', 'throughput'],
                       help='Optimization target')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run backend benchmark comparison')
    parser.add_argument('--performance_test', type=int, default=0,
                       help='Run performance test with N inferences')

    args = parser.parse_args()

    if args.benchmark:
        benchmark_inference()
    elif args.performance_test > 0:
        print(f"Running performance test with {args.performance_test} inferences...")

        engine = OptimizedLeapHandInference(
            args.model_path, args.config, args.device,
            args.use_onnx, args.optimize_for
        )

        # Test data
        pose = np.random.randn(3).astype(np.float32)
        point_cloud = np.random.randn(6144).astype(np.float32)
        tactile = np.random.randn(100).astype(np.float32)

        for _ in range(args.performance_test):
            trajectory, info = engine.infer_trajectory(pose, point_cloud, tactile)

        stats = engine.get_performance_stats()
        print("\nPerformance Statistics:")
        print(".4f")
        print(".4f")
        print(".1f")
    else:
        print("Use --benchmark or --performance_test to run tests")
        print("Example: python optimized_inference.py --benchmark")