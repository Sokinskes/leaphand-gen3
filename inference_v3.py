"""Inference script for LeapHand Planner V3 with real-time capabilities."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any

from leap_hand_planner_v3.models.planner_v3 import LeapHandPlannerV3
from leap_hand_planner_v3.utils.trajectory_utils import TrajectoryPostprocessor, SafetyChecker
from leap_hand_planner_v3.config import config


class LeapHandInference:
    """Real-time inference engine for LeapHand Planner V3."""

    def __init__(
        self,
        model_path: str,
        config_path: str = 'leap_hand_planner_v3/config/default.yaml',
        device: str = 'cuda'
    ):
        self.device = torch.device(device)

        # Load configuration
        if config_path != 'leap_hand_planner_v3/config/default.yaml':
            config.__init__(config_path)

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Setup post-processing and safety
        self.postprocessor = TrajectoryPostprocessor(
            action_dim=config.get('model.action_dim', 63),
            smoothing_window=config.get('postprocessing.smoothing_window', 5),
            velocity_limit=config.get('postprocessing.velocity_limit', 2.0),
            acceleration_limit=config.get('postprocessing.acceleration_limit', 5.0)
        )

        self.safety_checker = SafetyChecker(
            action_dim=config.get('model.action_dim', 63),
            velocity_threshold=config.get('safety.velocity_limit', 3.0),
            acceleration_threshold=config.get('safety.acceleration_limit', 8.0)
        )

        # Inference settings
        self.use_postprocessing = config.get('inference.use_postprocessing', True)
        self.use_safety_check = config.get('inference.use_safety_check', True)
        self.adaptive_safety = config.get('inference.adaptive_safety', True)
        self.uncertainty_threshold = config.get('inference.uncertainty_threshold', 0.1)

        # Performance tracking
        self.inference_times = []
        self.safety_violations = 0

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
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

    def preprocess_input(
        self,
        pose: np.ndarray,
        point_cloud: np.ndarray,
        tactile: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess input modalities into model-ready format.

        Args:
            pose: [3] object pose
            point_cloud: [6144] flattened point cloud
            tactile: [100] tactile sensor readings

        Returns:
            Preprocessed condition tensor [1, cond_dim]
        """
        # Ensure correct shapes
        pose = np.asarray(pose).flatten()
        point_cloud = np.asarray(point_cloud).flatten()
        tactile = np.asarray(tactile).flatten()

        # Concatenate conditions
        condition = np.concatenate([pose, point_cloud, tactile])

        # Convert to tensor and add batch dimension
        condition_tensor = torch.from_numpy(condition).float().unsqueeze(0).to(self.device)

        return condition_tensor

    def infer_trajectory(
        self,
        pose: np.ndarray,
        point_cloud: np.ndarray,
        tactile: np.ndarray,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Perform trajectory inference.

        Args:
            pose: Object pose [3]
            point_cloud: Point cloud [6144]
            tactile: Tactile readings [100]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            trajectory: Predicted trajectory [seq_len, action_dim]
            uncertainty: Uncertainty estimates (if requested)
            info: Additional inference information
        """
        start_time = time.time()

        # Preprocess input
        condition = self.preprocess_input(pose, point_cloud, tactile)

        # Model inference
        with torch.no_grad():
            # Split condition into modalities
            pose_tensor = condition[:, :3]  # pose_dim = 3
            pc_tensor = condition[:, 3:3+6144]  # pc_dim = 6144
            tactile_tensor = condition[:, 3+6144:]  # tactile_dim = 100
            
            pred_trajectory = self.model(pose_tensor, pc_tensor, tactile_tensor)
            uncertainty = None  # Temporarily disabled

        # Convert to numpy
        trajectory = pred_trajectory.squeeze(0).cpu().numpy()  # [seq_len, action_dim]
        uncertainty_np = uncertainty.squeeze(0).cpu().numpy() if uncertainty is not None else None

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Post-processing
        if self.use_postprocessing:
            trajectory = self.postprocessor.postprocess(trajectory)

        # Safety checking
        safety_info = {}
        if self.use_safety_check:
            is_safe, safety_report = self.safety_checker.comprehensive_safety_check(
                trajectory, point_cloud.reshape(-1, 3) if len(point_cloud) > 0 else None
            )

            safety_info = safety_report
            if not is_safe:
                self.safety_violations += 1

                # Adaptive safety: reduce trajectory magnitude if unsafe
                if self.adaptive_safety:
                    safety_factor = 0.8  # Conservative scaling
                    trajectory *= safety_factor

        # Uncertainty-based filtering
        if return_uncertainty and uncertainty_np is not None:
            high_uncertainty_mask = uncertainty_np > self.uncertainty_threshold
            if np.any(high_uncertainty_mask):
                # Reduce confidence in high-uncertainty timesteps
                confidence_weights = 1.0 / (1.0 + uncertainty_np)
                trajectory *= confidence_weights[:, np.newaxis]

        info = {
            'inference_time': inference_time,
            'safety_info': safety_info,
            'postprocessed': self.use_postprocessing,
            'safety_checked': self.use_safety_check
        }

        return trajectory, uncertainty_np, info

    def infer_single_step(
        self,
        pose: np.ndarray,
        point_cloud: np.ndarray,
        tactile: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single-step inference (last timestep of sequence).

        Args:
            pose: Object pose [3]
            point_cloud: Point cloud [6144]
            tactile: Tactile readings [100]

        Returns:
            action: Single action vector [action_dim]
            info: Inference information
        """
        trajectory, uncertainty, info = self.infer_trajectory(
            pose, point_cloud, tactile, return_uncertainty=False
        )

        # Return last timestep as action
        action = trajectory[-1]

        return action, info

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
            'safety_violations': self.safety_violations,
            'safety_violation_rate': self.safety_violations / len(times) if times.size > 0 else 0.0,
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0.0
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
        self.safety_violations = 0


class RealTimeController:
    """Real-time controller using LeapHand Planner V3."""

    def __init__(self, inference_engine: LeapHandInference, control_freq: float = 30.0):
        self.inference = inference_engine
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Control state
        self.current_trajectory = None
        self.trajectory_index = 0
        self.trajectory_start_time = 0.0

    def start_trajectory(self, pose: np.ndarray, point_cloud: np.ndarray, tactile: np.ndarray):
        """Start executing a new trajectory."""
        trajectory, _, _ = self.inference.infer_trajectory(pose, point_cloud, tactile)
        self.current_trajectory = trajectory
        self.trajectory_index = 0
        self.trajectory_start_time = time.time()

    def get_next_action(self, current_pose: np.ndarray, point_cloud: np.ndarray, tactile: np.ndarray) -> np.ndarray:
        """
        Get next action for real-time control.

        Args:
            current_pose: Current object pose [3]
            point_cloud: Current point cloud [6144]
            tactile: Current tactile readings [100]

        Returns:
            Action vector [action_dim]
        """
        current_time = time.time()
        elapsed_time = current_time - self.trajectory_start_time

        # Calculate target timestep
        target_index = int(elapsed_time / self.dt)

        if self.current_trajectory is None or target_index >= len(self.current_trajectory):
            # Re-plan trajectory
            self.start_trajectory(current_pose, point_cloud, tactile)
            target_index = 0

        # Get action for current timestep
        action = self.current_trajectory[target_index]

        # Update trajectory index
        self.trajectory_index = target_index

        return action

    def is_trajectory_complete(self) -> bool:
        """Check if current trajectory is complete."""
        if self.current_trajectory is None:
            return True

        current_time = time.time()
        elapsed_time = current_time - self.trajectory_start_time
        target_index = int(elapsed_time / self.dt)

        return target_index >= len(self.current_trajectory)


def main():
    parser = argparse.ArgumentParser(description='LeapHand Planner V3 Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='leap_hand_planner_v3/config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with dummy data')
    parser.add_argument('--performance_test', type=int, default=0,
                       help='Run performance test with N inferences')
    args = parser.parse_args()

    # Create inference engine
    print("Loading LeapHand Planner V3...")
    inference = LeapHandInference(args.model_path, args.config, args.device)

    if args.performance_test > 0:
        print(f"Running performance test with {args.performance_test} inferences...")

        # Generate dummy data
        pose = np.random.randn(3)
        point_cloud = np.random.randn(6144)
        tactile = np.random.randn(100)

        for i in range(args.performance_test):
            trajectory, uncertainty, info = inference.infer_trajectory(pose, point_cloud, tactile)

        # Print performance stats
        stats = inference.get_performance_stats()
        print("\nPerformance Statistics:")
        print(".4f")
        print(".4f")
        print(".2f")
        print(".1%")
        print(".1f")

    elif args.demo:
        print("Running demonstration...")

        # Create dummy input data
        pose = np.array([0.1, 0.2, 0.3])  # Example object pose
        point_cloud = np.random.randn(6144)  # Dummy point cloud
        tactile = np.random.randn(100)  # Dummy tactile readings

        print("Input pose:", pose)
        print("Point cloud shape:", point_cloud.shape)
        print("Tactile shape:", tactile.shape)

        # Single trajectory inference
        print("\nInferring trajectory...")
        trajectory, uncertainty, info = inference.infer_trajectory(pose, point_cloud, tactile)

        print("Trajectory shape:", trajectory.shape)
        print("Inference time: .4f")
        print("Safety status:", "Safe" if info['safety_info'].get('overall_safe', True) else "Unsafe")

        # Single step inference
        print("\nInferring single action...")
        action, action_info = inference.infer_single_step(pose, point_cloud, tactile)
        print("Action shape:", action.shape)
        print("Action inference time: .4f")

        # Real-time controller demo
        print("\nTesting real-time controller...")
        controller = RealTimeController(inference)

        controller.start_trajectory(pose, point_cloud, tactile)

        for i in range(5):  # Simulate 5 control steps
            action = controller.get_next_action(pose, point_cloud, tactile)
            print(f"Step {i+1} action norm: {np.linalg.norm(action):.4f}")
            time.sleep(0.1)  # Simulate control loop timing

        print("Demo completed!")

    else:
        print("Use --demo or --performance_test to run examples")
        print("Example: python inference_v3.py --model_path runs/leap_hand_v3/best_model.pth --demo")


if __name__ == '__main__':
    main()