#!/usr/bin/env python3
"""
Inference script for LeapHand trajectory generation.
Use this for real-time trajectory prediction on the robotic platform.
"""

import torch
import numpy as np
from pathlib import Path
from leap_hand_planner.models.bc_planner import BCPlanner
from leap_hand_planner.data.loader import load_data, normalize_conditions
from leap_hand_planner.utils.trajectory import postprocess_trajectory, safety_check
from leap_hand_planner.config import load_config


class LeapHandInference:
    """Real-time inference engine for LeapHand trajectory generation."""

    def __init__(self, model_path: str, config_path: str = "leap_hand_planner/config/default.yaml"):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)

        # Infer dimensions from data
        data_path = self.config['data']['data_file']
        _, conds = load_data(data_path)
        _, cond_mean, cond_std = normalize_conditions(conds, self.config['data']['stats_file'])

        action_dim = 63  # LeapHand has 16 joints * 3 + 15 = 63 DOF
        cond_dim = conds.shape[1]

        self.model = BCPlanner(
            cond_dim=cond_dim,
            action_dim=action_dim,
            hidden_dims=self.config['model']['hidden_dims'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Store normalization stats
        self.cond_mean = cond_mean
        self.cond_std = cond_std

        print(f"Loaded model from {model_path}")
        print(f"Model: {cond_dim} -> {action_dim}")

    def predict_trajectory(self, pose: np.ndarray, pc: np.ndarray, tactile: np.ndarray) -> np.ndarray:
        """
        Predict trajectory from sensor inputs.

        Args:
            pose: Object pose [3] (x, y, z)
            pc: Point cloud [6144] (flattened)
            tactile: Tactile readings [100]

        Returns:
            Trajectory [63] in radians
        """
        # Concatenate inputs
        cond = np.concatenate([pose, pc, tactile])

        # Normalize
        cond_norm = (cond - self.cond_mean.squeeze()) / self.cond_std.squeeze()
        cond_tensor = torch.from_numpy(cond_norm).float().unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            traj = self.model(cond_tensor)

        # Post-process
        traj_np = traj.squeeze(0).cpu().numpy()
        traj_smooth = postprocess_trajectory(
            traj_np,
            window_length=self.config['evaluation']['postprocess_window'],
            polyorder=self.config['evaluation']['postprocess_polyorder']
        )

        return traj_smooth

    def validate_trajectory(self, trajectory: np.ndarray, ground_truth: np.ndarray = None) -> dict:
        """
        Validate trajectory safety and optionally compute error.

        Args:
            trajectory: Predicted trajectory [63]
            ground_truth: Optional ground truth trajectory [63]

        Returns:
            Validation results dictionary
        """
        results = {}

        if ground_truth is not None:
            error = np.mean(np.abs(trajectory - ground_truth)) * 180 / np.pi
            is_safe, _ = safety_check(error, self.config['evaluation']['safety_threshold_deg'])
            results.update({
                'error_deg': error,
                'is_safe': is_safe
            })

        # Basic validation (joint limits, smoothness, etc.)
        results.update({
            'joint_range_valid': np.all(np.abs(trajectory) < np.pi),  # Basic joint limit check
            'smoothness': np.std(np.diff(trajectory)),  # Trajectory smoothness
        })

        return results


def main():
    """Example usage."""
    # Initialize inference engine
    model_path = "runs/run_bc_20251020_120954/best_model.pth"  # Update with your model path
    inference = LeapHandInference(model_path)

    # Example inputs (replace with real sensor data)
    pose = np.random.randn(3) * 0.1  # Object pose
    pc = np.random.randn(6144) * 0.01  # Point cloud
    tactile = np.random.randn(100) * 0.1  # Tactile readings

    # Predict trajectory
    trajectory = inference.predict_trajectory(pose, pc, tactile)

    # Validate
    validation = inference.validate_trajectory(trajectory)

    print(f"Generated trajectory shape: {trajectory.shape}")
    print(f"Validation results: {validation}")


if __name__ == "__main__":
    main()