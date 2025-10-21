"""Utility functions for trajectory post-processing, safety checking, and evaluation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional
from scipy import signal
import matplotlib.pyplot as plt


class TrajectoryPostprocessor:
    """
    Advanced trajectory post-processing with smoothing, safety constraints, and optimization.
    """

    def __init__(
        self,
        action_dim: int = 63,
        smoothing_window: int = 5,
        velocity_limit: float = 2.0,
        acceleration_limit: float = 5.0,
        joint_limits: Optional[np.ndarray] = None
    ):
        self.action_dim = action_dim
        self.smoothing_window = smoothing_window
        self.velocity_limit = velocity_limit
        self.acceleration_limit = acceleration_limit

        # Default joint limits for LeapHand (example values)
        if joint_limits is None:
            self.joint_limits = np.array([
                [-np.pi/2, np.pi/2],  # Joint 1
                [-np.pi/2, np.pi/2],  # Joint 2
                # ... extend for all 16 joints * 3 DOF + thumb
            ] * 16).flatten()[:action_dim]
        else:
            self.joint_limits = joint_limits

    def smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to trajectory using Savitzky-Golay filter.

        Args:
            trajectory: [seq_len, action_dim] or [action_dim]

        Returns:
            Smoothed trajectory
        """
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)

        smoothed = np.zeros_like(trajectory)

        for i in range(self.action_dim):
            # Apply Savitzky-Golay smoothing
            smoothed[:, i] = signal.savgol_filter(
                trajectory[:, i],
                window_length=min(self.smoothing_window, trajectory.shape[0]),
                polyorder=2
            )

        return smoothed

    def enforce_velocity_limits(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Enforce velocity limits by scaling down excessive velocities.

        Args:
            trajectory: [seq_len, action_dim]

        Returns:
            Velocity-constrained trajectory
        """
        if trajectory.shape[0] < 2:
            return trajectory

        # Calculate velocities
        velocities = np.diff(trajectory, axis=0)

        # Find excessive velocities
        velocity_magnitudes = np.linalg.norm(velocities, axis=1, keepdims=True)
        excess_mask = velocity_magnitudes > self.velocity_limit

        if np.any(excess_mask):
            # Scale down excessive velocities
            scale_factors = self.velocity_limit / (velocity_magnitudes + 1e-6)
            scale_factors = np.clip(scale_factors, 0, 1)
            velocities *= scale_factors

            # Reconstruct trajectory
            constrained_traj = np.zeros_like(trajectory)
            constrained_traj[0] = trajectory[0]
            constrained_traj[1:] = trajectory[0] + np.cumsum(velocities, axis=0)

            return constrained_traj

        return trajectory

    def enforce_acceleration_limits(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Enforce acceleration limits.

        Args:
            trajectory: [seq_len, action_dim]

        Returns:
            Acceleration-constrained trajectory
        """
        if trajectory.shape[0] < 3:
            return trajectory

        # Calculate accelerations
        accelerations = np.diff(trajectory, n=2, axis=0)

        # Find excessive accelerations
        accel_magnitudes = np.linalg.norm(accelerations, axis=1, keepdims=True)
        excess_mask = accel_magnitudes > self.acceleration_limit

        if np.any(excess_mask):
            # Simple acceleration limiting by smoothing
            return self.smooth_trajectory(trajectory)

        return trajectory

    def enforce_joint_limits(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Enforce joint angle limits by clipping.

        Args:
            trajectory: [seq_len, action_dim]

        Returns:
            Joint-limit constrained trajectory
        """
        constrained = trajectory.copy()

        for i in range(self.action_dim):
            min_limit, max_limit = self.joint_limits[i*2:(i+1)*2] if i*2+1 < len(self.joint_limits) else (-np.pi, np.pi)
            constrained[:, i] = np.clip(constrained[:, i], min_limit, max_limit)

        return constrained

    def postprocess(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply full post-processing pipeline.

        Args:
            trajectory: Raw trajectory

        Returns:
            Post-processed trajectory
        """
        processed = trajectory.copy()

        # Apply smoothing
        processed = self.smooth_trajectory(processed)

        # Enforce constraints
        processed = self.enforce_velocity_limits(processed)
        processed = self.enforce_acceleration_limits(processed)
        processed = self.enforce_joint_limits(processed)

        return processed


class SafetyChecker:
    """
    Comprehensive safety checking for LeapHand trajectories.
    """

    def __init__(
        self,
        action_dim: int = 63,
        velocity_threshold: float = 3.0,
        acceleration_threshold: float = 8.0,
        joint_limits: Optional[np.ndarray] = None,
        collision_check: bool = True
    ):
        self.action_dim = action_dim
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.collision_check = collision_check

        # Joint limits
        if joint_limits is None:
            self.joint_limits = np.array([
                [-np.pi/2, np.pi/2] for _ in range(action_dim)
            ])
        else:
            self.joint_limits = joint_limits

    def check_joint_limits(self, trajectory: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if trajectory violates joint limits.

        Returns:
            (is_safe, violation_info)
        """
        violations = []

        for t in range(trajectory.shape[0]):
            for i in range(self.action_dim):
                joint_angle = trajectory[t, i]
                min_limit, max_limit = self.joint_limits[i]

                # Debug: check if joint_angle is array
                if isinstance(joint_angle, np.ndarray):
                    print(f"joint_angle is array: {joint_angle.shape}, value: {joint_angle}")
                    joint_angle = joint_angle.item() if joint_angle.size == 1 else joint_angle[0]

                if joint_angle < min_limit or joint_angle > max_limit:
                    violations.append({
                        'timestep': t,
                        'joint': i,
                        'angle': joint_angle,
                        'limits': [min_limit, max_limit]
                    })

        is_safe = len(violations) == 0

        return is_safe, {
            'violations': violations,
            'violation_count': len(violations)
        }

    def check_velocities(self, trajectory: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Check velocity constraints.

        Returns:
            (is_safe, violation_info)
        """
        if trajectory.shape[0] < 2:
            return True, {'max_velocity': 0.0, 'violations': []}

        velocities = np.diff(trajectory, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        max_velocity = np.max(velocity_magnitudes)
        violations = []

        for t, vel in enumerate(velocity_magnitudes):
            if vel > self.velocity_threshold:
                violations.append({
                    'timestep': t,
                    'velocity': vel,
                    'threshold': self.velocity_threshold
                })

        is_safe = len(violations) == 0

        return is_safe, {
            'max_velocity': max_velocity,
            'violations': violations,
            'violation_count': len(violations)
        }

    def check_accelerations(self, trajectory: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Check acceleration constraints.

        Returns:
            (is_safe, violation_info)
        """
        if trajectory.shape[0] < 3:
            return True, {'max_acceleration': 0.0, 'violations': []}

        accelerations = np.diff(trajectory, n=2, axis=0)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)

        max_acceleration = np.max(accel_magnitudes)
        violations = []

        for t, accel in enumerate(accel_magnitudes):
            if accel > self.acceleration_threshold:
                violations.append({
                    'timestep': t,
                    'acceleration': accel,
                    'threshold': self.acceleration_threshold
                })

        is_safe = len(violations) == 0

        return is_safe, {
            'max_acceleration': max_acceleration,
            'violations': violations,
            'violation_count': len(violations)
        }

    def check_collision(self, trajectory: np.ndarray, point_cloud: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Basic collision checking using point cloud data.

        Returns:
            (is_safe, collision_info)
        """
        # Simplified collision check - check if fingertips are too close to point cloud
        # This is a placeholder for actual collision detection
        collisions = []

        # Assume last few joints are fingertips
        fingertip_indices = [-3, -2, -1]  # Last 3 joints

        for t in range(trajectory.shape[0]):
            fingertip_positions = trajectory[t, fingertip_indices]

            # Simple distance check to point cloud
            distances = np.linalg.norm(point_cloud - fingertip_positions, axis=1)
            min_distance = np.min(distances)

            if min_distance < 0.05:  # 5cm threshold
                collisions.append({
                    'timestep': t,
                    'min_distance': min_distance,
                    'threshold': 0.05
                })

        is_safe = len(collisions) == 0

        return is_safe, {
            'collisions': collisions,
            'collision_count': len(collisions)
        }

    def comprehensive_safety_check(
        self,
        trajectory: np.ndarray,
        point_cloud: Optional[np.ndarray] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform comprehensive safety checking.

        Returns:
            (is_safe, safety_report)
        """
        safety_report = {
            'overall_safe': True,
            'joint_limits': {},
            'velocities': {},
            'accelerations': {},
            'collisions': {}
        }

        # Check joint limits
        safe, info = self.check_joint_limits(trajectory)
        safety_report['joint_limits'] = info
        safety_report['overall_safe'] &= safe

        # Check velocities
        safe, info = self.check_velocities(trajectory)
        safety_report['velocities'] = info
        safety_report['overall_safe'] &= safe

        # Check accelerations
        safe, info = self.check_accelerations(trajectory)
        safety_report['accelerations'] = info
        safety_report['overall_safe'] &= safe

        # Check collisions if enabled and point cloud provided
        if self.collision_check and point_cloud is not None:
            safe, info = self.check_collision(trajectory, point_cloud)
            safety_report['collisions'] = info
            safety_report['overall_safe'] &= safe

        # Add total violation count
        total_violations = 0
        for check_name, check_info in safety_report.items():
            if isinstance(check_info, dict) and 'violation_count' in check_info:
                total_violations += check_info['violation_count']
        safety_report['total_violations'] = total_violations

        return safety_report['overall_safe'], safety_report


class TrajectoryEvaluator:
    """
    Comprehensive trajectory evaluation metrics.
    """

    def __init__(self, action_dim: int = 63):
        self.action_dim = action_dim

    def compute_trajectory_error(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute various trajectory error metrics.

        Args:
            predicted: [seq_len, action_dim]
            ground_truth: [seq_len, action_dim]

        Returns:
            Error metrics dictionary
        """
        if predicted.shape != ground_truth.shape:
            raise ValueError("Predicted and ground truth shapes must match")

        errors = predicted - ground_truth

        # Mean Absolute Error
        mae = np.mean(np.abs(errors))

        # Root Mean Square Error
        rmse = np.sqrt(np.mean(errors**2))

        # Max error
        max_error = np.max(np.abs(errors))

        # Per-joint errors
        per_joint_mae = np.mean(np.abs(errors), axis=0)
        per_joint_rmse = np.sqrt(np.mean(errors**2, axis=0))

        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'per_joint_mae': per_joint_mae,
            'per_joint_rmse': per_joint_rmse
        }

    def compute_smoothness_metrics(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Compute trajectory smoothness metrics.

        Args:
            trajectory: [seq_len, action_dim]

        Returns:
            Smoothness metrics
        """
        if trajectory.shape[0] < 3:
            return {'velocity_variation': 0.0, 'acceleration_variation': 0.0}

        # Velocity variation (jerk-like metric)
        velocities = np.diff(trajectory, axis=0)
        velocity_changes = np.diff(velocities, axis=0)
        velocity_variation = np.mean(np.abs(velocity_changes))

        # Acceleration variation
        accelerations = np.diff(trajectory, n=2, axis=0)
        accel_changes = np.diff(accelerations, axis=0)
        acceleration_variation = np.mean(np.abs(accel_changes))

        return {
            'velocity_variation': velocity_variation,
            'acceleration_variation': acceleration_variation
        }

    def compute_efficiency_metrics(
        self,
        trajectory: np.ndarray,
        start_pose: np.ndarray,
        goal_pose: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute trajectory efficiency metrics.

        Args:
            trajectory: [seq_len, action_dim]
            start_pose: [action_dim]
            goal_pose: [action_dim]

        Returns:
            Efficiency metrics
        """
        # Path length
        displacements = np.diff(trajectory, axis=0)
        path_length = np.sum(np.linalg.norm(displacements, axis=1))

        # Direct distance
        direct_distance = np.linalg.norm(goal_pose - start_pose)

        # Efficiency ratio
        efficiency = direct_distance / (path_length + 1e-6)

        # Total displacement
        total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

        return {
            'path_length': path_length,
            'direct_distance': direct_distance,
            'efficiency': efficiency,
            'total_displacement': total_displacement
        }

    def evaluate_trajectory(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        point_cloud: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive trajectory evaluation.

        Returns:
            Full evaluation report
        """
        report = {}

        # Error metrics
        report['errors'] = self.compute_trajectory_error(predicted, ground_truth)

        # Smoothness metrics
        report['smoothness'] = self.compute_smoothness_metrics(predicted)

        # Efficiency metrics
        start_pose = ground_truth[0]
        goal_pose = ground_truth[-1]
        report['efficiency'] = self.compute_efficiency_metrics(
            predicted, start_pose, goal_pose
        )

        # Safety check
        safety_checker = SafetyChecker()
        is_safe, safety_report = safety_checker.comprehensive_safety_check(
            predicted, point_cloud
        )
        report['safety'] = safety_report

        # Overall score (weighted combination)
        error_score = 1.0 / (1.0 + report['errors']['mae'])  # Lower error = higher score
        smoothness_score = 1.0 / (1.0 + report['smoothness']['velocity_variation'])
        efficiency_score = report['efficiency']['efficiency']
        safety_score = 1.0 if is_safe else 0.0

        report['overall_score'] = (
            0.4 * error_score +
            0.2 * smoothness_score +
            0.2 * efficiency_score +
            0.2 * safety_score
        )

        return report