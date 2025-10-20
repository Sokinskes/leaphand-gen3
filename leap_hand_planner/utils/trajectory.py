"""Utility functions for trajectory processing and evaluation."""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple


def postprocess_trajectory(
    traj: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for trajectory smoothing.

    Args:
        traj: Trajectory array [action_dim]
        window_length: Filter window length
        polyorder: Polynomial order

    Returns:
        Smoothed trajectory
    """
    if len(traj) < window_length:
        return traj  # No smoothing if too short
    return savgol_filter(traj, window_length, polyorder)


def safety_check(error_deg: float, threshold: float = 10.0) -> Tuple[bool, float]:
    """
    Check if error is within safe threshold.

    Args:
        error_deg: Error in degrees
        threshold: Safety threshold in degrees

    Returns:
        Tuple of (is_safe, error_deg)
    """
    return error_deg <= threshold, error_deg


def compute_trajectory_error(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    degrees: bool = True
) -> float:
    """
    Compute mean absolute error between predicted and ground truth trajectories.

    Args:
        pred_traj: Predicted trajectory [action_dim]
        gt_traj: Ground truth trajectory [action_dim]
        degrees: Whether to convert to degrees (assuming radians input)

    Returns:
        Mean absolute error
    """
    error = np.mean(np.abs(pred_traj - gt_traj))
    if degrees:
        error *= 180 / np.pi
    return error