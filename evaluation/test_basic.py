"""Basic tests for LeapHand planner components."""

import numpy as np
import torch
from leap_hand_planner.models.bc_planner import BCPlanner
from leap_hand_planner.data.loader import load_data, normalize_conditions
from leap_hand_planner.utils.trajectory import postprocess_trajectory, safety_check


def test_bc_planner():
    """Test BC Planner model."""
    model = BCPlanner(cond_dim=10, action_dim=5, hidden_dims=[32, 16])

    # Test forward pass
    batch_size = 4
    cond = torch.randn(batch_size, 10)
    output = model(cond)

    assert output.shape == (batch_size, 5), f"Expected shape {(batch_size, 5)}, got {output.shape}"
    print("✓ BC Planner forward pass test passed")


def test_trajectory_processing():
    """Test trajectory processing utilities."""
    # Test postprocessing
    traj = np.random.randn(63)
    smoothed = postprocess_trajectory(traj)

    assert smoothed.shape == traj.shape, "Postprocessing should preserve shape"
    print("✓ Trajectory postprocessing test passed")

    # Test safety check
    error = 5.0
    is_safe, err = safety_check(error, threshold=10.0)
    assert is_safe == True, "Error below threshold should be safe"

    is_safe, err = safety_check(error, threshold=3.0)
    assert is_safe == False, "Error above threshold should be unsafe"
    print("✓ Safety check test passed")


def test_data_loading():
    """Test data loading (requires data files)."""
    try:
        trajs, conds = load_data("data/data.npz")
        assert trajs.shape[1] == 63, f"Expected 63 action dims, got {trajs.shape[1]}"
        assert conds.shape[1] == 6247, f"Expected 6247 cond dims, got {conds.shape[1]}"
        print("✓ Data loading test passed")
    except FileNotFoundError:
        print("⚠ Data loading test skipped (data files not found)")


if __name__ == "__main__":
    print("Running basic tests...")
    test_bc_planner()
    test_trajectory_processing()
    test_data_loading()
    print("All tests completed!")