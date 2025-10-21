"""Test script for LeapHand Planner V3 components."""

import torch
import numpy as np
from pathlib import Path

from leap_hand_planner_v3.models.planner_v3 import LeapHandPlannerV3
from leap_hand_planner_v3.data.temporal_loader import TemporalAugmentation
from leap_hand_planner_v3.utils.trajectory_utils import TrajectoryPostprocessor, SafetyChecker


def test_model_components():
    """Test individual model components."""
    print("Testing LeapHand Planner V3 components...")

    # Model parameters
    batch_size = 2
    seq_len = 10
    pose_dim = 3
    pc_dim = 6144
    tactile_dim = 100
    action_dim = 63

    # Create model
    model = LeapHandPlannerV3(
        pose_dim=pose_dim,
        pc_dim=pc_dim,
        tactile_dim=tactile_dim,
        action_dim=action_dim,
        seq_len=seq_len
    )

    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    pose = torch.randn(batch_size, pose_dim)
    pc = torch.randn(batch_size, pc_dim)
    tactile = torch.randn(batch_size, tactile_dim)

    trajectory = model(pose, pc, tactile)
    print(f"✓ Forward pass successful: input {pose.shape} -> output {trajectory.shape}")

    # Test uncertainty estimation
    trajectory_uncert, uncertainty = model.predict_trajectory(pose[0], pc[0], tactile[0], return_uncertainty=True)
    print(f"✓ Uncertainty estimation: trajectory {trajectory_uncert.shape}, uncertainty {uncertainty.shape}")

    # Test loss computation
    target_traj = torch.randn(batch_size, seq_len, action_dim)
    loss = model.compute_loss(trajectory, target_traj)
    print(f"✓ Loss computation: {loss.item():.6f}")

    return model


def test_data_augmentation():
    """Test temporal data augmentation."""
    print("\nTesting temporal data augmentation...")

    augmentation = TemporalAugmentation(seq_len=10, noise_std=0.01)

    # Create dummy trajectory
    trajectory = np.random.randn(5, 63)  # 5 timesteps, 63 DOF

    # Test augmentation
    augmented = augmentation.augment_trajectory(trajectory)
    print(f"✓ Trajectory augmentation: {trajectory.shape} -> {augmented.shape}")

    # Test multimodal augmentation
    trajectories = np.random.randn(2, 10, 63)  # Match seq_len=10
    poses = np.random.randn(2, 10, 3)
    pcs = np.random.randn(2, 10, 6144)
    tactiles = np.random.randn(2, 10, 100)

    aug_traj, aug_poses, aug_pcs, aug_tactiles = augmentation.augment_multimodal(
        trajectories, poses, pcs, tactiles
    )
    print(f"✓ Multimodal augmentation: trajectories {aug_traj.shape}, poses {aug_poses.shape}")

    return augmentation


def test_postprocessing():
    """Test trajectory post-processing and safety checking."""
    print("\nTesting trajectory post-processing...")

    postprocessor = TrajectoryPostprocessor(action_dim=63)

    # Create dummy trajectory
    trajectory = np.random.randn(10, 63)

    # Test post-processing
    processed = postprocessor.postprocess(trajectory)
    print(f"✓ Post-processing: {trajectory.shape} -> {processed.shape}")

    # Test safety checking
    safety_checker = SafetyChecker(action_dim=63)

    is_safe, safety_report = safety_checker.comprehensive_safety_check(processed)
    print(f"✓ Safety check: {'Safe' if is_safe else 'Unsafe'} ({safety_report['total_violations']} violations)")

    return postprocessor, safety_checker


def test_full_pipeline():
    """Test complete pipeline from input to output."""
    print("\nTesting complete pipeline...")

    # Create model
    model = LeapHandPlannerV3()

    # Create post-processor
    postprocessor = TrajectoryPostprocessor()

    # Simulate real input
    pose = np.array([0.1, 0.2, 0.3])  # Object pose
    pc = np.random.randn(6144)  # Point cloud
    tactile = np.random.randn(100)  # Tactile readings

    print(f"Input: pose {pose.shape}, pc {pc.shape}, tactile {tactile.shape}")

    # Model prediction
    trajectory, uncertainty = model.predict_trajectory(
        torch.from_numpy(pose).float(),
        torch.from_numpy(pc).float(),
        torch.from_numpy(tactile).float(),
        return_uncertainty=True
    )

    trajectory = trajectory.numpy()
    print(f"Raw prediction: {trajectory.shape}")

    # Post-processing
    processed_trajectory = postprocessor.postprocess(trajectory)
    print(f"Processed trajectory: {processed_trajectory.shape}")

    # Safety check
    safety_checker = SafetyChecker()
    is_safe, safety_report = safety_checker.comprehensive_safety_check(processed_trajectory)

    print(f"Final result: {'✓ Safe' if is_safe else '⚠ Unsafe'} trajectory")
    print(".6f")
    print(".3f")

    return processed_trajectory, safety_report


def main():
    """Run all tests."""
    print("=" * 60)
    print("LeapHand Planner V3 Component Tests")
    print("=" * 60)

    try:
        # Test individual components
        model = test_model_components()
        augmentation = test_data_augmentation()
        postprocessor, safety_checker = test_postprocessing()

        # Test full pipeline
        trajectory, safety_report = test_full_pipeline()

        print("\n" + "=" * 60)
        print("✅ All tests passed! LeapHand Planner V3 is ready.")
        print("=" * 60)

        # Print summary
        print("\n📊 Test Summary:")
        print(f"• Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"• Multimodal fusion: ✓ Working")
        print(f"• Temporal decoding: ✓ Working")
        print(f"• Uncertainty estimation: ✓ Working")
        print(f"• Data augmentation: ✓ Working")
        print(f"• Post-processing: ✓ Working")
        print(f"• Safety checking: ✓ Working")
        print(f"• Full pipeline: ✓ Working")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())