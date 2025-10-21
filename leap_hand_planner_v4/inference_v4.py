#!/usr/bin/env python3
"""
LeapHand Planner V4 Inference Script (ONNX)

This script provides deployment-ready inference using the exported ONNX model.
"""

import numpy as np
import onnxruntime as ort
import time
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LeapHandV4ONNXInference:
    """ONNX-based inference for LeapHand Planner V4."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize ONNX inference session.

        Args:
            model_path: Path to ONNX model file
            device: Device for inference ('cpu', 'cuda', 'tensorrt')
        """
        self.model_path = model_path
        self.device = device

        # Configure providers
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        elif device == 'tensorrt':
            providers.extend([
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider'
            ])
        providers.append('CPUExecutionProvider')

        # Create session
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Loaded ONNX model: {model_path}")
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")

    def preprocess_inputs(
        self,
        pose: np.ndarray,
        pointcloud: np.ndarray,
        tactile: Optional[np.ndarray] = None,
        language: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess inputs for model inference.

        Args:
            pose: End-effector pose [x, y, z, qx, qy, qz, qw] or [x, y, z]
            pointcloud: Point cloud data (will be flattened)
            tactile: Tactile sensor data
            language: Language embedding (optional)

        Returns:
            Preprocessed inputs dictionary
        """
        # Ensure correct shapes
        if pose.ndim == 1:
            pose = pose.reshape(1, -1)
        if pointcloud.ndim == 1:
            pointcloud = pointcloud.reshape(1, -1)
        if tactile is not None and tactile.ndim == 1:
            tactile = tactile.reshape(1, -1)
        if language is not None and language.ndim == 1:
            language = language.reshape(1, -1)

        # Prepare inputs
        inputs = {
            'pose': pose.astype(np.float32),
            'pointcloud': pointcloud.astype(np.float32),
        }

        if tactile is not None:
            inputs['tactile'] = tactile.astype(np.float32)
        else:
            inputs['tactile'] = np.zeros((1, 100), dtype=np.float32)

        if language is not None:
            inputs['language'] = language.astype(np.float32)
        else:
            inputs['language'] = np.zeros((1, 768), dtype=np.float32)

        return inputs

    def generate_trajectory(
        self,
        pose: np.ndarray,
        pointcloud: np.ndarray,
        tactile: Optional[np.ndarray] = None,
        language: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate hand trajectory.

        Args:
            pose: End-effector pose
            pointcloud: Point cloud data
            tactile: Tactile sensor data
            language: Language embedding

        Returns:
            Generated trajectory and inference time
        """
        # Preprocess inputs
        inputs = self.preprocess_inputs(pose, pointcloud, tactile, language)

        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, inputs)
        inference_time = time.time() - start_time

        trajectory = outputs[0]  # [1, seq_len, action_dim]

        return trajectory.squeeze(0), inference_time

    def batch_generate(
        self,
        poses: np.ndarray,
        pointclouds: np.ndarray,
        tactiles: Optional[np.ndarray] = None,
        languages: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate trajectories for batch of inputs.

        Args:
            poses: Batch of poses [batch_size, pose_dim]
            pointclouds: Batch of point clouds [batch_size, pc_dim]
            tactiles: Batch of tactile data [batch_size, tactile_dim]
            languages: Batch of language embeddings [batch_size, lang_dim]

        Returns:
            Generated trajectories and average inference time
        """
        batch_size = poses.shape[0]

        # Preprocess batch
        inputs = self.preprocess_inputs(poses, pointclouds, tactiles, languages)

        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, inputs)
        inference_time = time.time() - start_time

        trajectories = outputs[0]  # [batch_size, seq_len, action_dim]

        return trajectories, inference_time / batch_size


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='LeapHand V4 ONNX Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to ONNX model')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'tensorrt'],
                       help='Inference device')

    args = parser.parse_args()

    # Create inference engine
    inference = LeapHandV4ONNXInference(args.model_path, args.device)

    # Example inputs
    pose = np.array([0.1, 0.2, 0.3])  # Example pose
    pointcloud = np.random.randn(6144)  # Example point cloud
    tactile = np.random.randn(100)  # Example tactile data

    # Generate trajectory
    trajectory, inference_time = inference.generate_trajectory(
        pose, pointcloud, tactile
    )

    print(f"Generated trajectory shape: {trajectory.shape}")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Trajectory sample: {trajectory[0][:5]}")


if __name__ == '__main__':
    main()
