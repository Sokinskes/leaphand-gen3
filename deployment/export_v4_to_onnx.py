#!/usr/bin/env python3
"""ONNX Export for LeapHand Planner V4.

This script exports the trained V4 model to ONNX format for deployment.
"""

import torch
import torch.onnx
import numpy as np
import argparse
from pathlib import Path
import logging
import onnx
import onnxruntime as ort

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from leap_hand_planner_v4.models import LeapHandPlannerV4

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    simplify: bool = True,
    validate: bool = True
):
    """Export V4 model to ONNX format."""

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = LeapHandPlannerV4(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_hand('leaphand')
    model.eval()

    # Create dummy inputs for ONNX export
    # These should match the model's forward method signature
    dummy_pose = torch.randn(1, 3)  # [batch_size, pose_dim]
    dummy_pc = torch.randn(1, 6144)  # [batch_size, pc_dim]
    dummy_tactile = torch.randn(1, 100)  # [batch_size, tactile_dim]
    dummy_language = torch.zeros(1, 768)  # [batch_size, language_dim]

    # For trajectory generation (inference mode), we don't need trajectory and t
    # The model will automatically call generate_trajectory when trajectory is None

    logger.info("Exporting model to ONNX...")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_pose, dummy_pc, dummy_tactile, dummy_language),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pose', 'pointcloud', 'tactile', 'language'],
        output_names=['trajectory'],
        dynamic_axes={
            'pose': {0: 'batch_size'},
            'pointcloud': {0: 'batch_size'},
            'tactile': {0: 'batch_size'},
            'language': {0: 'batch_size'},
            'trajectory': {0: 'batch_size'}
        },
        verbose=False
    )

    logger.info(f"Model exported to {output_path}")

    # Simplify ONNX model if requested
    if simplify:
        try:
            import onnxsim
            logger.info("Simplifying ONNX model...")

            # Load and simplify
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnxsim.simplify(onnx_model)

            if check:
                onnx.save(simplified_model, output_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed")

        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    # Validate ONNX model
    if validate:
        logger.info("Validating ONNX model...")

        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(output_path)

            # Prepare inputs for validation
            test_inputs = {
                'pose': dummy_pose.numpy(),
                'pointcloud': dummy_pc.numpy(),
                'tactile': dummy_tactile.numpy(),
                'language': dummy_language.numpy()
            }

            # Run inference
            ort_outputs = ort_session.run(None, test_inputs)

            # Compare with PyTorch model
            with torch.no_grad():
                torch_output = model(dummy_pose, dummy_pc, dummy_tactile, dummy_language)

            # Check outputs match (within tolerance)
            torch_output_np = torch_output.cpu().numpy()
            max_diff = np.max(np.abs(ort_outputs[0] - torch_output_np))

            if max_diff < 1e-5:
                logger.info(f"ONNX validation passed (max diff: {max_diff:.2e})")
            else:
                logger.warning(f"ONNX validation failed (max diff: {max_diff:.2e})")

        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")

    # Print model information
    onnx_model = onnx.load(output_path)
    logger.info(f"ONNX model info:")
    logger.info(f"  - Opset version: {onnx_model.opset_import[0].version}")
    logger.info(f"  - Producer: {onnx_model.producer_name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  - Total parameters: {total_params:,}")

    return output_path


def create_deployment_script(onnx_path: str, output_dir: str):
    """Create a deployment-ready inference script."""

    script_content = '''#!/usr/bin/env python3
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
        if pointcloud.ndim == 2:
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
'''

    script_path = Path(output_dir) / 'leap_hand_v4_inference.py'
    with open(script_path, 'w') as f:
        f.write(script_content)

    logger.info(f"Created deployment script: {script_path}")


def create_dockerfile(onnx_path: str, output_dir: str):
    """Create Dockerfile for deployment."""

    dockerfile_content = '''# LeapHand Planner V4 Deployment Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    numpy \\
    onnxruntime \\
    onnxruntime-gpu \\
    torch \\
    torchvision

# Create app directory
WORKDIR /app

# Copy model and inference script
COPY leap_hand_v4.onnx ./model.onnx
COPY leap_hand_v4_inference.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port (if needed for API)
EXPOSE 8000

# Default command
CMD ["python", "leap_hand_v4_inference.py", "--model_path", "model.onnx"]
'''

    dockerfile_path = Path(output_dir) / 'Dockerfile'
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)

    logger.info(f"Created Dockerfile: {dockerfile_path}")


def main():
    """Main ONNX export function."""
    parser = argparse.ArgumentParser(description='Export LeapHand Planner V4 to ONNX')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='leap_hand_v4.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate ONNX model')
    parser.add_argument('--create_deployment', action='store_true', default=True,
                       help='Create deployment artifacts')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Export to ONNX
    onnx_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify=args.simplify,
        validate=args.validate
    )

    # Create deployment artifacts
    if args.create_deployment:
        output_dir = Path(args.output).parent
        create_deployment_script(onnx_path, str(output_dir))
        create_dockerfile(onnx_path, str(output_dir))

        logger.info("Deployment artifacts created successfully!")

    logger.info(f"ONNX export completed! Model saved to: {onnx_path}")


if __name__ == '__main__':
    main()