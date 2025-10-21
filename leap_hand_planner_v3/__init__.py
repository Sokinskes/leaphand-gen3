"""
LeapHand Trajectory Planner V3 - Third Generation
Advanced multimodal trajectory generation with attention fusion and temporal modeling.

Key Innovations:
- Hierarchical attention fusion for multimodal inputs (pose + pointcloud + tactile)
- Temporal sequence prediction with Transformer decoder
- Adaptive safety thresholds based on uncertainty estimation
- Meta-learning for fast adaptation to new tasks/objects
- Real-time inference optimization with ONNX support
"""

from .models.planner_v3 import LeapHandPlannerV3
from .data.temporal_loader import TemporalDataLoader
from .utils.trajectory_utils import TrajectoryUtils
from .meta.meta_learner import MetaLearner
from .config.default import get_default_config

__version__ = "3.0.0"
__all__ = [
    "LeapHandPlannerV3",
    "TemporalDataLoader",
    "TrajectoryUtils",
    "MetaLearner",
    "get_default_config"
]