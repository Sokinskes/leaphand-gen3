# LeapHand Planner V3
# Multimodal Transformer for Trajectory Planning

"""
Advanced multimodal trajectory generation with attention fusion and temporal modeling.

Key Innovations:
- Hierarchical attention fusion for multimodal inputs
- Temporal sequence prediction with Transformer decoder
- Adaptive safety thresholds based on uncertainty
- Meta-learning for fast adaptation to new tasks/objects
"""

from .models.planner_v3 import LeapHandPlannerV3
from .data.temporal_loader import TemporalDataLoader
from .utils.trajectory_utils import TrajectoryPostprocessor, SafetyChecker, TrajectoryEvaluator

__version__ = "3.0.0"
__all__ = ["LeapHandPlannerV3", "TemporalDataLoader", "TrajectoryPostprocessor", "SafetyChecker", "TrajectoryEvaluator"]