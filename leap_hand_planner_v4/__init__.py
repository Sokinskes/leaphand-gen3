"""LeapHand Planner V4 Package.

V4 Architecture: Diffusion-Transformer Hybrid with Memory Gates
for dexterous manipulation planning across multiple robotic hands.
"""

from .models.planner_v4 import (
    LeapHandPlannerV4,
    DEFAULT_HAND_CONFIGS,
    SinusoidalPositionalEncoding,
    MemoryGatedAttention,
    DiffusionEncoder,
    UnifiedHandModel
)
from .data.temporal_loader import TemporalLeapHandDataset
from .utils.multi_gpu_training import MultiGPUTrainer

__version__ = "4.0.0"
__all__ = [
    "LeapHandPlannerV4",
    "DEFAULT_HAND_CONFIGS",
    "SinusoidalPositionalEncoding",
    "MemoryGatedAttention",
    "DiffusionEncoder",
    "UnifiedHandModel",
    "TemporalLeapHandDataset",
    "MultiGPUTrainer"
]