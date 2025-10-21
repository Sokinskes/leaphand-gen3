"""Configuration settings for LeapHand Planner V3."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for LeapHand Planner V3."""

    def __init__(self, config_path: str = "leap_hand_planner_v3/config/default.yaml"):
        self.config_path = Path(__file__).parent / "default.yaml"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return self._get_default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Model architecture
            'model': {
                'action_dim': 63,
                'pose_dim': 3,
                'pc_dim': 6144,
                'tactile_dim': 100,
                'cond_dim': 6247,  # pose + pc + tactile
                'seq_len': 10,
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'attention_fusion': True,
                'temporal_modeling': True,
                'uncertainty_estimation': True
            },

            # Training parameters
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'patience': 10,
                'gradient_clip': 1.0,
                'use_ema': True,
                'ema_decay': 0.999,
                'temporal_weight': 0.7,
                'reconstruction_weight': 0.3
            },

            # Meta-learning parameters
            'meta_learning': {
                'algorithm': 'maml',  # 'maml' or 'reptile'
                'inner_lr': 0.01,
                'meta_lr': 0.001,
                'num_inner_steps': 5,
                'num_tasks': 10,
                'support_size': 50,
                'query_size': 50,
                'first_order': True
            },

            # Data parameters
            'data': {
                'data_path': 'data/data.npz',
                'augmentation': {
                    'noise_std': 0.01,
                    'time_warp_factor': 0.1,
                    'dropout_prob': 0.1,
                    'rotation_range': 0.1,
                    'translation_range': 0.05
                },
                'temporal': {
                    'seq_len': 10,
                    'overlap': 0.5
                }
            },

            # Safety and post-processing
            'safety': {
                'velocity_limit': 2.0,
                'acceleration_limit': 5.0,
                'joint_limits': [-1.57, 1.57],  # -pi/2 to pi/2
                'collision_threshold': 0.05,
                'enable_collision_check': True
            },

            'postprocessing': {
                'smoothing_window': 5,
                'velocity_limit': 2.0,
                'acceleration_limit': 5.0
            },

            # Evaluation parameters
            'evaluation': {
                'metrics': ['mae', 'rmse', 'smoothness', 'efficiency', 'safety'],
                'leave_one_out': True,
                'num_samples': 1000,
                'save_plots': True,
                'plot_dir': 'evaluation_plots'
            },

            # Inference parameters
            'inference': {
                'batch_size': 1,
                'use_postprocessing': True,
                'use_safety_check': True,
                'temporal_horizon': 10,
                'uncertainty_threshold': 0.1,
                'adaptive_safety': True
            },

            # Logging and checkpointing
            'logging': {
                'log_dir': 'runs/leap_hand_v3',
                'checkpoint_freq': 10,
                'save_best': True,
                'tensorboard': True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: str = None):
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)


# Global configuration instance
config = Config()