#!/usr/bin/env python3
"""LeapHand Planner V4 Training Script with 8-GPU Support.

This script provides comprehensive training capabilities for the V4 architecture,
optimized for multi-GPU setups with advanced features like mixed precision,
gradient accumulation, and distributed data parallel training.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from leap_hand_planner_v4.models import LeapHandPlannerV4, DEFAULT_HAND_CONFIGS
from leap_hand_planner_v4.utils.multi_gpu_training import MultiGPUTrainer, get_optimal_batch_size
from leap_hand_planner_v4.data.temporal_loader import TemporalLeapHandDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(hand_name: str = 'leaphand') -> LeapHandPlannerV4:
    """Create V4 model instance."""
    model = LeapHandPlannerV4(
        hand_configs=DEFAULT_HAND_CONFIGS,
        pose_dim=3,
        pc_dim=6144,
        tactile_dim=100,
        language_dim=768,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        seq_len=10,
        diffusion_steps=50,
        memory_dim=256
    )

    model.set_hand(hand_name)
    return model


def create_datasets(
    data_path: str,
    hand_name: str = 'leaphand',
    train_split: float = 0.8,
    seq_len: int = 10
):
    """Create training and validation datasets."""

    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    # Create dataset
    dataset = TemporalLeapHandDataset(
        data_path=data_path,
        hand_name=hand_name,
        seq_len=seq_len,
        augment=True
    )

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Created datasets: Train={train_size}, Val={val_size}")
    return train_dataset, val_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LeapHand Planner V4')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--hand_name', type=str, default='leaphand',
                       choices=['leaphand', 'shadowhand', 'allegrohand'],
                       help='Hand type for training')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Learning rate warmup steps')

    # Training optimizations
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--auto_batch_size', action='store_true',
                       help='Automatically find optimal batch size')

    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./runs',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_every', type=int, default=500,
                       help='Run validation every N steps')

    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


def save_training_config(args, save_dir: Path):
    """Save training configuration."""
    config = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'model_config': {
            'hand_configs': DEFAULT_HAND_CONFIGS,
            'pose_dim': 3,
            'pc_dim': 6144,
            'tactile_dim': 100,
            'language_dim': 768,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'seq_len': 10,
            'diffusion_steps': 50,
            'memory_dim': 256
        },
        'gpu_info': {
            'num_gpus': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        }
    }

    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved training config to {config_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_training_config(args, save_dir)

    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"Training hand: {args.hand_name}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        args.data_path,
        args.hand_name,
        seq_len=10
    )

    # Auto batch size detection
    if args.auto_batch_size:
        logger.info("Finding optimal batch size...")
        model = create_model(args.hand_name)
        optimal_batch_size = get_optimal_batch_size(model)
        args.batch_size = optimal_batch_size
        logger.info(f"Using batch size: {args.batch_size}")

    # Model creation function for distributed training
    def model_fn():
        return create_model(args.hand_name)

    # Launch training (single GPU for now)
    logger.info("Starting single GPU training...")

    # Create model
    model = create_model(args.hand_name)

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        args.data_path,
        args.hand_name,
        seq_len=10
    )

    # Create trainer
    trainer = MultiGPUTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_dir=str(save_dir),
        gradient_clip_val=args.gradient_clip_val,
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every
    )

    # Start training
    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()