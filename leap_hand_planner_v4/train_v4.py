#!/usr/bin/env python3
"""Simplified Single GPU Training for LeapHand Planner V4.

This script provides single GPU training for development and testing.
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from leap_hand_planner_v4.models import LeapHandPlannerV4, DEFAULT_HAND_CONFIGS
from leap_hand_planner_v4.data.temporal_loader import TemporalLeapHandDataset, create_data_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleGPUTrainer:
    """Single GPU trainer for LeapHand Planner V4."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: TemporalLeapHandDataset,
        val_dataset: TemporalLeapHandDataset = None,
        batch_size: int = 16,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        save_dir: str = "./runs",
        gradient_clip_val: float = 1.0,
        mixed_precision: bool = False,
        log_every: int = 10,
        save_every: int = 100,
        eval_every: int = 50,
    ):
        self.model = model.cuda()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_dir = Path(save_dir)
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.save_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(exist_ok=True)

        # Setup data loaders
        self.train_loader = create_data_loader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        if val_dataset is not None:
            self.val_loader = create_data_loader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
        else:
            self.val_loader = None

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs * len(self.train_loader)
        )

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = []

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats,
            'config': self.model.get_init_kwargs() if hasattr(self.model, 'get_init_kwargs') else {}
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.run_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.run_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _log_stats(self, stats: dict, step: int):
        """Log training statistics."""
        log_str = f"Step {step}: "
        for key, value in stats.items():
            log_str += f"{key}={value:.4f} "
        logger.info(log_str)

        # Save to stats file
        stats_file = self.run_dir / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                # The model forward returns predicted noise during training
                output = self.model(**batch)
                # For diffusion models, we typically use MSE loss between predicted and actual noise
                # But here we're using the model's internal loss computation
                loss = torch.mean((output - batch['trajectory']) ** 2)

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                stats = {
                    'loss': loss.item(),
                    'lr': current_lr,
                    'epoch': getattr(self, 'current_epoch', 0)
                }
                self._log_stats(stats, self.global_step)
                self.training_stats.append({**stats, 'step': self.global_step})

            # Save checkpoint
            if self.global_step % self.save_every == 0:
                self._save_checkpoint(getattr(self, 'current_epoch', 0), loss.item())

        return epoch_loss / num_batches

    def validate(self) -> float:
        """Run validation."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    output = self.model(**batch)
                    loss = torch.mean((output - batch['trajectory']) ** 2)

                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self._save_checkpoint(getattr(self, 'current_epoch', 0), avg_val_loss, is_best=True)

        return avg_val_loss

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()

            # Validation
            if self.val_loader is not None and epoch % (self.eval_every // len(self.train_loader) + 1) == 0:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        logger.info("Training completed!")


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
    seq_len: int = 10,
    train_split: float = 0.8
):
    """Create training and validation datasets."""

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

    logger.info(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    return train_dataset, val_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LeapHand Planner V4 (Single GPU)')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--hand_name', type=str, default='leaphand',
                       choices=['leaphand', 'shadowhand', 'allegrohand'],
                       help='Hand type for training')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay')

    # Training optimizations
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                       help='Use mixed precision training')

    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./runs',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_every', type=int, default=50,
                       help='Run validation every N steps')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
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
        }
    }

    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Training hand: {args.hand_name}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        args.data_path,
        args.hand_name,
        seq_len=10,
        train_split=0.8
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(args.hand_name)

    # Create trainer
    trainer = SingleGPUTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=str(save_dir),
        gradient_clip_val=args.gradient_clip_val,
        mixed_precision=args.mixed_precision,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()