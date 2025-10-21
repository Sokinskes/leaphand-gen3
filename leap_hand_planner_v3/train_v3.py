"""Training script for LeapHand Planner V3 with attention fusion and temporal modeling."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from typing import Dict, Any

from leap_hand_planner_v3.models.planner_v3 import LeapHandPlannerV3
from leap_hand_planner_v3.data.temporal_loader import TemporalDataLoader, TemporalAugmentation
from leap_hand_planner_v3.meta.meta_learner import MetaLearner, TaskSampler
from leap_hand_planner_v3.utils.trajectory_utils import TrajectoryEvaluator
from leap_hand_planner_v3.config import config


class Trainer:
    """Trainer for LeapHand Planner V3."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('training.learning_rate', 1e-4),
            weight_decay=config.get('training.weight_decay', 1e-5)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss weights
        self.temporal_weight = config.get('training.temporal_weight', 0.7)
        self.reconstruction_weight = config.get('training.reconstruction_weight', 0.3)

        # EMA for model weights
        if config.get('training.use_ema', True):
            self.ema_model = self._create_ema_model().to(self.device)
            self.ema_decay = config.get('training.ema_decay', 0.999)

        # Setup logging
        self.setup_logging()

        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience = config.get('training.patience', 10)
        self.patience_counter = 0

    def _create_ema_model(self):
        """Create EMA model copy."""
        ema_model = type(self.model)(**self.model.get_init_kwargs())
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def setup_logging(self):
        """Setup logging and directories."""
        self.log_dir = Path(config.get('logging.log_dir', 'runs/leap_hand_v3'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

        self.logger = logging.getLogger(__name__)

    def temporal_loss(self, pred_trajectory: torch.Tensor, target_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal coherence loss.

        Args:
            pred_trajectory: [batch, seq_len, action_dim]
            target_trajectory: [batch, seq_len, action_dim]

        Returns:
            Temporal loss
        """
        # Velocity consistency loss
        pred_vel = pred_trajectory[:, 1:] - pred_trajectory[:, :-1]
        target_vel = target_trajectory[:, 1:] - target_trajectory[:, :-1]
        vel_loss = nn.functional.mse_loss(pred_vel, target_vel)

        # Acceleration consistency loss
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss = nn.functional.mse_loss(pred_acc, target_acc)

        return vel_loss + 0.5 * acc_loss

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        return nn.functional.mse_loss(pred, target)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                trajectory = batch['trajectory'].to(self.device)
                condition = batch['condition'].to(self.device)

                # Split condition into modalities
                pose = condition[:, :3]  # pose_dim = 3
                pc = condition[:, 3:3+6144]  # pc_dim = 6144
                tactile = condition[:, 3+6144:]  # tactile_dim = 100

                self.optimizer.zero_grad()

                # Forward pass
                pred_trajectory = self.model(pose, pc, tactile)

                # Compute losses
                temp_loss = self.temporal_loss(pred_trajectory, trajectory)
                recon_loss = self.reconstruction_loss(pred_trajectory[:, -1], trajectory[:, -1])  # Last timestep

                total_loss = (
                    self.temporal_weight * temp_loss +
                    self.reconstruction_weight * recon_loss
                )

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training.gradient_clip', 1.0)
                )

                self.optimizer.step()

                # Update EMA
                if hasattr(self, 'ema_model'):
                    self.update_ema()

                epoch_loss += total_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{total_loss.item():.6f}',
                    'temp': f'{temp_loss.item():.6f}',
                    'recon': f'{recon_loss.item():.6f}'
                })

        return epoch_loss / num_batches

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validating') as pbar:
                for batch in pbar:
                    trajectory = batch['trajectory'].to(self.device)
                    condition = batch['condition'].to(self.device)

                    # Split condition into modalities
                    pose = condition[:, :3]
                    pc = condition[:, 3:3+6144]
                    tactile = condition[:, 3+6144:]

                    pred_trajectory = self.model(pose, pc, tactile)

                    # Compute loss (focus on final prediction)
                    loss = self.reconstruction_loss(pred_trajectory[:, -1], trajectory[:, -1])
                    val_loss += loss.item()
                    num_batches += 1

                    # Store for evaluation
                    all_predictions.append(pred_trajectory[:, -1].cpu().numpy())
                    all_targets.append(trajectory[:, -1].cpu().numpy())

                    pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        # Compute evaluation metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        evaluator = TrajectoryEvaluator()
        eval_results = evaluator.evaluate_trajectory(predictions, targets)

        results = {
            'val_loss': val_loss / num_batches,
            'mae': eval_results['errors']['mae'],
            'rmse': eval_results['errors']['rmse'],
            'smoothness': eval_results['smoothness']['velocity_variation'],
            'efficiency': eval_results['efficiency']['efficiency'],
            'safety_score': eval_results['overall_score']
        }

        return results

    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.config
        }

        if hasattr(self, 'ema_model'):
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, self.log_dir / filename)

    def train(self, num_epochs: int):
        """Main training loop."""
        self.logger.info("Starting training...")

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_results = self.validate()

            # Log results
            log_msg = f"Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, "
            log_msg += ", ".join([f"{k}: {v:.6f}" for k, v in val_results.items()])
            self.logger.info(log_msg)

            # Learning rate scheduling
            self.scheduler.step(val_results['val_loss'])

            # Save best model
            if val_results['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_results['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info("Saved best model!")
            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if (epoch + 1) % self.config.get('logging.checkpoint_freq', 10) == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info("Early stopping triggered!")
                break

        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train LeapHand Planner V3')
    parser.add_argument('--config', type=str, default='leap_hand_planner_v3/config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/data.npz',
                       help='Path to data file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--meta_learning', action='store_true',
                       help='Use meta-learning')
    args = parser.parse_args()

    # Load configuration
    if args.config != 'leap_hand_planner_v3/config/default.yaml':
        config.__init__(args.config)

    # Set device
    device = torch.device(args.device)

    # Load data
    print("Loading data...")
    data = np.load(args.data_path)
    trajectories = data['trajectories']
    conditions = np.concatenate([
        data['poses'], data['pcs'], data['tactiles']
    ], axis=1)

    # Create data loaders
    augmentation = TemporalAugmentation(
        seq_len=config.get('data.temporal.seq_len', 10),
        noise_std=config.get('data.augmentation.noise_std', 0.01)
    )

    # Split data
    N = trajectories.shape[0]
    train_size = int(0.8 * N)
    indices = np.random.permutation(N)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_loader = TemporalDataLoader(
        trajectories[train_indices],
        conditions[train_indices],
        batch_size=config.get('training.batch_size', 32),
        seq_len=config.get('data.temporal.seq_len', 10),
        augmentation=augmentation
    )

    val_loader = TemporalDataLoader(
        trajectories[val_indices],
        conditions[val_indices],
        batch_size=config.get('training.batch_size', 32),
        seq_len=config.get('data.temporal.seq_len', 10),
        shuffle=False
    )

    # Create model
    print("Creating model...")
    model = LeapHandPlannerV3(
        action_dim=config.get('model.action_dim', 63),
        pose_dim=config.get('model.pose_dim', 3),
        pc_dim=config.get('model.pc_dim', 6144),
        tactile_dim=config.get('model.tactile_dim', 100),
        seq_len=config.get('model.seq_len', 10),
        hidden_dim=config.get('model.hidden_dim', 512),
        num_heads=config.get('model.num_heads', 8),
        num_layers=config.get('model.num_layers', 4)
    ).to(device)

    # Meta-learning setup
    if args.meta_learning:
        print("Setting up meta-learning...")
        task_sampler = TaskSampler(
            base_data={'trajectories': trajectories, 'conditions': conditions},
            num_tasks=config.get('meta_learning.num_tasks', 10),
            support_size=config.get('meta_learning.support_size', 50),
            query_size=config.get('meta_learning.query_size', 50)
        )

        meta_learner = MetaLearner(
            model,
            meta_algorithm=config.get('meta_learning.algorithm', 'maml'),
            inner_lr=config.get('meta_learning.inner_lr', 0.01),
            meta_lr=config.get('meta_learning.meta_lr', 0.001),
            num_inner_steps=config.get('meta_learning.num_inner_steps', 5)
        )

        print("Training with meta-learning...")
        meta_learner.train_meta(task_sampler, num_meta_epochs=50)

    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config, device)

    print("Starting training...")
    trainer.train(config.get('training.num_epochs', 100))

    print("Training completed! Best model saved.")


if __name__ == '__main__':
    main()