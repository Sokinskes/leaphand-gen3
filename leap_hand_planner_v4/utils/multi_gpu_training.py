"""Multi-GPU Training Configuration for LeapHand Planner V4.

This module provides distributed training utilities optimized for 8-GPU setups,
enabling efficient parallel training of the Diffusion-Transformer hybrid model.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPUTrainer:
    """Multi-GPU trainer for LeapHand Planner V4 with optimized 8-GPU setup."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 1000,
        save_dir: str = "./runs",
        gradient_clip_val: float = 1.0,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        log_every: int = 10,
        save_every: int = 1000,
        eval_every: int = 500,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.save_dir = Path(save_dir)
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every

        # Distributed training setup
        self.world_size = torch.cuda.device_count()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_master = self.rank == 0

        # Create save directory
        if self.is_master:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir = self.save_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run_dir.mkdir(exist_ok=True)

        # Setup components
        self._setup_distributed()
        self._setup_data_loaders()
        self._setup_optimizer()
        self._setup_scaler()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = []

    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_initialized():
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )

        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Move model to device and wrap with DDP
        self.model = self.model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True  # For dynamic architectures
        )

    def _setup_data_loaders(self):
        """Setup distributed data loaders."""
        # Training sampler
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.world_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )

        # Validation loader (optional)
        if self.val_dataset is not None:
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size // self.world_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Separate parameters for different learning rates
        diffusion_params = []
        transformer_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'diffusion' in name.lower():
                diffusion_params.append(param)
            elif 'transformer' in name.lower() or 'attention' in name.lower():
                transformer_params.append(param)
            else:
                other_params.append(param)

        # Different learning rates for different components
        self.optimizer = AdamW([
            {'params': diffusion_params, 'lr': self.learning_rate * 0.1, 'weight_decay': self.weight_decay},
            {'params': transformer_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay},
            {'params': other_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay}
        ])

        # Cosine annealing scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs * len(self.train_loader),
            eta_min=self.learning_rate * 0.01
        )

    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision."""
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_master:
            return

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats,
            'config': self.model.module.get_init_kwargs() if hasattr(self.model.module, 'get_init_kwargs') else {}
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.run_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.run_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

        # Save latest
        latest_path = self.run_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _log_stats(self, stats: Dict[str, float], step: int):
        """Log training statistics."""
        if not self.is_master:
            return

        # Log to console
        log_str = f"Step {step}: "
        for key, value in stats.items():
            log_str += f"{key}={value:.4f} "
        logger.info(log_str)

        # Save to stats file
        stats_file = self.run_dir / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss = self.model(**batch)
                loss = loss / self.accumulation_steps  # Normalize for accumulation

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
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
            epoch_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                stats = {
                    'loss': loss.item() * self.accumulation_steps,
                    'lr': current_lr,
                    'epoch': self.current_epoch
                }
                self._log_stats(stats, self.global_step)
                self.training_stats.append({**stats, 'step': self.global_step})

            # Save checkpoint
            if self.global_step % self.save_every == 0:
                self._save_checkpoint(self.current_epoch, loss.item())

            # Validation
            if self.val_loader is not None and self.global_step % self.eval_every == 0:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(self.current_epoch, val_loss, is_best=True)

        return {'train_loss': epoch_loss / num_batches}

    def validate(self) -> float:
        """Run validation."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    loss = self.model(**batch)

                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches

        if self.is_master:
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training on {self.world_size} GPUs")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)

            # Train epoch
            epoch_stats = self.train_epoch()

            # Log epoch summary
            if self.is_master:
                logger.info(f"Epoch {epoch}: {epoch_stats}")

            # Save epoch checkpoint
            if self.is_master and epoch % 10 == 0:
                self._save_checkpoint(epoch, epoch_stats['train_loss'])

        # Final validation and save
        if self.val_loader is not None:
            final_val_loss = self.validate()
            if final_val_loss < self.best_val_loss:
                self._save_checkpoint(self.num_epochs - 1, final_val_loss, is_best=True)

        if self.is_master:
            logger.info("Training completed!")


def setup_distributed():
    """Setup distributed training environment."""
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    os.environ['RANK'] = os.environ.get('RANK', '0')
    os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')


def launch_training(
    model_fn: Callable[[], nn.Module],
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    **trainer_kwargs
):
    """Launch distributed training across all available GPUs."""

    def train_worker(rank: int, world_size: int):
        """Worker function for each GPU process."""
        # Set environment variables
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # Setup distributed training
        setup_distributed()

        # Create model
        model = model_fn()

        # Create trainer
        trainer = MultiGPUTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **trainer_kwargs
        )

        # Start training
        trainer.train()

    # Launch processes
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        # Single GPU training
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        setup_distributed()
        model = model_fn()
        trainer = MultiGPUTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **trainer_kwargs
        )
        trainer.train()


# Performance monitoring utilities
class GPUMonitor:
    """Monitor GPU utilization during training."""

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.step = 0

    def log_gpu_stats(self):
        """Log GPU memory and utilization stats."""
        if self.step % self.log_every == 0:
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
                logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        self.step += 1


def get_optimal_batch_size(model: nn.Module, max_batch_size: int = 128) -> int:
    """Find optimal batch size that fits in GPU memory."""

    def try_batch_size(batch_size: int) -> bool:
        try:
            # Create dummy input
            dummy_input = {
                'pose': torch.randn(batch_size, 3),
                'pc': torch.randn(batch_size, 6144),
                'tactile': torch.randn(batch_size, 100),
                'trajectory': torch.randn(batch_size, 10, 63),
                't': torch.randint(0, 50, (batch_size,))
            }

            # Move to GPU
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}

            # Forward pass
            with torch.no_grad():
                output = model(**dummy_input)

            # Clear cache
            torch.cuda.empty_cache()
            return True

        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    # Binary search for optimal batch size
    low, high = 1, max_batch_size
    optimal = 1

    while low <= high:
        mid = (low + high) // 2
        if try_batch_size(mid):
            optimal = mid
            low = mid + 1
        else:
            high = mid - 1

    logger.info(f"Optimal batch size: {optimal}")
    return optimal