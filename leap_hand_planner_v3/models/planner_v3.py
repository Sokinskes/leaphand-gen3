"""Advanced multimodal trajectory generation models with attention fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import math


class MultimodalAttentionFusion(nn.Module):
    """
    Hierarchical attention fusion for multimodal inputs.
    Fuses pose, point cloud, and tactile modalities with cross-attention.
    """

    def __init__(
        self,
        pose_dim: int = 3,
        pc_dim: int = 6144,
        tactile_dim: int = 100,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.pc_dim = pc_dim
        self.tactile_dim = tactile_dim
        self.hidden_dim = hidden_dim

        # Modality-specific encoders
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pc_encoder = nn.Sequential(
            nn.Linear(pc_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-attention layers for modality fusion
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, pose: torch.Tensor, pc: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        """
        Fuse multimodal inputs using hierarchical attention.

        Args:
            pose: [B, pose_dim]
            pc: [B, pc_dim]
            tactile: [B, tactile_dim]

        Returns:
            Fused representation [B, hidden_dim]
        """
        # Encode each modality
        pose_emb = self.pose_encoder(pose)  # [B, hidden_dim]
        pc_emb = self.pc_encoder(pc)        # [B, hidden_dim]
        tactile_emb = self.tactile_encoder(tactile)  # [B, hidden_dim]

        # Stack modalities for cross-attention
        modalities = torch.stack([pose_emb, pc_emb, tactile_emb], dim=1)  # [B, 3, hidden_dim]

        # Apply cross-attention layers
        for attn_layer in self.cross_attention_layers:
            # Self-attention within modalities
            attn_out, _ = attn_layer(modalities, modalities, modalities)
            modalities = modalities + attn_out  # Residual connection

        # Global fusion
        fused = modalities.mean(dim=1)  # [B, hidden_dim]
        fused = self.fusion_layer(torch.cat([pose_emb, pc_emb, tactile_emb], dim=-1))

        return fused


class TemporalTrajectoryDecoder(nn.Module):
    """
    Transformer-based decoder for temporal trajectory generation.
    Predicts sequence of joint angles with temporal dependencies.
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim: int = 63,
        seq_len: int = 10,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Condition embedding
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # Positional encoding for temporal sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # Trajectory decoder (Transformer decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Trajectory embedding for autoregressive generation
        self.traj_embedding = nn.Linear(action_dim, hidden_dim)

        # Start token for generation
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        condition: torch.Tensor,
        target_traj: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Generate trajectory sequence.

        Args:
            condition: [B, cond_dim]
            target_traj: [B, seq_len, action_dim] (for training)
            teacher_forcing: Whether to use teacher forcing during training

        Returns:
            Predicted trajectory [B, seq_len, action_dim]
        """
        batch_size = condition.shape[0]

        # Embed condition
        cond_emb = self.cond_embedding(condition).unsqueeze(1)  # [B, 1, hidden_dim]

        if self.training and teacher_forcing and target_traj is not None:
            # Teacher forcing: use ground truth as input
            traj_emb = self._embed_trajectory(target_traj)  # [B, seq_len, hidden_dim]
            # Add positional encoding
            traj_emb = traj_emb + self.pos_embedding

            # Create causal mask for autoregressive generation
            causal_mask = self._generate_causal_mask(self.seq_len)

            # Decode
            decoded = self.decoder(
                tgt=traj_emb,
                memory=cond_emb,
                tgt_mask=causal_mask
            )
            output = self.output_proj(decoded)  # [B, seq_len, action_dim]
        else:
            # Autoregressive generation
            output = self._autoregressive_generate(cond_emb, batch_size)

        return output

    def _embed_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Embed trajectory sequence into hidden space."""
        # Use a separate embedding layer for trajectory tokens
        if not hasattr(self, 'traj_embedding'):
            self.traj_embedding = nn.Linear(self.action_dim, self.hidden_dim)
        return self.traj_embedding(traj)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def _autoregressive_generate(self, cond_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Generate trajectory autoregressively."""
        device = cond_emb.device

        # Start with start token
        current_emb = self.start_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]

        generated_actions = []

        for t in range(self.seq_len):
            # Add positional encoding
            pos_emb = self.pos_embedding[:, t:t+1]  # [1, 1, hidden_dim]
            current_emb = current_emb + pos_emb

            # Decode current step
            decoded = self.decoder(
                tgt=current_emb,
                memory=cond_emb
            )  # [B, t+1, hidden_dim]

            # Project to action space for current timestep only
            action_pred = self.output_proj(decoded[:, -1:])  # [B, 1, action_dim]
            generated_actions.append(action_pred)

            # Embed action prediction for next step input
            next_emb = self.traj_embedding(action_pred)  # [B, 1, hidden_dim]
            current_emb = torch.cat([current_emb, next_emb], dim=1)

        return torch.cat(generated_actions, dim=1)  # [B, seq_len, action_dim]


class UncertaintyEstimator(nn.Module):
    """
    Estimate prediction uncertainty for adaptive safety thresholds.
    Uses ensemble of models or dropout-based uncertainty estimation.
    """

    def __init__(self, num_samples: int = 10):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, model: nn.Module, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction and uncertainty.

        Args:
            model: The model to estimate uncertainty for

        Returns:
            Tuple of (prediction, uncertainty)
        """
        predictions = []

        # Enable dropout during inference for uncertainty estimation
        was_training = model.training
        model.train()  # Enable dropout

        # Generate multiple predictions with dropout
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = model(*args, **kwargs)
                predictions.append(pred)

        # Restore original training mode
        model.train(was_training)

        predictions = torch.stack(predictions, dim=0)  # [num_samples, ...]

        # Calculate mean and uncertainty (variance)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1, keepdim=True)  # Reduce to scalar uncertainty

        return mean_pred, uncertainty


class LeapHandPlannerV3(nn.Module):
    """
    Third-generation LeapHand trajectory planner with advanced features.
    """

    def __init__(
        self,
        pose_dim: int = 3,
        pc_dim: int = 6144,
        tactile_dim: int = 100,
        action_dim: int = 63,
        seq_len: int = 10,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        # Save initialization parameters
        self.pose_dim = pose_dim
        self.pc_dim = pc_dim
        self.tactile_dim = tactile_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.fusion = MultimodalAttentionFusion(
            pose_dim=pose_dim,
            pc_dim=pc_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=3
        )

        # Temporal trajectory decoder
        self.decoder = TemporalTrajectoryDecoder(
            cond_dim=hidden_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Uncertainty estimator (暂时禁用以避免递归)
        # self.uncertainty_estimator = UncertaintyEstimator(self)

    def forward(
        self,
        pose: torch.Tensor,
        pc: torch.Tensor,
        tactile: torch.Tensor,
        target_traj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for trajectory generation.

        Args:
            pose: [B, pose_dim]
            pc: [B, pc_dim]
            tactile: [B, tactile_dim]
            target_traj: [B, seq_len, action_dim] (optional, for training)

        Returns:
            Predicted trajectory [B, seq_len, action_dim]
        """
        # Fuse multimodal inputs
        fused_condition = self.fusion(pose, pc, tactile)  # [B, hidden_dim]

        # Generate trajectory sequence
        trajectory = self.decoder(fused_condition, target_traj)

        return trajectory

    def predict_trajectory(
        self,
        pose: torch.Tensor,
        pc: torch.Tensor,
        tactile: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict trajectory with optional uncertainty estimation.

        Args:
            pose: [B, pose_dim] or [pose_dim]
            pc: [B, pc_dim] or [pc_dim]
            tactile: [B, tactile_dim] or [tactile_dim]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (trajectory, uncertainty)
        """
        # Ensure batch dimension
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        if pc.dim() == 1:
            pc = pc.unsqueeze(0)
        if tactile.dim() == 1:
            tactile = tactile.unsqueeze(0)

        if return_uncertainty:
            # 暂时禁用不确定性估计
            trajectory = self(pose, pc, tactile)
            uncertainty = torch.zeros_like(trajectory[:, :, :1])  # 占位符
            return trajectory.squeeze(0), uncertainty.squeeze(0)
        else:
            trajectory = self(pose, pc, tactile)
            return trajectory.squeeze(0), None

    def get_init_kwargs(self) -> Dict[str, Any]:
        """Get initialization arguments for creating model copies."""
        return {
            'pose_dim': self.pose_dim,
            'pc_dim': self.pc_dim,
            'tactile_dim': self.tactile_dim,
            'action_dim': self.action_dim,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }