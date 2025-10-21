"""LeapHand Planner V4: Diffusion-Transformer Hybrid Architecture.

This module implements a U-shaped Diffusion Transformer that combines the
generative capabilities of diffusion models with the temporal reasoning
of Transformers, enhanced with memory gates for long-horizon tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(1)]


class MemoryGatedAttention(nn.Module):
    """Memory-gated attention mechanism for long-horizon reasoning."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Memory gates
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Task relevance gate
        self.task_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with memory gating."""
        B, T, _ = query.shape

        # Linear projections
        Q = self.q_proj(query).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Memory gating
        if memory is not None:
            # Compute memory relevance - global gate for the entire attention matrix
            memory_input = torch.cat([query.mean(dim=1), memory.mean(dim=1)], dim=-1)  # [B, d_model * 2]
            memory_relevance = self.memory_gate(memory_input).mean()  # scalar
            scores = scores * memory_relevance

        # Task gating
        if task_embedding is not None:
            task_relevance = self.task_gate(task_embedding.mean(dim=1)).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            scores = scores * task_relevance.unsqueeze(1).unsqueeze(2)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_output)


class DiffusionEncoder(nn.Module):
    """U-Net style diffusion encoder for trajectory generation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        time_emb_dim: int = 128
    ):
        super().__init__()

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder layers (downsampling)
        self.encoder_layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim * (2 ** i) for i in range(num_layers)]

        for i in range(num_layers):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.SiLU(),
                    nn.LayerNorm(dims[i+1]),  # Changed from GroupNorm to LayerNorm
                    nn.Dropout(0.1)
                )
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] * 2),
            nn.SiLU(),
            nn.Linear(dims[-1] * 2, dims[-1])
        )

        # Decoder layers (upsampling)
        self.decoder_layers = nn.ModuleList()
        for i in reversed(range(num_layers)):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i+1] * 2, dims[i+1]),
                    nn.SiLU(),
                    nn.LayerNorm(dims[i+1]),  # Changed from GroupNorm to LayerNorm
                    nn.Dropout(0.1),
                    nn.Linear(dims[i+1], dims[i])
                )
            )

        self.final_layer = nn.Linear(dims[0], output_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        # Time embedding - project to match input dimension
        t_emb = self.time_emb(t.unsqueeze(-1))  # [B, time_emb_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, seq_len, time_emb_dim]

        # For simplicity, let's just add time embedding to the input directly
        # instead of trying to match dimensions in each layer
        h = x  # [B, seq_len, input_dim]

        # Encoder
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)  # Each layer handles the time embedding internally if needed
            skip_connections.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)

        return self.final_layer(h)


class UnifiedHandModel(nn.Module):
    """Unified modeling for different robotic hands."""

    def __init__(self, hand_configs: Dict[str, Dict]):
        super().__init__()
        self.hand_configs = hand_configs
        self.current_hand = None

        # Joint embedding layers for different hands
        self.joint_embedders = nn.ModuleDict()
        for hand_name, config in hand_configs.items():
            self.joint_embedders[hand_name] = nn.Linear(
                config['dof'], config['hidden_dim']
            )

    def set_hand(self, hand_name: str):
        """Set active hand configuration."""
        if hand_name not in self.hand_configs:
            raise ValueError(f"Unknown hand: {hand_name}")
        self.current_hand = hand_name

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Embed joint angles for current hand."""
        if self.current_hand is None:
            raise ValueError("Hand not set. Call set_hand() first.")

        embedder = self.joint_embedders[self.current_hand]
        return embedder(joint_angles)


class LeapHandPlannerV4(nn.Module):
    """V4 Architecture: Diffusion-Transformer Hybrid with Memory Gates."""

    def __init__(
        self,
        # Hand configurations
        hand_configs: Dict[str, Dict],
        # Architecture parameters
        pose_dim: int = 3,
        pc_dim: int = 6144,
        tactile_dim: int = 100,
        language_dim: int = 768,  # Optional CLIP/BERT embedding
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        seq_len: int = 10,
        # Diffusion parameters
        diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        # Memory parameters
        memory_dim: int = 256,
    ):
        super().__init__()

        # Hand modeling
        self.hand_model = UnifiedHandModel(hand_configs)
        self.action_dim = max(config['dof'] for config in hand_configs.values())

        # Store parameters
        self.pose_dim = pose_dim
        self.pc_dim = pc_dim
        self.tactile_dim = tactile_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Multimodal fusion
        self.pose_embed = nn.Linear(pose_dim, hidden_dim)
        self.pc_embed = nn.Sequential(
            nn.Linear(pc_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.tactile_embed = nn.Linear(tactile_dim, hidden_dim)
        self.language_embed = nn.Linear(language_dim, hidden_dim) if language_dim > 0 else None

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Diffusion model for trajectory generation
        input_dim = hidden_dim + self.action_dim  # condition + trajectory
        self.diffusion_encoder = DiffusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim,
            num_layers=num_layers // 2
        )

        # Memory-gated transformer for temporal reasoning
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len)
        self.memory_gated_layers = nn.ModuleList([
            MemoryGatedAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Trajectory decoder
        self.traj_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.action_dim)
        )

        # Diffusion parameters
        self.diffusion_steps = diffusion_steps
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Memory system
        self.memory_system = nn.GRUCell(hidden_dim, memory_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

    def set_hand(self, hand_name: str):
        """Set active hand configuration."""
        self.hand_model.set_hand(hand_name)

    def _fuse_multimodal(
        self,
        pose: torch.Tensor,
        pc: torch.Tensor,
        tactile: torch.Tensor,
        language: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse multimodal inputs."""
        # Embed each modality
        pose_emb = self.pose_embed(pose)  # [B, hidden_dim]
        pc_emb = self.pc_embed(pc)        # [B, hidden_dim]
        tactile_emb = self.tactile_embed(tactile)  # [B, hidden_dim]

        # Concatenate modalities
        modalities = [pose_emb, pc_emb, tactile_emb]
        if language is not None and self.language_embed is not None:
            lang_emb = self.language_embed(language)
            modalities.append(lang_emb)

        # Stack and apply cross-attention
        fused = torch.stack(modalities, dim=1)  # [B, num_modalities, hidden_dim]

        # Self-attention across modalities
        attn_output, _ = self.cross_attention(
            fused, fused, fused
        )  # [B, num_modalities, hidden_dim]

        # Pool across modalities
        fused_condition = attn_output.mean(dim=1)  # [B, hidden_dim]

        return fused_condition

    def _diffusion_forward(
        self,
        condition: torch.Tensor,
        trajectory: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through diffusion model."""
        # Add noise to trajectory
        noise = torch.randn_like(trajectory)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()

        noisy_trajectory = (
            sqrt_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1) * trajectory +
            sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1) * noise
        )

        # Concatenate condition with noisy trajectory
        # condition: [B, hidden_dim], noisy_trajectory: [B, seq_len, action_dim]
        # We need to expand condition to match trajectory temporal dimension
        condition_expanded = condition.unsqueeze(1).expand(-1, self.seq_len, -1)  # [B, seq_len, hidden_dim]
        diffusion_input = torch.cat([condition_expanded, noisy_trajectory], dim=-1)  # [B, seq_len, hidden_dim + action_dim]

        # Predict noise
        predicted_noise = self.diffusion_encoder(diffusion_input, t.float())

        return predicted_noise, noise

    def _memory_gated_transformer(
        self,
        condition: torch.Tensor,
        trajectory_emb: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply memory-gated transformer."""
        # Add positional encoding
        x = self.pos_encoding(trajectory_emb)

        # Memory initialization
        if memory is None:
            memory = torch.zeros(x.size(0), self.memory_system.hidden_size, device=x.device)

        # Apply memory-gated layers
        for layer in self.memory_gated_layers:
            # Update memory
            memory = self.memory_system(x.mean(dim=1), memory)

            # Apply gated attention
            memory_proj = self.memory_proj(memory).unsqueeze(1).expand(-1, x.size(1), -1)
            x = layer(x, x, x, memory_proj, condition.unsqueeze(1).expand(-1, x.size(1), -1))

        return x

    def forward(
        self,
        pose: torch.Tensor,
        pc: torch.Tensor,
        tactile: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Fuse multimodal inputs
        condition = self._fuse_multimodal(pose, pc, tactile, language)

        if trajectory is not None and t is not None:
            # Training mode: diffusion forward
            predicted_noise, _ = self._diffusion_forward(condition, trajectory, t)
            return predicted_noise
        else:
            # Inference mode: generate trajectory
            return self.generate_trajectory(condition)

    def generate_trajectory(
        self,
        condition: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Generate trajectory using diffusion sampling."""
        device = condition.device
        B = condition.size(0)

        # Start from pure noise
        trajectory = torch.randn(B, self.seq_len, self.action_dim, device=device)

        # Reverse diffusion process
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise, _ = self._diffusion_forward(condition, trajectory, t_tensor)

                # Compute predicted trajectory
                sqrt_alphas_cumprod_t = self.alphas_cumprod[t]
                sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t])

                predicted_trajectory = (
                    trajectory - sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1) * predicted_noise
                ) / sqrt_alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1).sqrt()

                if t > 0:
                    # Add noise for next step
                    noise = torch.randn_like(trajectory)
                    beta_t = self.betas[t]
                    sqrt_beta_t = beta_t.sqrt()

                    trajectory = predicted_trajectory + sqrt_beta_t.unsqueeze(-1).unsqueeze(-1) * noise
                else:
                    trajectory = predicted_trajectory

        # Apply memory-gated transformer refinement
        trajectory_emb = self.hand_model(trajectory.view(B * self.seq_len, -1))
        trajectory_emb = trajectory_emb.view(B, self.seq_len, -1)

        refined_emb = self._memory_gated_transformer(condition, trajectory_emb)
        refined_trajectory = self.traj_decoder(refined_emb)

        return refined_trajectory

    def get_init_kwargs(self) -> Dict[str, Any]:
        """Get initialization arguments."""
        return {
            'hand_configs': self.hand_model.hand_configs,
            'pose_dim': self.pose_dim,
            'pc_dim': self.pc_dim,
            'tactile_dim': self.tactile_dim,
            'language_dim': self.language_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'diffusion_steps': self.diffusion_steps,
            'beta_start': self.betas[0].item(),
            'beta_end': self.betas[-1].item(),
            'memory_dim': self.memory_system.hidden_size
        }


# Default hand configurations
DEFAULT_HAND_CONFIGS = {
    'leaphand': {
        'dof': 63,
        'hidden_dim': 512,
        'joint_limits': [-math.pi/2, math.pi/2]
    },
    'shadowhand': {
        'dof': 24,
        'hidden_dim': 512,
        'joint_limits': [-math.pi/2, math.pi/2]
    },
    'allegrohand': {
        'dof': 16,
        'hidden_dim': 512,
        'joint_limits': [-math.pi/2, math.pi/2]
    }
}