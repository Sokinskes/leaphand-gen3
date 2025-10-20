from diffusers import DDPMScheduler, UNet1DModel
import torch.nn as nn
import torch


class DiffusionPlanner(nn.Module):
    def __init__(self, cond_dim=6144+100+3, seq_len=63, in_channels=1, hidden_dim=256):
        """
        seq_len: the length of the 1D sequence (we treat action vector as sequence length)
        in_channels: number of channels (we use 1)
        hidden_dim: internal MLP dim for cond encoder
        """
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # cond encoder: map cond to additive embedding
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.seq_len)
        )

        # Simple Conv1D network for denoising
        self.unet = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.in_channels, kernel_size=3, padding=1)
        )
        # debug flag for shape tracing
        self.DEBUG = False

    def forward(self, noise, t, cond):
        # noise: [B, in_channels, seq_len]
        # cond: [B, cond_dim]
        # one-time debug print
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            try:
                print(f"[DEBUG] forward input shapes: noise={tuple(noise.shape)}, t={tuple(t.shape) if hasattr(t, 'shape') else t}, cond={tuple(cond.shape) if hasattr(cond, 'shape') else cond}, device={next(self.parameters()).device}")
            except Exception:
                pass
            self._debug_printed = True

        B = noise.shape[0]

        # cond_emb: [B, seq_len]
        cond_emb = self.cond_encoder(cond)
        if getattr(self, 'DEBUG', False):
            print(f"[DEBUG] cond_emb shape after encoder: {tuple(cond_emb.shape)}")

        # expand to [B, 1, seq_len]
        cond_emb = cond_emb.unsqueeze(1)
        if getattr(self, 'DEBUG', False):
            print(f"[DEBUG] cond_emb shape after unsqueeze: {tuple(cond_emb.shape)}")

        x = noise + cond_emb  # [B, 1, seq_len]
        if getattr(self, 'DEBUG', False):
            print(f"[DEBUG] x shape before UNet: {tuple(x.shape)}, expected in_channels={self.in_channels}")
        # strict assertion to catch shape mismatches
        assert x.shape == (B, self.in_channels, self.seq_len), f"x shape {x.shape} != expected {(B, self.in_channels, self.seq_len)}"

        # Simple Conv1D network
        pred = self.unet(x)  # [B, 1, seq_len]

        return pred

    def generate_trajectory(self, cond, num_steps=None):
        """Generate a denoised action sequence given condition cond.
        If num_steps is None, use self.seq_len as sequence length.
        Returns numpy array of shape [seq_len] (flattened channels).
        """
        device = next(self.parameters()).device
        seq_len = self.seq_len if num_steps is None else num_steps
        # noise shape [1, 1, seq_len]
        noise = torch.randn(1, self.in_channels, seq_len, device=device)
        cond = cond.to(device)
        for t in self.scheduler.timesteps:
            pred = self.forward(noise, t, cond)
            noise = self.scheduler.step(pred, t, noise).prev_sample
        out = noise.squeeze(0).cpu().numpy()  # [in_channels, seq_len]
        return out.squeeze(0)  # [seq_len]

# ---- MPC约束轨迹优化 ----
import numpy as np
from scipy.optimize import minimize

def mpc_optimize_trajectory(traj, collision_fn, smooth_weight=1.0):
    # traj: [traj_len, action_dim]，collision_fn: callable(traj) -> 碰撞损失
    def loss(flat_traj):
        traj_ = flat_traj.reshape(traj.shape)
        smooth = np.sum(np.diff(traj_, axis=0)**2)
        collision = collision_fn(traj_)
        return smooth_weight * smooth + collision
    res = minimize(loss, traj.flatten(), method='L-BFGS-B')
    return res.x.reshape(traj.shape)