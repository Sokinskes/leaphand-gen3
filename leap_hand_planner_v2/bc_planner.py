import torch
import torch.nn as nn
import numpy as np

class BCPlanner(nn.Module):
    """
    Behavioral Cloning Planner: Simple MLP regressor for trajectory generation.
    Input: condition vector (poses + pcs + tactiles)
    Output: trajectory vector (action_dim)
    """
    def __init__(self, cond_dim, action_dim, hidden_dims=[1024, 512, 256]):
        super().__init__()
        layers = []
        in_dim = cond_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Add dropout for robustness
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        """
        cond: [B, cond_dim]
        Returns: [B, action_dim]
        """
        return self.net(cond)

    def generate_trajectory(self, cond):
        """
        Generate trajectory for single condition.
        cond: [1, cond_dim] or [cond_dim]
        Returns: numpy array [action_dim]
        """
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        with torch.no_grad():
            traj = self(cond)
        return traj.squeeze(0).cpu().numpy()