"""Behavioral Cloning models for trajectory generation."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional


class BCPlanner(nn.Module):
    """
    Behavioral Cloning Planner: Simple MLP regressor for trajectory generation.

    Input: condition vector (poses + pcs + tactiles)
    Output: trajectory vector (action_dim)
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.1
    ):
        """
        Initialize BC Planner.

        Args:
            cond_dim: Dimension of condition input
            action_dim: Dimension of action output
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()
        layers = []
        in_dim = cond_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cond: Condition tensor [B, cond_dim]

        Returns:
            Trajectory tensor [B, action_dim]
        """
        return self.net(cond)

    def generate_trajectory(self, cond: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Generate trajectory for single condition.

        Args:
            cond: Condition vector [cond_dim] or [1, cond_dim]

        Returns:
            Trajectory as numpy array [action_dim]
        """
        if isinstance(cond, np.ndarray):
            cond = torch.from_numpy(cond).float()

        if cond.ndim == 1:
            cond = cond.unsqueeze(0)

        with torch.no_grad():
            traj = self(cond)

        return traj.squeeze(0).cpu().numpy()