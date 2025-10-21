"""Meta-learning components for fast adaptation and few-shot learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning for fast adaptation to new tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        first_order: bool = True
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Meta optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

    def inner_loop(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        task_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inner loop adaptation on support set and evaluate on query set.

        Args:
            support_data: Support set data
            query_data: Query set data
            task_model: Model for this task

        Returns:
            Loss and predictions on query set
        """
        # Inner loop adaptation
        adapted_params = []
        for param in task_model.parameters():
            adapted_params.append(param.clone())

        # Adapt on support set
        for _ in range(self.num_inner_steps):
            support_pred = task_model(support_data['condition'])
            support_loss = F.mse_loss(support_pred, support_data['trajectory'][:, -1])  # Last timestep

            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=not self.first_order
            )

            # Update parameters
            for i, (param, grad) in enumerate(zip(adapted_params, grads)):
                adapted_params[i] = param - self.inner_lr * grad

        # Load adapted parameters
        for param, adapted_param in zip(task_model.parameters(), adapted_params):
            param.data.copy_(adapted_param.data)

        # Evaluate on query set
        query_pred = task_model(query_data['condition'])
        query_loss = F.mse_loss(query_pred, query_data['trajectory'][:, -1])

        return query_loss, query_pred

    def meta_update(self, task_losses: List[torch.Tensor]):
        """
        Perform meta-update using losses from multiple tasks.

        Args:
            task_losses: List of losses from different tasks
        """
        meta_loss = torch.stack(task_losses).mean()

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

    def adapt_to_task(
        self,
        task_data: Dict[str, torch.Tensor],
        num_adaptation_steps: int = 10
    ) -> nn.Module:
        """
        Adapt the model to a new task using few-shot data.

        Args:
            task_data: Task-specific data for adaptation
            num_adaptation_steps: Number of adaptation steps

        Returns:
            Adapted model for the task
        """
        # Create task-specific model copy
        task_model = type(self.model)(**self.model.get_init_kwargs())
        task_model.load_state_dict(self.model.state_dict())

        # Adapt parameters
        optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)

        for _ in range(num_adaptation_steps):
            pred = task_model(task_data['condition'])
            loss = F.mse_loss(pred, task_data['trajectory'][:, -1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return task_model


class Reptile(nn.Module):
    """
    REPTILE meta-learning algorithm for simple and effective few-shot adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps

        self.meta_optimizer = torch.optim.SGD(self.model.parameters(), lr=meta_lr)

    def adapt_and_update(
        self,
        task_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Adapt to a task and update meta-parameters.

        Args:
            task_data: Task data for adaptation

        Returns:
            Adaptation loss
        """
        # Store original parameters
        original_params = []
        for param in self.model.parameters():
            original_params.append(param.clone())

        # Inner adaptation
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.num_inner_steps):
            pred = self.model(task_data['condition'])
            loss = F.mse_loss(pred, task_data['trajectory'][:, -1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # REPTILE update: move meta-parameters towards adapted parameters
        for original_param, adapted_param in zip(original_params, self.model.parameters()):
            original_param.data += self.meta_lr * (adapted_param.data - original_param.data)

        # Restore original parameters for next task
        for original_param, param in zip(original_params, self.model.parameters()):
            param.data.copy_(original_param.data)

        return loss


class TaskSampler:
    """
    Sampler for creating diverse meta-learning tasks.
    """

    def __init__(
        self,
        base_data: Dict[str, np.ndarray],
        num_tasks: int = 10,
        support_size: int = 50,
        query_size: int = 50,
        task_similarity_threshold: float = 0.8
    ):
        self.base_data = base_data
        self.num_tasks = num_tasks
        self.support_size = support_size
        self.query_size = query_size
        self.similarity_threshold = task_similarity_threshold

        self.tasks = self._generate_tasks()

    def _generate_tasks(self) -> List[Dict[str, np.ndarray]]:
        """Generate diverse meta-learning tasks."""
        tasks = []

        for _ in range(self.num_tasks):
            # Sample task data
            N = self.base_data['trajectories'].shape[0]
            task_indices = np.random.choice(N, size=self.support_size + self.query_size, replace=False)

            support_indices = task_indices[:self.support_size]
            query_indices = task_indices[self.support_size:]

            task = {
                'support': {
                    'trajectories': self.base_data['trajectories'][support_indices],
                    'poses': self.base_data['poses'][support_indices],
                    'pcs': self.base_data['pcs'][support_indices],
                    'tactiles': self.base_data['tactiles'][support_indices],
                    'conditions': self.base_data['conditions'][support_indices]
                },
                'query': {
                    'trajectories': self.base_data['trajectories'][query_indices],
                    'poses': self.base_data['poses'][query_indices],
                    'pcs': self.base_data['pcs'][query_indices],
                    'tactiles': self.base_data['tactiles'][query_indices],
                    'conditions': self.base_data['conditions'][query_indices]
                }
            }

            tasks.append(task)

        return tasks

    def get_task(self, task_id: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Get a specific task."""
        return self.tasks[task_id]

    def __len__(self):
        return len(self.tasks)


class MetaLearner:
    """
    High-level meta-learning trainer and evaluator.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_algorithm: str = 'maml',
        **meta_kwargs
    ):
        self.model = model
        self.meta_algorithm = meta_algorithm

        if meta_algorithm == 'maml':
            self.meta_learner = MAML(model, **meta_kwargs)
        elif meta_algorithm == 'reptile':
            self.meta_learner = Reptile(model, **meta_kwargs)
        else:
            raise ValueError(f"Unknown meta algorithm: {meta_algorithm}")

    def train_meta(
        self,
        task_sampler: TaskSampler,
        num_meta_epochs: int = 100
    ):
        """
        Train using meta-learning.

        Args:
            task_sampler: Sampler for meta-learning tasks
            num_meta_epochs: Number of meta-training epochs
        """
        for epoch in range(num_meta_epochs):
            epoch_losses = []

            for task_id in range(len(task_sampler)):
                task = task_sampler.get_task(task_id)

                # Convert to tensors
                support_data = {
                    'condition': torch.from_numpy(task['support']['conditions']).float(),
                    'trajectory': torch.from_numpy(task['support']['trajectories']).float()
                }
                query_data = {
                    'condition': torch.from_numpy(task['query']['conditions']).float(),
                    'trajectory': torch.from_numpy(task['query']['trajectories']).float()
                }

                if self.meta_algorithm == 'maml':
                    loss, _ = self.meta_learner.inner_loop(support_data, query_data, self.model)
                    epoch_losses.append(loss)
                elif self.meta_algorithm == 'reptile':
                    loss = self.meta_learner.adapt_and_update(support_data)
                    epoch_losses.append(loss)

            if self.meta_algorithm == 'maml':
                self.meta_learner.meta_update(epoch_losses)

            avg_loss = torch.stack(epoch_losses).mean().item()
            print(f"Meta epoch {epoch+1}/{num_meta_epochs}, Loss: {avg_loss:.6f}")

    def adapt_to_new_task(
        self,
        task_data: Dict[str, np.ndarray],
        num_adaptation_steps: int = 10
    ) -> nn.Module:
        """
        Adapt to a new task using few-shot learning.

        Args:
            task_data: Few-shot data for the new task

        Returns:
            Adapted model
        """
        tensor_data = {
            'condition': torch.from_numpy(task_data['conditions']).float(),
            'trajectory': torch.from_numpy(task_data['trajectories']).float()
        }

        if self.meta_algorithm == 'maml':
            adapted_model = self.meta_learner.adapt_to_task(tensor_data, num_adaptation_steps)
        else:
            # For REPTILE, adaptation is already built into training
            adapted_model = self.model

        return adapted_model