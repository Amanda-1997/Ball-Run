import gym
import torch
import torch.nn as nn
from buffers import RolloutBuffer


class ModelNet(nn.Module):
    """A neural network for learning the transition function"""
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 n_latent_var: int):
        super(ModelNet, self).__init__()

        state_dim = observation_space.shape[-1]
        action_dim = action_space.shape[-1]

        self.output = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, state_dim)
            )

    def forward(self, x):
        return self.output(x)


def train_model(model: ModelNet, optimiser: torch.optim.Optimizer, rollout_buffer: RolloutBuffer):
    """Train the transition model"""
    for data in rollout_buffer.get():
        optimiser.zero_grad()
        X = torch.cat([data.observations[:-1, :], data.actions[:-1, :]], dim = 1)
        Y = data.observations[1:, :]
        Y_pred = model(X)
        loss = torch.nn.functional.mse_loss(Y_pred, Y)
        loss.backward()
        optimiser.step()


class RewardNet(nn.Module):
    """A neural network for learning the instantaneous reward function"""
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 n_latent_var: int):
        super(RewardNet, self).__init__()
        state_dim = observation_space.shape[-1]
        action_dim = action_space.shape[-1]
        self.output = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.output(x)


def train_rewmodel(model: RewardNet, optimiser: torch.optim.Optimizer,
                   rollout_buffer: RolloutBuffer, fixed_cost: float):
    """Train the reward model"""
    for data in rollout_buffer.get():
        optimiser.zero_grad()
        X = torch.cat([data.observations, data.actions], dim=1)
        Y = rollout_buffer.to_torch(rollout_buffer.rewards)
        # remove the fixed costs
        Y[data.did_acts] += fixed_cost
        Y_pred = model(X)
        loss = torch.nn.functional.mse_loss(Y_pred, Y)
        loss.backward()
        optimiser.step()
