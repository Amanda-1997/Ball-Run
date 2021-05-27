from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.utils import explained_variance, update_learning_rate, get_device
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer

from test_env import TestSafetyGym


FIXED_COST = 0.1
N_EPOCHS = 10
BATCH_SIZE = 100
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


def clip_range_fn(_: float) -> float:
    return 0.2


def lr_schedule_fn(_: float) -> float:
    return 1e-3


def train(policy: ActorCriticPolicy, rollout_buffer: RolloutBuffer, progress: float) -> None:
    """
    Update policy using the currently gathered rollout buffer.
    """
    # Update optimizer learning rate
    lr = lr_schedule_fn(progress)
    logger.record("train/learning_rate", lr)
    update_learning_rate(policy.optimizer, lr)

    # Compute current clip range
    clip_range = clip_range_fn(progress)

    entropy_losses, all_kl_divs = [], []
    pg_losses, value_losses = [], []
    clip_fractions = []

    # train for n_epochs epochs
    for _ in range(N_EPOCHS):
        approx_kl_divs = []
        # Do a complete pass on the rollout buffer
        for rollout_data in rollout_buffer.get(BATCH_SIZE):
            actions = rollout_data.actions

            values, log_prob, entropy = policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            values_pred = values
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + ENT_COEF * entropy_loss + VF_COEF * value_loss


            # Optimization step
            policy.optimizer.zero_grad()
            loss.backward()
            for param in policy.parameters():
                if torch.isnan(param.grad).any():
                    print("NAN!!!!!!!", param)
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            policy.optimizer.step()
            approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

        all_kl_divs.append(np.mean(approx_kl_divs))

    explained_var = explained_variance(rollout_buffer.values.flatten(),
                                       rollout_buffer.returns.flatten())

    # Logs
    logger.record("train/entropy_loss", np.mean(entropy_losses))
    logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    logger.record("train/value_loss", np.mean(value_losses))
    logger.record("train/approx_kl", np.mean(approx_kl_divs))
    logger.record("train/clip_fraction", np.mean(clip_fractions))
    logger.record("train/loss", loss.item())
    logger.record("train/explained_variance", explained_var)
    if hasattr(policy, "log_std"):
        logger.record("train/std", torch.exp(policy.log_std).mean().item())

    logger.record("train/clip_range", clip_range)
    logger.dump()


def main():
    env = TestSafetyGym()
    total_episodes = 10000
    n_steps_learn = 100

    n_steps = 1
    p1_rollout_buffer = RolloutBuffer(
        buffer_size=BATCH_SIZE, observation_space=env.observation_space,
        action_space=env.action_space)
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule_fn,
        net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
        activation_fn=torch.nn.ReLU,
    )

    device = get_device("cpu")
    ep_rewards = deque(maxlen=100)
    ep_timesteps = deque(maxlen=100)
    for i in range(total_episodes):
        obs = env.reset()
        done = False
        last_done = True
        ep_reward = 0
        ep_timestep = 0
        while not done:
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor([obs]).to(device)
                actions_1, values_1, log_prob_1  = policy.forward(obs_tensor)
                actions = actions_1.cpu().numpy()
            # Clip the actions to avoid out of bound error
            clipped_actions = np.clip(actions, env.action_space.low, env.action_space.high)

            new_obs, reward, done, infos = env.step(clipped_actions)

            p1_rollout_buffer.add(obs, actions, reward, last_done, values_1, log_prob_1)
            obs = new_obs
            last_done = done
            ep_reward += reward
            ep_timestep += 1

            if n_steps >= n_steps_learn:
                with torch.no_grad():
                    # Compute action and value for the last timestep
                    obs_tensor = torch.as_tensor([obs]).to(device)
                    actions_1, values_1, log_prob_1  = policy.forward(obs_tensor)
                    p1_rollout_buffer.compute_returns_and_advantage(
                        last_values=values_1, dones=done)
                current_progess_remaining = 1 - i / total_episodes
                train(policy, p1_rollout_buffer, current_progess_remaining)
                p1_rollout_buffer.reset()
                n_steps = 0
    
            n_steps += 1
        ep_rewards.append(ep_reward)
        ep_timesteps.append(ep_timestep)
        print(np.mean(ep_timesteps), np.mean(ep_rewards))


if __name__ == """__main__""":
    main()
