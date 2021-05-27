from collections import deque

import gym
import bullet_safety_gym
import numpy as np
import torch
from torch.nn import functional as F
import random
from stable_baselines3.common import logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, update_learning_rate, get_device
from stable_baselines3.common.type_aliases import Schedule

# from test_env import TestSafetyGym
from buffers import RolloutBuffer
from models import ModelNet, train_model, RewardNet, train_rewmodel


from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
base_dir = "./logs/AC_safety_model_ball/"

device = get_device("cpu")
FIXED_COST = 0.3
N_EPOCHS = 10
BATCH_SIZE = 2000
ENT_COEF = 0.
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
GAMMA = 1.0

total_timesteps = 1e7


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(5)

def clip_range_fn(_: float) -> float:
    return 0.2


def lr_schedule_fn_1(progress: float) -> float:
    return 1e-3


def lr_schedule_fn_2(progress: float) -> float:
    return 5e-4


def train(policy: ActorCriticPolicy, rollout_buffer: RolloutBuffer, progress: float,
          lr_schedule_fn: Schedule) -> None:
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
            ratio = torch.exp(rollout_data.did_acts * (log_prob - rollout_data.old_log_prob))

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio * rollout_data.did_acts
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
    # logger.make_output_format('tensorboard', './logs/ppo_newest/')
    #logger.dump()

    return loss.item(), np.mean(entropy_losses), np.mean(pg_losses), np.mean(value_losses)


def get_action(obs: np.ndarray, p1_policy: ActorCriticPolicy, p2_policy: ActorCriticPolicy,
               env: gym.Env, model: ModelNet, rewmodel: RewardNet):
    obs_tensor = torch.as_tensor([obs]).to(device)

    # get player 1's proposed action
    actions_1, values_1, log_prob_1  = p1_policy.forward(obs_tensor)
    # get the predicted next state and cost
    #actions_1_np = actions_1.cpu().numpy()
    #pred_obs, _, _, pred_cost = env.pure_step(obs, actions_1_np)
    stacked = torch.Tensor(np.hstack([obs_tensor, actions_1]))
    pred_obs = model.forward(stacked)
    pred_cost = rewmodel.forward(stacked)
    pred_obs_tensor = torch.as_tensor(pred_obs).to(device)

    # evaluate its expected cost return
    _, latent_vf, _ = p2_policy._get_latent(pred_obs_tensor)
    q_1 = pred_cost + GAMMA * p2_policy.value_net(latent_vf)

    # get player 2's proposed action
    actions_2, _, _ = p2_policy.forward(obs_tensor, deterministic=False)
    # get the predicted next state and cost
    #actions_2_np = actions_2.cpu().numpy()
    #pred_obs, _, _, pred_cost = env.pure_step(obs, actions_2_np)
    stacked = torch.Tensor(np.hstack([obs_tensor, actions_2]))
    pred_obs = model.forward(stacked)
    pred_cost = rewmodel.forward(stacked)

    pred_obs_tensor = torch.as_tensor(pred_obs).to(device)
    # evaluate its expected cost return
    _, latent_vf, _ = p2_policy._get_latent(pred_obs_tensor)
    q_2 = pred_cost - FIXED_COST + GAMMA * p2_policy.value_net(latent_vf)

    # determine who should act
    intervene = q_2.cpu().numpy()[0, 0] >= q_1.cpu().numpy()[0, 0]
    actions = actions_2 if intervene else actions_1

    # evaluate the probability of each action
    values_1, log_prob_1, _ = p1_policy.evaluate_actions(obs_tensor, actions)
    values_2, log_prob_2, _ = p2_policy.evaluate_actions(obs_tensor, actions)
    return intervene, actions.cpu().numpy()[0, :], values_1, log_prob_1, values_2, log_prob_2


def main():
    ### add tensorboard
    cur_time = datetime.now() + timedelta(hours=0)
    log_dir = base_dir + cur_time.strftime("[%m-%d]%H.%M.%S")
    writer = SummaryWriter(logdir=log_dir)

    ENV_NAME = "SafetyBallRun-v0"
    #env = TestSafetyGym()
    env = gym.make(ENV_NAME)
    total_episodes = 20000
    n_steps_learn = BATCH_SIZE

    p1_rollout_buffer = RolloutBuffer(
        buffer_size=BATCH_SIZE, observation_space=env.observation_space,
        action_space=env.action_space, gamma=GAMMA)

    p2_rollout_buffer = RolloutBuffer(
        buffer_size=BATCH_SIZE, observation_space=env.observation_space,
        action_space=env.action_space, gamma=GAMMA)

    p1_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule_fn_1,
        net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
        activation_fn=torch.nn.ReLU,
    )

    p2_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule_fn_2,
        net_arch=[{"pi": [64, 64], "vf": [64, 64]}],
        activation_fn=torch.nn.ReLU,
    )

    model = ModelNet(observation_space=env.observation_space,
                     action_space=env.action_space, n_latent_var=128)
    model.to(device)
    model_optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)

    rewmodel = RewardNet (observation_space=env.observation_space,
                          action_space=env.action_space, n_latent_var=128)
    rewmodel_optimiser = torch.optim.Adam(rewmodel.parameters(), lr=2e-3)

    n_steps = 1
    ep_rewards = deque(maxlen=100)
    ep_timesteps = deque(maxlen=100)
    ep_costs = deque(maxlen=100)
    ep_losses = deque(maxlen=100)
    ep_entropy_loss= deque(maxlen=100)
    ep_pg_loss= deque(maxlen=100)
    ep_value_loss= deque(maxlen=100)
    ep_intervenue= deque(maxlen=100)

    # for i in range(total_episodes):
    n_epi = 0
    num_timesteps = 0
    while num_timesteps < total_timesteps:
        obs = env.reset()
        done = False
        last_done = True
        ep_reward = 0
        ep_timestep = 0
        ep_cost = 0
        intervene_count = 0

        losses_1, entropy_losses_1, pg_losses_1, value_losses_1 = [0], [0], [0], [0]
        while not done:
            with torch.no_grad():
                # Convert to pytorch tensor
                intervene, actions, values_1, log_prob_1, values_2, log_prob_2 = get_action(
                    obs, p1_policy, p2_policy, env, model, rewmodel)
            # Clip the actions to avoid out of bound error
            #clipped_actions = np.clip(actions, env.action_space.low, env.action_space.high)
            if intervene:
                intervene_count += 1

            new_obs, reward, done, info = env.step(actions)

            cost = info['cost'] + FIXED_COST if intervene else info['cost']
            p1_rollout_buffer.add(obs, actions, reward, last_done, values_1, log_prob_1,
                                  not intervene)
            p2_rollout_buffer.add(obs, actions, -cost, last_done, values_2, log_prob_2, intervene)

            obs = new_obs
            last_done = done
            ep_reward += reward
            ep_cost += info['cost']
            ep_timestep += 1

            if n_steps >= n_steps_learn:
                with torch.no_grad():
                    # Compute action and value for the last timestep
                    _, _, values_1, _, values_2, _ = get_action(
                        obs, p1_policy, p2_policy, env, model, rewmodel)
                    p1_rollout_buffer.compute_returns_and_advantage(last_values=values_1, dones=done)
                    p2_rollout_buffer.compute_returns_and_advantage(last_values=values_2, dones=done)

                current_progess_remaining = 1 - n_epi / total_episodes
                train_model(model, model_optimiser, p1_rollout_buffer)
                train_rewmodel(rewmodel, rewmodel_optimiser, p2_rollout_buffer, FIXED_COST)
                loss_1, entropy_loss_1, pg_loss_1, value_loss_1 = train(p1_policy, p1_rollout_buffer, current_progess_remaining, lr_schedule_fn_1)
                losses_1.append(loss_1)
                entropy_losses_1.append(entropy_loss_1)
                pg_losses_1.append(pg_loss_1)
                value_losses_1.append(value_loss_1)

                loss_2, entropy_loss_2, pg_loss_2, value_loss_2 = train(p2_policy, p2_rollout_buffer, current_progess_remaining, lr_schedule_fn_2)
                # train(p1_policy, p1_rollout_buffer, current_progess_remaining, lr_schedule_fn_1)
                # train(p2_policy, p2_rollout_buffer, current_progess_remaining, lr_schedule_fn_2)
                p1_rollout_buffer.reset()
                p2_rollout_buffer.reset()
                n_steps = 0
    
            n_steps += 1
            num_timesteps += 1

        n_epi += 1
        ep_intervenue.append(intervene_count)
        ep_rewards.append(ep_reward)
        ep_timesteps.append(ep_timestep)
        ep_costs.append(ep_cost)
        ep_losses.append(np.mean(losses_1))
        ep_entropy_loss.append(np.mean(entropy_losses_1))
        ep_pg_loss.append(np.mean(pg_losses_1))
        ep_value_loss.append(np.mean(value_losses_1))
        print(num_timesteps, np.mean(ep_timesteps), np.mean(ep_costs), np.mean(ep_rewards))

        ### save for every epoch
        writer.add_scalar('logs/ep_costs', np.mean(ep_costs), num_timesteps)
        writer.add_scalar('logs/ep_rewards', np.mean(ep_rewards), num_timesteps)
        writer.add_scalar('logs/loss_player1', np.mean(ep_losses), num_timesteps)
        writer.add_scalar('logs/entropy_loss_player1', np.mean(ep_entropy_loss), num_timesteps)
        writer.add_scalar('logs/pg_loss_player1', np.mean(ep_pg_loss), num_timesteps)
        writer.add_scalar('logs/value_loss_player1', np.mean(ep_value_loss), num_timesteps)
        writer.add_scalar('logs/intervenue', np.mean(ep_intervenue), num_timesteps)

    # post training eval loop
    obs = env.reset()
    done = False
    while not done:
        # post-learning evaluation loop
        with torch.no_grad():
            print(obs)
            intervene, action, *_ = get_action(obs, p1_policy, p2_policy, env, model, rewmodel)

        if intervene:
            print('intervene')
        obs, _, done, info = env.step(action)
        #env.render()


if __name__ == """__main__""":
    for i in range(0, 5, 1):
        main()
