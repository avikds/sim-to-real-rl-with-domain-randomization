"""
Sim-to-Real RL with Domain Randomization

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - set_pendulum_mass
def set_pendulum_mass(env, mass):
    """Set a Pendulum environment's mass physics parameter in place.

    Args:
        env: Gymnasium Pendulum-v1 environment.
        mass: Positive float mass to assign.

    Returns:
        The same env with unwrapped mass updated.
    """
    env.unwrapped.m = mass
    return env

# Step 2 - set_pendulum_length
def set_pendulum_length(env, length):
    """Set a Pendulum env's rod length physics parameter and return env."""

    env.unwrapped.l = length
    return env

# Step 3 - set_pendulum_gravity
def set_pendulum_gravity(env, gravity):
    """Set a Pendulum environment's gravity physics parameter to a given value."""

    env.unwrapped.g = gravity
    return env

# Step 4 - sample_physics_config
def sample_physics_config(mass_range, length_range, gravity_range, rng):
    """Sample a physics config (mass, length, gravity) uniformly from ranges.

    Args:
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        rng: numpy.random.Generator used for all sampling.

    Returns:
        Dict with keys 'mass', 'length', 'gravity' (floats).
    """
    mass = rng.uniform(mass_range[0], mass_range[1])
    length = rng.uniform(length_range[0], length_range[1])
    gravity = rng.uniform(gravity_range[0], gravity_range[1])

    return {
        "mass": float(mass),
        "length": float(length),
        "gravity": float(gravity),
    }

# Step 5 - build_parallel_pendulum_envs
def build_parallel_pendulum_envs(n_envs, mass_range, length_range, gravity_range, seed):
    """Build parallel Pendulum-v1 envs each with its own sampled physics.

    Args:
        n_envs: Number of environments to create.
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        seed: Integer seed for the physics-sampling RNG.

    Returns:
        envs: List of n_envs Gymnasium Pendulum-v1 environments.
        configs: List of physics dicts with keys 'mass', 'length', 'gravity'.
    """
    import gymnasium as gym
    import numpy as np

    rng = np.random.default_rng(seed)

    envs = []
    configs = []

    for _ in range(n_envs):
        config = sample_physics_config(
            mass_range,
            length_range,
            gravity_range,
            rng,
        )

        env = gym.make("Pendulum-v1")
        env = set_pendulum_mass(env, config["mass"])
        env = set_pendulum_length(env, config["length"])
        env = set_pendulum_gravity(env, config["gravity"])

        envs.append(env)
        configs.append(config)

    return envs, configs

# Step 6 - shape_upright_hold_reward
def shape_upright_hold_reward(
    obs,
    base_reward,
    action,
    angle_thresh=0.2,
    angvel_thresh=0.5,
    hold_bonus=1.0,
):
    """Shape reward with a bonus for holding the pendulum upright and still.

    Args:
        obs: np.ndarray of shape (3,) or (n, 3) as [cos(theta), sin(theta), theta_dot].
        base_reward: float or np.ndarray of shape (n,) from the environment.
        action: float or np.ndarray (accepted for wrapper compatibility).
        angle_thresh: max absolute angle from upright to count as upright.
        angvel_thresh: max absolute angular velocity to count as still.
        hold_bonus: extra reward added when upright and still.

    Returns:
        Shaped reward with the same shape as base_reward.
    """
    import numpy as np

    obs = np.asarray(obs)
    base_reward = np.asarray(base_reward)

    # Recover angle from the cos(theta), sin(theta) observation components.
    theta = np.arctan2(obs[..., 1], obs[..., 0])
    theta_dot = obs[..., 2]

    # Both conditions are strictly below their respective thresholds.
    hold_mask = (
        (np.abs(theta) < angle_thresh)
        & (np.abs(theta_dot) < angvel_thresh)
    )

    shaped_reward = base_reward + hold_bonus * hold_mask

    # Preserve the shape of base_reward, including scalar input.
    if base_reward.ndim == 0:
        return float(shaped_reward)

    return shaped_reward

# Step 7 - build_actor_network
def build_actor_network(obs_dim, action_dim, hidden_dim=64):
    """Build a Gaussian actor: forward(obs) -> (mean, std).

    Store the learnable log-std as an attribute named `log_std`
    (an nn.Parameter of shape (action_dim,), initialized to zeros) --
    later steps read it via `actor.log_std`.
    """
    import torch
    import torch.nn as nn

    class ActorNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

            self.mean_head = nn.Linear(hidden_dim, action_dim)

            self.log_std = nn.Parameter(
                torch.zeros(action_dim)
            )

        def forward(self, obs):
            x = self.network(obs)
            mean = self.mean_head(x)
            std = torch.exp(self.log_std)

            return mean, std

    return ActorNetwork()

# Step 8 - build_critic_network
def build_critic_network(obs_dim, hidden_dim=64):
    """Build a critic network mapping observations to scalar state values."""

    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )

# Step 9 - sample_action_log_prob_entropy
def sample_action_log_prob_entropy(actor, obs, deterministic=False):
    """Sample actions from the Gaussian policy; return (actions, log_probs, entropy).

    Use the actor's mean output and its learnable `log_std` parameter
    (std = exp(actor.log_std)). Sum log-probs and entropy over action dims.
    """
    import torch

    # The actor's forward pass may return extra outputs; mean comes first.
    mean = actor(obs)[0]

    # Use the actor's learnable log standard deviation.
    std = torch.exp(actor.log_std)

    dist = torch.distributions.Normal(mean, std)

    if deterministic:
        actions = mean
    else:
        actions = dist.sample()

    log_probs = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    return actions, log_probs, entropy

# Step 10 - collect_rollout
def collect_rollout(envs, actor, critic, n_steps, device="cpu"):
    """Collect a fixed-length rollout from parallel envs into a trajectory dict.

    Args:
        envs: list of Gymnasium environments (already constructed).
        actor: nn.Module policy used with sample_action_log_prob_entropy.
        critic: nn.Module mapping observations to value estimates.
        n_steps: number of parallel steps to collect.
        device: torch device string for stored tensors.

    Returns:
        Dict of torch tensors with keys obs, actions, rewards, dones, values,
        log_probs (each leading dims (n_steps, n_envs, ...)), plus last_obs
        and last_dones for bootstrapping.
    """
    import numpy as np
    import torch

    n_envs = len(envs)

    if n_envs == 0:
        raise ValueError("envs must contain at least one environment.")

    # Reset every environment and collect initial observations.
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)

    obs = np.asarray(obs_list, dtype=np.float32)

    obs_buffer = []
    actions_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []
    log_probs_buffer = []

    for _ in range(n_steps):
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            actions_tensor, log_probs_tensor, _ = (
                sample_action_log_prob_entropy(actor, obs_tensor)
            )
            values_tensor = critic(obs_tensor).squeeze(-1)

        actions = actions_tensor.cpu().numpy()

        next_obs_list = []
        rewards = []
        dones = []

        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, _ = env.step(actions[i])

            done = bool(terminated or truncated)

            shaped_reward = shape_upright_hold_reward(
                next_obs,
                reward,
                actions[i],
            )

            rewards.append(float(shaped_reward))
            dones.append(float(done))

            if done:
                next_obs, _ = env.reset()

            next_obs_list.append(next_obs)

        obs_buffer.append(obs_tensor.detach())
        actions_buffer.append(actions_tensor.detach())
        rewards_buffer.append(
            torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=device,
            )
        )
        # IMPORTANT: Deep-ML expects float32 dones, not bool.
        dones_buffer.append(
            torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=device,
            )
        )
        values_buffer.append(values_tensor.detach())
        log_probs_buffer.append(log_probs_tensor.detach())

        obs = np.asarray(next_obs_list, dtype=np.float32)

    return {
        "obs": torch.stack(obs_buffer, dim=0),
        "actions": torch.stack(actions_buffer, dim=0),
        "rewards": torch.stack(rewards_buffer, dim=0),
        "dones": torch.stack(dones_buffer, dim=0),
        "values": torch.stack(values_buffer, dim=0),
        "log_probs": torch.stack(log_probs_buffer, dim=0),
        "last_obs": torch.as_tensor(
            obs, dtype=torch.float32, device=device
        ),
        "last_dones": torch.as_tensor(
            dones, dtype=torch.float32, device=device
        ),
    }

# Step 11 - rollout_observations
def rollout_observations(rollout):
    """Extract the recorded observations tensor from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs, obs_dim) stored under key 'obs'.
    """
    return rollout["obs"]

# Step 12 - rollout_actions (not yet solved)
# TODO: implement

# Step 13 - rollout_rewards (not yet solved)
# TODO: implement

# Step 14 - rollout_dones (not yet solved)
# TODO: implement

# Step 15 - rollout_values (not yet solved)
# TODO: implement

# Step 16 - rollout_log_probs (not yet solved)
# TODO: implement

# Step 17 - compute_gae (not yet solved)
# TODO: implement

# Step 18 - normalize_advantages (not yet solved)
# TODO: implement

# Step 19 - clipped_surrogate_objective (not yet solved)
# TODO: implement

# Step 20 - value_loss_and_entropy_bonus (not yet solved)
# TODO: implement

# Step 21 - ppo_loss (not yet solved)
# TODO: implement

# Step 22 - ppo_update_epoch (not yet solved)
# TODO: implement

# Step 23 - train_ppo (not yet solved)
# TODO: implement

# Step 24 - resample_envs_physics (not yet solved)
# TODO: implement

# Step 25 - evaluate_fixed_physics (not yet solved)
# TODO: implement

# Step 26 - measure_generalization_gap (not yet solved)
# TODO: implement

# Step 27 - sweep_physics_parameter (not yet solved)
# TODO: implement

# Step 28 - compare_dr_vs_fixed_policy (not yet solved)
# TODO: implement

