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

# Step 12 - rollout_actions
def rollout_actions(rollout):
    """Extract the recorded actions tensor from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs, action_dim).
    """
    return rollout["actions"]

# Step 13 - rollout_rewards
def rollout_rewards(rollout):
    """Extract the recorded rewards tensor from a collected rollout.

    Args:
        rollout: dict returned by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'rewards'.
    """
    return rollout["rewards"]

# Step 14 - rollout_dones
def rollout_dones(rollout):
    """Extract the recorded episode-termination flags from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'dones'.
    """
    return rollout["dones"]

# Step 15 - rollout_values
def rollout_values(rollout):
    """Extract the recorded critic value estimates from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout, containing a 'values' tensor.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) with critic value estimates.
    """
    return rollout["values"]

# Step 16 - rollout_log_probs
def rollout_log_probs(rollout):
    """Extract the recorded action log-probabilities from a collected rollout.

    Args:
        rollout: dict returned by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'log_probs'.
    """
    return rollout["log_probs"]

# Step 17 - compute_gae
def compute_gae(
    rewards,
    values,
    dones,
    last_values,
    last_dones,
    gamma=0.99,
    lam=0.95,
):
    """Compute GAE advantages and value targets from a rollout.

    Args:
        rewards: Tensor (T, N) of per-step rewards.
        values: Tensor (T, N) of critic values V(s_t).
        dones: Tensor (T, N) of episode-termination flags.
        last_values: Tensor (N,) bootstrap values after the final step.
        last_dones: Tensor (N,) done flags after the final step.
        gamma: Discount factor (default 0.99).
        lam: GAE lambda (default 0.95).

    Returns:
        advantages: Tensor (T, N).
        returns: Tensor (T, N), equal to advantages + values.
    """
    import torch

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    gae = torch.zeros_like(last_values)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
            next_dones = last_dones
        else:
            next_values = values[t + 1]
            next_dones = dones[t + 1]

        # Do not bootstrap across episode boundaries.
        nonterminal = 1.0 - next_dones

        delta = (
            rewards[t]
            + gamma * next_values * nonterminal
            - values[t]
        )

        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values

    return advantages, returns

# Step 18 - normalize_advantages
def normalize_advantages(advantages, eps=1e-8):
    """Normalize advantages to zero mean and unit standard deviation."""

    mean = advantages.mean()
    std = advantages.std()

    return (advantages - mean) / (std + eps)

# Step 19 - clipped_surrogate_objective
def clipped_surrogate_objective(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_eps=0.2,
):
    """Compute the PPO clipped surrogate policy objective from log-probs and advantages."""

    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    return -torch.mean(torch.minimum(unclipped, clipped))

# Step 20 - value_loss_and_entropy_bonus
def value_loss_and_entropy_bonus(
    values_pred,
    value_targets,
    entropy,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """Compute value-function loss and entropy bonus for PPO.

    Args:
        values_pred: Predicted state values, shape (batch,).
        value_targets: Target returns, shape (batch,).
        entropy: Per-sample entropy (batch,) or a scalar mean.
        value_coef: Scale on the MSE value loss (default 0.5).
        entropy_coef: Scale on mean entropy (default 0.01).

    Returns:
        (value_loss, entropy_bonus) as scalar torch.Tensors.
    """
    value_loss = value_coef * torch.mean(
        (values_pred - value_targets) ** 2
    )

    entropy_bonus = entropy_coef * torch.mean(entropy)

    return value_loss, entropy_bonus

# Step 21 - ppo_loss
def ppo_loss(policy_loss, value_loss, entropy_bonus):
    """Combine clipped surrogate, value loss, and entropy bonus into one PPO loss.

    Args:
        policy_loss: Scalar tensor from the clipped surrogate objective.
        value_loss: Scalar tensor value-function loss.
        entropy_bonus: Scalar tensor entropy bonus term.

    Returns:
        Scalar torch.Tensor total_loss = policy_loss + value_loss - entropy_bonus.
    """
    return policy_loss + value_loss - entropy_bonus

# Step 22 - ppo_update_epoch
def ppo_update_epoch(
    actor,
    critic,
    optimizer,
    rollout,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    minibatch_size=64,
):
    """Run one PPO update epoch over shuffled minibatches with gradient clipping.

    Args:
        actor: Gaussian policy module (forward -> means, has log_std param).
        critic: Value module (forward -> state values).
        optimizer: Optimizer over actor and critic parameters.
        rollout: Dict with 'observations', 'actions', 'log_probs'.
        advantages: Tensor (T, N) or (T*N,).
        returns: Tensor (T, N) or (T*N,), value targets.
        clip_eps: PPO clip range (default 0.2).
        value_coef: Value loss coefficient (default 0.5).
        entropy_coef: Entropy bonus coefficient (default 0.01).
        max_grad_norm: Gradient clip norm (default 0.5).
        minibatch_size: Minibatch size (default 64).

    Returns:
        Dict of mean floats: policy_loss, value_loss, entropy, total_loss.
    """
    import torch

    # Flatten rollout tensors from (T, N, ...) to (T*N, ...).
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["log_probs"]

    observations = observations.reshape(-1, observations.shape[-1])
    actions = actions.reshape(observations.shape[0], -1)
    old_log_probs = old_log_probs.reshape(-1)

    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)

    # Normalize the full advantage batch before creating minibatches.
    advantages = normalize_advantages(advantages)

    batch_size = observations.shape[0]

    if not (
        actions.shape[0]
        == old_log_probs.shape[0]
        == advantages.shape[0]
        == returns.shape[0]
        == batch_size
    ):
        raise ValueError("Rollout and target tensors must have matching batch sizes.")

    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")

    # Shuffle the complete batch once for this PPO epoch.
    indices = torch.randperm(batch_size, device=observations.device)

    policy_losses = []
    value_losses = []
    entropies = []
    total_losses = []

    actor.train()
    critic.train()

    for start in range(0, batch_size, minibatch_size):
        mb_idx = indices[start:start + minibatch_size]

        mb_obs = observations[mb_idx]
        mb_actions = actions[mb_idx]
        mb_old_log_probs = old_log_probs[mb_idx]
        mb_advantages = advantages[mb_idx]
        mb_returns = returns[mb_idx]

        # Actor forward pass: mean is the first output.
        mean = actor(mb_obs)[0]
        std = torch.exp(actor.log_std)

        dist = torch.distributions.Normal(mean, std)

        new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        policy_loss = clipped_surrogate_objective(
            new_log_probs,
            mb_old_log_probs,
            mb_advantages,
            clip_eps=clip_eps,
        )

        values_pred = critic(mb_obs).squeeze(-1)

        value_loss, entropy_bonus = value_loss_and_entropy_bonus(
            values_pred,
            mb_returns,
            entropy,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        total_loss = ppo_loss(
            policy_loss,
            value_loss,
            entropy_bonus,
        )

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_grad_norm,
        )

        optimizer.step()

        policy_losses.append(policy_loss.detach().item())
        value_losses.append(value_loss.detach().item())
        entropies.append(entropy.mean().detach().item())
        total_losses.append(total_loss.detach().item())

    return {
        "policy_loss": float(sum(policy_losses) / len(policy_losses)),
        "value_loss": float(sum(value_losses) / len(value_losses)),
        "entropy": float(sum(entropies) / len(entropies)),
        "total_loss": float(sum(total_losses) / len(total_losses)),
    }

# Step 23 - train_ppo
def train_ppo(
    actor,
    critic,
    optimizer,
    envs,
    n_iters,
    n_steps,
    n_epochs,
    minibatch_size=64,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    mass_range=None,
    length_range=None,
    gravity_range=None,
    seed=0,
):
    """Train PPO for n_iters collect-update cycles with optional domain randomization.

    Args:
        actor: Gaussian policy network.
        critic: Value network.
        optimizer: Optimizer over actor and critic parameters.
        envs: List of Gymnasium environments.
        n_iters: Number of collect-update iterations.
        n_steps: Rollout horizon per iteration.
        n_epochs: PPO epochs per iteration.
        minibatch_size: Minibatch size for PPO updates.
        gamma: Discount factor.
        lam: GAE lambda.
        clip_eps: PPO clip range.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Gradient clip norm.
        mass_range: Optional (min, max) mass for domain randomization.
        length_range: Optional (min, max) length for domain randomization.
        gravity_range: Optional (min, max) gravity for domain randomization.
        seed: Random seed.

    Returns:
        Dict with 'returns_history': list of mean rollout rewards per iteration.
        Actor and critic are updated in place.
    """
    import numpy as np
    import torch

    returns_history = []
    rng = np.random.default_rng(seed)

    # Domain randomization is enabled only when all three ranges are supplied.
    randomize = (
        mass_range is not None
        and length_range is not None
        and gravity_range is not None
    )

    for _ in range(n_iters):
        # Sample and apply fresh physics to every environment before
        # collecting this iteration's rollout.
        if randomize:
            for env in envs:
                config = sample_physics_config(
                    mass_range,
                    length_range,
                    gravity_range,
                    rng,
                )
                set_pendulum_mass(env, config["mass"])
                set_pendulum_length(env, config["length"])
                set_pendulum_gravity(env, config["gravity"])

        rollout = collect_rollout(
            envs,
            actor,
            critic,
            n_steps=n_steps,
        )

        # Compute the value estimate for the state following the final step.
        with torch.no_grad():
            last_values = critic(rollout["last_obs"]).squeeze(-1)

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_values,
            rollout["last_dones"],
            gamma=gamma,
            lam=lam,
        )

        # ppo_update_epoch expects the rollout observation key to be
        # 'observations', while collect_rollout stores it as 'obs'.
        update_rollout = {
            "observations": rollout["obs"],
            "actions": rollout["actions"],
            "log_probs": rollout["log_probs"],
        }

        for _ in range(n_epochs):
            ppo_update_epoch(
                actor,
                critic,
                optimizer,
                update_rollout,
                advantages,
                returns,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                minibatch_size=minibatch_size,
            )

        # Mean shaped reward over the rollout.
        returns_history.append(
            float(rollout["rewards"].mean().item())
        )

    return {
        "returns_history": returns_history,
    }

# Step 24 - resample_envs_physics
def resample_envs_physics(envs, mass_range, length_range, gravity_range, rng):
    """Resample physics for every env and return the applied configs.

    Args:
        envs: List of Gymnasium Pendulum-v1 environments.
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        rng: numpy.random.Generator used for all sampling.

    Returns:
        List of dicts with keys 'mass', 'length', 'gravity', one per env,
        in the same order as `envs`.
    """
    configs = []

    for env in envs:
        config = sample_physics_config(
            mass_range,
            length_range,
            gravity_range,
            rng,
        )

        set_pendulum_mass(env, config["mass"])
        set_pendulum_length(env, config["length"])
        set_pendulum_gravity(env, config["gravity"])

        configs.append(config)

    return configs

# Step 25 - evaluate_fixed_physics
def evaluate_fixed_physics(
    actor,
    mass,
    length,
    gravity,
    n_episodes=5,
    seed=0,
    max_steps=200,
):
    """Measure mean episodic return on one fixed Pendulum physics config.

    Args:
        actor: Trained actor network (Gaussian policy).
        mass: Pendulum mass to evaluate under.
        length: Rod length to evaluate under.
        gravity: Gravity to evaluate under.
        n_episodes: Number of evaluation episodes.
        seed: Base seed; episode i uses seed + i.
        max_steps: Max steps per episode before stopping.

    Returns:
        Mean episodic return as a Python float.
    """
    import gymnasium as gym
    import numpy as np
    import torch

    env = gym.make("Pendulum-v1")

    set_pendulum_mass(env, mass)
    set_pendulum_length(env, length)
    set_pendulum_gravity(env, gravity)

    episode_returns = []

    actor.eval()

    with torch.no_grad():
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed + i)
            episode_return = 0.0

            for _ in range(max_steps):
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                ).reshape(1, -1)

                actions, _, _ = sample_action_log_prob_entropy(
                    actor,
                    obs_tensor,
                    deterministic=True,
                )

                # Support both batched and scalar action outputs.
                action = np.asarray(actions.detach().cpu().numpy()).reshape(-1)

                obs, reward, terminated, truncated, _ = env.step(action)

                episode_return += float(reward)

                if terminated or truncated:
                    break

            episode_returns.append(episode_return)

    env.close()

    return float(np.mean(episode_returns))

# Step 26 - measure_generalization_gap
def measure_generalization_gap(
    actor,
    train_ranges,
    heldout_ranges,
    n_configs=5,
    n_episodes=3,
    seed=0,
):
    """Measure in-dist vs held-out returns and the generalization gap.

    Args:
        actor: Trained actor network (Gaussian policy).
        train_ranges: Dict with keys 'mass', 'length', 'gravity' -> (min, max).
        heldout_ranges: Same structure as train_ranges, held-out box.
        n_configs: Number of physics configs to sample per side.
        n_episodes: Episodes per config for evaluate_fixed_physics.
        seed: Seed for config sampling and evaluation.

    Returns:
        Dict with float keys 'in_dist_return', 'heldout_return', 'gap'
        where gap = in_dist_return - heldout_return.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    def sample_configs(ranges):
        configs = []

        for _ in range(n_configs):
            config = sample_physics_config(
                ranges["mass"],
                ranges["length"],
                ranges["gravity"],
                rng,
            )
            configs.append(config)

        return configs

    train_configs = sample_configs(train_ranges)
    heldout_configs = sample_configs(heldout_ranges)

    train_returns = []
    heldout_returns = []

    for config in train_configs:
        result = evaluate_fixed_physics(
            actor,
            config["mass"],
            config["length"],
            config["gravity"],
            n_episodes=n_episodes,
            seed=seed,
        )
        train_returns.append(float(result))

    for config in heldout_configs:
        result = evaluate_fixed_physics(
            actor,
            config["mass"],
            config["length"],
            config["gravity"],
            n_episodes=n_episodes,
            seed=seed,
        )
        heldout_returns.append(float(result))

    in_dist_return = float(np.mean(train_returns))
    heldout_return = float(np.mean(heldout_returns))
    gap = float(in_dist_return - heldout_return)

    return {
        "in_dist_return": in_dist_return,
        "heldout_return": heldout_return,
        "gap": gap,
    }

# Step 27 - sweep_physics_parameter (not yet solved)
# TODO: implement

# Step 28 - compare_dr_vs_fixed_policy (not yet solved)
# TODO: implement

