"""DQN, PPO and Actor-Critic mathematics with reusable rollout primitives."""

from rl_platform.algorithms.actor_critic import (
    ActorCriticBatch,
    ActorCriticConfig,
    actor_critic_entropy,
    actor_critic_loss,
    actor_critic_policy_loss,
    actor_critic_returns,
    actor_critic_value_loss,
    compute_advantages,
    update_actor_critic_parameters,
)
from rl_platform.algorithms.dqn import (
    DQNConfig,
    EpsilonScheduler,
    bellman_target,
    dqn_loss,
    epsilon_greedy_action,
    update_dqn_parameters,
    update_target_network,
)
from rl_platform.algorithms.ppo import (
    PPOBatch,
    PPOConfig,
    clipped_policy_objective,
    compute_gae,
    compute_returns,
    entropy_bonus,
    normalize_advantages,
    ppo_loss,
    update_ppo_parameters,
    value_loss,
)
from rl_platform.algorithms.replay import ReplayBatch, ReplayBuffer, Transition
from rl_platform.algorithms.rollout import RolloutBatch, RolloutBuffer

# Distribution helpers live in networks.distributions; re-export here for
# convenience so callers that already import from rl_platform.algorithms
# do not need to change their imports.
from rl_platform.networks.distributions import (
    policy_entropy,
    policy_log_prob,
    sample_policy_action,
)

__all__ = [
    # Actor-Critic
    "ActorCriticBatch",
    "ActorCriticConfig",
    "actor_critic_entropy",
    "actor_critic_loss",
    "actor_critic_policy_loss",
    "actor_critic_returns",
    "actor_critic_value_loss",
    "compute_advantages",
    "update_actor_critic_parameters",
    # DQN
    "DQNConfig",
    "EpsilonScheduler",
    "bellman_target",
    "dqn_loss",
    "epsilon_greedy_action",
    "update_dqn_parameters",
    "update_target_network",
    # PPO
    "PPOBatch",
    "PPOConfig",
    "clipped_policy_objective",
    "compute_gae",
    "compute_returns",
    "entropy_bonus",
    "normalize_advantages",
    "ppo_loss",
    "update_ppo_parameters",
    "value_loss",
    # Replay / Rollout buffers
    "ReplayBatch",
    "ReplayBuffer",
    "RolloutBatch",
    "RolloutBuffer",
    "Transition",
    # Distribution helpers (canonical source: networks.distributions)
    "policy_entropy",
    "policy_log_prob",
    "sample_policy_action",
]
