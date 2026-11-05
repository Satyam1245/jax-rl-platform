"""Reusable JAX/Flax neural-network foundations."""

from rl_platform.networks.mlp import (
    MLP,
    MLPConfig,
    Activation,
    batched_mlp_forward,
    get_activation,
    initialize_mlp,
    mlp_forward,
)
from rl_platform.networks.optimization import (
    create_optimizer,
    create_train_state,
    optimizer_step,
)
from rl_platform.networks.q_network import QNetwork, QNetworkConfig
from rl_platform.networks.policy import Actor, Critic, PolicyNetworkConfig
from rl_platform.networks.distributions import policy_entropy, policy_log_prob, sample_policy_action

__all__ = [
    "Activation",
    "MLP",
    "MLPConfig",
    "QNetwork",
    "QNetworkConfig",
    "Actor",
    "Critic",
    "PolicyNetworkConfig",
    "policy_entropy",
    "policy_log_prob",
    "sample_policy_action",
    "batched_mlp_forward",
    "create_optimizer",
    "create_train_state",
    "get_activation",
    "initialize_mlp",
    "mlp_forward",
    "optimizer_step",
]
