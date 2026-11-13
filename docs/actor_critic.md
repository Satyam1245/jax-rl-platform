# Actor-Critic

Phase 8 adds a standard one-step Actor-Critic agent using the shared actor,
critic, categorical distribution, optimizer, training, evaluation, metrics,
configuration, and checkpoint layers.

The algorithm computes the bootstrapped TD return
`r + gamma * (1 - terminated) * V(next_observation)`, then uses
`return - V(observation)` as the policy-gradient advantage. The policy loss
stops gradients through the advantage, the critic uses half mean-squared value
loss, and entropy is subtracted as a configurable exploration regularizer.
Truncated time-limit transitions remain bootstrap-able.

`ActorCriticAgent` implements the common `select_action`, `observe`, and
`update` interface. It updates once per observed transition and exposes
checkpoint state containing both model/optimizer states, its RNG key, and
training progress. The existing experiment manager can therefore validate,
save, load, evaluate, and resume it without Actor-Critic-specific persistence
code.

