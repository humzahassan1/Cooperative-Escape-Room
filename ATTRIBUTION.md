# PPO reference

The PPO implementation is independently written for this two-agent environment, following the algorithm organization and implementation guidance in CleanRL:

- https://docs.cleanrl.dev/rl-algorithms/ppo/
- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

CleanRL is by Shengyi Huang and contributors and is distributed under the MIT license. The local implementation uses separate policies, factorized movement/message distributions, synchronous joint environment steps, and explicit timeout bootstrapping. It does not depend on the CleanRL package.
