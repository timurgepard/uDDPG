Monte-Carlo version of DDPG for BipedalWalker-v3 environment (A concept)

The idea is to do 1 update by Monte-Carlo, 1 update by Temporal Difference.
The roll-out is n-steps (200), and is collected after done to have "a full length roll-out" for an each step.
If terminal reward is occured, it is divided by n_steps.

reward itself is divided by n_steps, so that Return can be normalized between -1 and 1, the tail is squashed by tanh function.

Old transitions contain Returns from old polices, it is solved by sampling with prioirty regarding history.

The paper contains various experiments and complicated, but the code here is up-to-date and simplified.
https://ieeexplore.ieee.org/document/9945743

T. Ishuov, Z. Otarbay and M. Folgheraiter, "A Concept of Unbiased Deep Deterministic Policy Gradient for Better Convergence in Bipedal Walker," 2022 International Conference on Smart Information Systems and Technologies (SIST), Nur-Sultan, Kazakhstan, 2022, pp. 1-6, doi: 10.1109/SIST54437.2022.9945743.
