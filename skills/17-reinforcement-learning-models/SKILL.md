---
name: 17-reinforcement-learning-models
description: Policy network surgery, value function decomposition, actor-critic balance, and entropy-preserving weight adjustments.
---

# 17 Reinforcement Learning & Policy Network Surgery

## Purpose
Reinforcement Learning (PPO, SAC, TD3, Decision Transformers) models feature tight coupling between actor (policy $\pi_\theta$) and critic (value function $V_\phi$ or $Q_\phi$). Weight-space operations on RL models must maintain policy entropy and avoid catastrophic policy collapse. This skill provides rules for policy network surgery.

---

## Technical Guidelines

### 1. Actor vs. Critic Separation
- Never merge parameters between policy network heads and value estimation heads.
- In shared-backbone architectures (`features_extractor.*`), preserve early layers and isolate policy-specific adaptations to `action_net.*`.
- Logit variance in policy output layers dictates exploration entropy:
  $$H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$
  If weight magnitude scaling increases output logit scale by $2\times$, entropy drops exponentially, freezing exploration.

### 2. Polyak Averaging as Weight Merging
Standard RL utilizes Polyak target network updates:
$$\theta_{target} \leftarrow \tau \theta_{online} + (1 - \tau) \theta_{target}$$
This is mathematically equivalent to Linear Task Arithmetic with $\alpha = \tau$. When blending policies trained on different reward signals, use `merge_models.py linear` with $\alpha \in [0.1, 0.3]$.

### 3. Decision Transformer & Trajectory Models
Decision Transformers treat RL as sequence modeling over $(R, s, a)$ tokens:
- Action prediction heads require exact norm stabilization.
- Attention maps between reward-to-go tokens and action tokens can be visualized via `activation_capture.py`.

---

## Tool Execution
```bash
# Linear policy merge
python -m weight_space_programming_skill merge linear \
    models/policy_a.pt models/policy_b.pt \
    --alpha 0.25 \
    --output models/policy_ensemble.pt

# Log RL fusion event
python wsp_protocol/tools/experiment_ledger.py log \
    --event rl_policy_merge \
    --model models/policy_ensemble.pt \
    --notes "Blended exploration policy A with exploitation policy B (alpha=0.25)"
```
