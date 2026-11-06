---
name: 07-gradient-optimization
description: Gradient-free search (CMA-ES, coordinate descent), targeted micro-gradients, and gradient surgery for weight-space programming.
---

# 07 Gradient Optimization & Gradient-Free Search

## Purpose
While traditional deep learning relies on backpropagating loss through all layers, weight-space programming frequently operates under constraints where full backpropagation is impossible, too expensive, or non-differentiable (discrete evaluations, hard safety gates). This skill details targeted micro-gradient updates and zero-order search techniques.

---

## Algorithms

### 1. Zero-Order Coordinate Search / Directional Perturbation
When computing gradients is infeasible, estimate gradient via directional derivatives:
$$g_i \approx \frac{\mathcal{L}(\theta + \epsilon \cdot v_i) - \mathcal{L}(\theta - \epsilon \cdot v_i)}{2\epsilon}$$
Where $v_i$ is a random orthogonal perturbation vector in weight slice space.

### 2. Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
For micro-edits in low-dimensional weight subspaces ($d \le 256$ parameters, such as a single adapter vector or steering projection):
1. Sample $\lambda$ candidate weight perturbations: $\Delta W_k \sim \mathcal{N}(m, \sigma^2 C)$.
2. Evaluate candidates against the objective and rank by score.
3. Update mean $m$ and covariance matrix $C$ using top $\mu$ performers.

```python
import numpy as np

def coordinate_search_step(eval_fn, current_weights, step_size=0.01, n_samples=8):
    best_weights = current_weights.copy()
    best_score = eval_fn(best_weights)
    
    for _ in range(n_samples):
        direction = np.random.randn(*current_weights.shape)
        direction /= (np.linalg.norm(direction) + 1e-8)
        
        candidate = current_weights + step_size * direction
        score = eval_fn(candidate)
        if score > best_score:
            best_score = score
            best_weights = candidate
            
    return best_weights, best_score
```

### 3. Gradient Surgery (PCGrad - Projecting Conflicting Gradients)
When optimizing for a new objective without violating a secondary constraint:
If $g_1 \cdot g_2 < 0$, project $g_1$ onto the normal plane of $g_2$:
$$g_1 \leftarrow g_1 - \frac{g_1 \cdot g_2}{\|g_2\|^2} g_2$$

---

## Tool Protocol
Record all optimization trajectories into `experiment_ledger.py`:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event search_step \
    --model models/target.pt \
    --notes "CMA-ES iteration 4: score improved 0.81 -> 0.87"
```
