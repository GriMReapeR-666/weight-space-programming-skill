# Weight-space mathematical methods

Use methods only when their assumptions fit the architecture and objective.

- Low-rank update: W' = W + BA, with dimensions checked.
- Gradient update: theta_(t+1) = theta_t - eta * optimizer_update(grad L).
- Interpolation: theta' = (1-a)theta_A + a theta_B, only with parameter correspondence.
- Relative quantization error: ||W-Q(W)|| / (||W||+epsilon), where W and Q(W) are comparable dequantized tensors.
- RMS: sqrt(mean(W^2)).
- Spectral norm: largest singular value, useful for selected linear operators.
- Sensitivity: estimate with gradients, Fisher approximations, ablation or controlled perturbations.

These statistics do not independently measure intelligence or prove a defect.
