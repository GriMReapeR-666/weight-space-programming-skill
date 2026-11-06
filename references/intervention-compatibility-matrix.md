# Intervention compatibility matrix (screening guide)

Legend: **Common** = often applicable after checks; **Conditional** = architecture/objective-specific; **Not generic** = do not apply without a dedicated method. This is a screening guide, never permission to mutate.

| Model family | Fine-tune / PEFT | Prune / quantize | Merge / interpolate | Distill | Special invariants |
|---|---|---|---|---|---|
| MLP / CNN / ResNet | Common | Common, structured options | Conditional | Common | channel dimensions, BN buffers, residual compatibility |
| RNN / LSTM / GRU | Common | Conditional | Conditional | Common | gate ordering, recurrent stability, hidden-state semantics |
| Transformer / ViT | Common; PEFT conditional | Common with calibration | Conditional, lineage required | Common | head dimensions, tied embeddings, positional schemes, KV cache |
| MoE / routed | Common, routing-aware | Conditional | Highly conditional | Common | expert indexing, router load balance, capacity and dispatch |
| SSM / hybrid recurrent | Common if implementation supports | Conditional | Highly conditional | Common | discretization, state dimensions, convolution/scan kernels, recurrence |
| Diffusion / flow / score | Common | Conditional | Conditional | Common | noise/time schedule, parameterization, EMA weights, sampler parity |
| GAN / VAE / flow density | Common | Conditional | Conditional | Common | coupled networks, latent prior, invertibility/Jacobian constraints |
| Audio / speech / codec | Common | Conditional | Conditional | Common | sample rate, frontend, alignment, codec/token rates, streaming state |
| Multimodal / VLM | Common | Conditional | Highly conditional | Common | modality bridges, tokenizer/processor, resolution and alignment |
| GNN / geometric | Common | Conditional | Conditional | Common | permutation/equivariance, graph batching, edge/node semantics |
| RL policy / value / world model | Objective-specific | Conditional | Conditional | Common | action bounds, reward scale, policy distribution, environment version |
| Forecasting / time-series | Common | Conditional | Conditional | Common | chronology, leakage, covariate availability, horizon and calibration |
| PINN / neural operator | Objective-specific | Conditional | Conditional | Common | PDE constraints, units, boundary conditions, conservation and mesh/grid |
| Spiking / neuromorphic | Specialized | Specialized | Not generic | Conditional | temporal coding, thresholds, membrane state, event semantics |
| Tree / boosting / SVM / GLM | Model-specific fitting | Not neural quantization | Not generic | Conditional | tree topology, split thresholds, kernel/support-vector semantics |
| Retrieval / embedding index | Re-embed or train encoder | Storage-specific | Not generic | Conditional | index rebuild, vector normalization, metric and corpus version |

## Mandatory preflight
1. Identify the actual parameter semantics and architecture invariants.
2. Confirm operation is implemented for that exact class/version.
3. Verify data, calibration and objective requirements.
4. Estimate compute, memory, and quality risks.
5. Run a small reversible pilot.
6. Evaluate task quality, robustness, calibration and runtime parity.

## Prohibited shortcuts
- No blind global scaling, random perturbation, clipping, spectral truncation or zeroing.
- No merge based solely on equal tensor shapes.
- No quantization claim based only on file size or nominal bit width.
- No pruning claim without measuring sparse-kernel/runtime benefit.
- No RL-policy edit without environment-level evaluation.
- No scientific-model edit without physical/unit/constraint validation.
