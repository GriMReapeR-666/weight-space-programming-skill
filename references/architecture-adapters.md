# Architecture adapter contract

Each adapter declares: model classes and versions, source implementation, module discovery, tensor mapping, hook points, supported trainable modules, transformations, export route, known unsupported features, memory estimate and smoke tests.

Families: MLP/FFN, CNN/residual, RNN/LSTM/GRU, Transformer, MoE, SSM/recurrent, ViT, diffusion/generative, audio, multimodal, GNN, RL policy/value, scientific/PINN/operator and custom/unknown.

Unknown/custom remains discovery-only until a tested adapter exists.
