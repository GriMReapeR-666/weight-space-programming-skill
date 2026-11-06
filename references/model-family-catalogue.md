# Model-family and architecture coverage catalogue

This catalogue is a discovery map, not a claim that one adapter supports every implementation. A family label is insufficient: identify the concrete graph, parameter semantics, training objective, checkpoint format and runtime before acting.

## 1. General-purpose neural networks
- Linear/affine models; multilayer perceptrons; residual MLPs; gated MLPs; highway networks.
- Deep residual networks, DenseNet, Inception, EfficientNet, ConvNeXt, U-Net, encoder-decoder CNNs.
- Temporal/convolutional networks: TCN, WaveNet-like dilated causal CNNs, separable/depthwise CNNs.
- Capsule networks and dynamic-routing networks.

## 2. Sequence and attention models
- Vanilla RNN, Elman/Jordan networks, LSTM, GRU, bidirectional RNN, stacked and projected recurrent nets.
- Encoder-decoder seq2seq, attention-based seq2seq, Transformer encoder/decoder/encoder-decoder.
- Decoder-only causal LMs, prefix-LMs, retrieval-augmented models, long-context and recurrent-memory Transformers.
- Linear/efficient attention, grouped-query and multi-query attention, multi-head latent attention, sliding-window/local/global attention, hybrid attention.
- State-space models: S4-family, Mamba-family, selective SSMs, structured SSMs, hybrid SSM-attention models.
- RWKV-style recurrent/channel-mixing models, gated linear recurrence, RetNet-like retention models, other recurrent Transformers.
- Neural memory, fast-weight, associative-memory and modern Hopfield-style sequence models.

## 3. Sparse, routed and modular networks
- Mixture-of-Experts (token-choice, expert-choice, shared-expert, hierarchical MoE).
- Conditional computation, routing networks, modular networks, multi-task/shared-backbone systems.
- Hypernetworks, weight generators, implicit parameter networks, neural architecture search supernets.
- Ensembles, snapshot ensembles, parameter-sharing ensembles and cascades.

## 4. Vision and spatial perception
- CNN families and vision backbones; Vision Transformers; Swin/windowed transformers; hybrid CNN-Transformer.
- Object detection, instance/semantic/panoptic segmentation, pose/keypoint, depth and optical-flow networks.
- Siamese/twin networks, metric-learning and re-identification models.
- Video CNNs, 3D CNNs, temporal vision Transformers, video diffusion and action-recognition networks.
- Neural fields: NeRF, radiance fields, signed-distance fields, occupancy networks, 3D Gaussian Splatting parameter predictors.
- Geometric/deep equivariant networks, point-cloud networks, voxel/sparse-convolution networks, mesh networks.

## 5. Generative and latent-variable models
- GANs and conditional/style-based GANs; VAE, beta-VAE, VQ-VAE and hierarchical VAEs.
- Normalizing flows, autoregressive density estimators, energy-based models, score matching and diffusion/score models.
- Latent diffusion, pixel-space diffusion, rectified-flow/flow-matching models, consistency and distillation variants.
- Masked generative models, image tokenizers, discrete visual autoregressive models, world generators.
- Transformer and non-Transformer generators for image, video, 3D, motion and structured outputs.

## 6. Audio, speech and music
- ASR, CTC, RNN-T/transducer, encoder-decoder speech recognition.
- TTS, neural vocoders, voice conversion, speaker embedding/verification, diarization, enhancement and separation.
- Audio event classification, sound tagging, audio-text contrastive encoders, audio-language models.
- Neural audio codecs, codec language models, music generation and symbolic music models.
- Conformer, wav2vec-style, HuBERT-style, Whisper-like and hybrid architectures (identify actual implementation).

## 7. Multimodal and embodied systems
- Vision-language encoders, contrastive dual encoders, cross-attention fusion, Q-former/resampler bridges.
- VLMs, image-text and video-text models, audio-language and speech-language models.
- Unified multimodal autoregressive models, modality-specific experts, hybrid multimodal SSM/Transformer systems.
- Vision-action policies, language-conditioned robotics, visuomotor policies, imitation-learning and embodied world models.
- Sensor fusion, radar/LiDAR/camera fusion, multi-sensor temporal models.

## 8. Graph, relational and geometric learning
- GCN, GraphSAGE, GAT, GIN, message-passing neural networks, graph Transformers.
- Heterogeneous, temporal, dynamic, hypergraph and knowledge-graph neural networks.
- Molecular/property prediction, atomistic potentials, equivariant graph networks and geometric deep learning.
- Link prediction, node/graph classification and graph generative models.

## 9. Reinforcement learning, control and planning
- Policy networks, value/Q networks, actor-critic, distributional and recurrent RL networks.
- DQN-family, PPO/A2C/A3C, SAC/TD3/DDPG, model-based RL and learned dynamics/world models.
- Offline RL, sequence-decision Transformers, reward/value/preference models.
- Hierarchical policies, options, multi-agent policies, centralized critics and population-based systems.
- Model predictive control with learned components, differentiable control and system identification networks.

## 10. Time-series, forecasting and tabular models
- MLP/CNN/RNN/Transformer forecasting; temporal fusion and decomposition architectures.
- Long-horizon forecasting, probabilistic forecasting, quantile models, anomaly detection and imputation networks.
- TabNet-like, neural additive models, FT-Transformer/SAINT-like tabular nets, entity embeddings.
- Hybrid statistical-neural systems and learned state-space forecasters.

## 11. Scientific and physical machine learning
- PINNs, DeepONet, Fourier/neural operators, graph neural operators and physics-informed operator models.
- Surrogate models, PDE solvers, differentiable simulators, constitutive/material models.
- Hamiltonian/Lagrangian networks, symplectic nets, conservation/equivariance-constrained models.
- Neural ODE, Neural CDE, latent ODE, continuous-time and differential-equation networks.
- Inverse problems, Bayesian neural networks, probabilistic programming neural components.

## 12. Spiking, neuromorphic and biologically inspired
- Spiking neural networks (rate, temporal, surrogate-gradient, event-driven).
- Liquid/state-based recurrent networks, reservoir computing, echo-state networks.
- Cellular neural networks, neural cellular automata and differentiable local-rule systems.

## 13. Classical and non-neural parameterized models
- Linear/logistic regression, generalized linear models, factorization machines, matrix/tensor factorization.
- SVMs and kernel models (support vectors/dual coefficients are not ordinary neural weights).
- Decision trees, random forests, gradient-boosted trees (LightGBM, XGBoost, CatBoost), rule ensembles.
- k-means/prototypes, nearest-neighbor and retrieval indices (often have learned vectors or stored exemplars, not conventional weights).
- Classical statistical/state-space models with estimated coefficients.

For non-neural models, use a parameter-object adapter, not neural tensor assumptions. Preserve tree topology, split thresholds, categorical encodings, support vectors, kernel parameters, optimizer state and calibration objects as applicable.

## 14. Parameter representation and deployment variants (cross-cutting)
- FP32/FP16/BF16, mixed precision, FP8, integer quantization, binary/ternary networks.
- Per-tensor/per-channel/groupwise quantization; GPTQ/AWQ/GGUF and vendor-specific packed formats.
- Pruned, sparse, block-sparse, structured-sparse and low-rank/factorized checkpoints.
- Sharded/distributed checkpoints, tensor/pipeline/expert parallel layouts, FSDP/ZeRO shards.
- Adapter-only checkpoints: LoRA and variants, IA3, prefix/prompt tuning, side-tuning and modular deltas.
- Compiled/graph-captured/ONNX/TensorRT/CoreML/TFLite/ExecuTorch deployment artifacts.

## 15. Universal discovery rule
1. Detect file/container format without assuming it identifies the architecture.
2. Read config, metadata, source, model card and dependency versions.
3. Load in an isolated, resource-limited environment when safe.
4. Inspect module graph and parameter objects; capture shapes, aliases, tied parameters and buffers.
5. Run shape-checked dry forward/inference with representative inputs.
6. Identify objective, task heads, processor and post-processing.
7. Assign one or more family labels plus concrete architecture/version.
8. If evidence conflicts, mark architecture unresolved and do not modify.

## Coverage statement
The catalog aims to cover major model families and common variants across current ML practice. It cannot guarantee every research architecture, private implementation, future design, custom operator or proprietary checkpoint. Unknown systems must enter the custom adapter workflow and earn support through tests.
