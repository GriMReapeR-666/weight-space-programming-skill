# TransformerLens, native hooks and compatibility

TransformerLens is useful for supported Transformer implementations and mechanistic interpretability workflows. It is not a universal loader or universal circuit extractor.

Use it only when:
- the exact model architecture/version is supported;
- tokenizer and positional conventions are compatible;
- weight conversion is verified;
- numerical outputs match a reference implementation within tolerance.

For unsupported, hybrid, recurrent, multimodal, diffusion, convolutional or custom architectures, prefer native framework instrumentation or implement an adapter around the actual forward graph. PyTorch module hooks are not universal: compiled/fused graphs, custom kernels, JAX/XLA, TensorRT and remote inference endpoints may need different instrumentation.

Always compare instrumented and uninstrumented outputs to detect hook-induced behavior changes.
