# Cnn Resnet adapter

## Required discovery
Inspect configuration, source implementation, module tree, forward path, input/output shapes, parameter mapping, preprocessing, loss and inference state.

## Compatibility
Do not assume support for generic LoRA, attention hooks, pruning or export. Verify each operation with a minimal forward/backward and save/load test.

## Required records
Supported model classes, module names, tensor mappings, hook locations, trainable parameters, supported transforms, resource requirements, export route, known limitations and regression tests.

## Status
Template only. Mark this adapter unsupported until a concrete implementation and tests exist.
