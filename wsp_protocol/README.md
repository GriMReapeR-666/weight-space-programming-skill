# Weight-Space Programming Protocol (WSP) — v4 extension

This module upgrades the bundle from a collection of architecture adapters into an agent-usable, tool-mediated weight-space programming workflow.

## Core principle

The language model is the hypothesis generator and experiment director. Deterministic tools perform tensor inspection, activation capture, candidate generation, patch application, evaluation, statistical comparison, and rollback. The model must never be asked to mentally parse or regenerate giant raw tensors.

## What this adds

- Behavioral diagnosis before tensor diagnosis.
- Circuit/feature maps built from module topology, activation traces, attribution and controlled interventions.
- A tool contract for safe read-only inspection, activation capture, ablation, candidate search, patch validation, commit and rollback.
- Gradient-free search methods alongside gradient-based methods as explicitly separate choices.
- Sparse, reversible delta patches rather than whole-matrix text output.
- Multi-objective evaluation, holdout protection, uncertainty estimates and regression gates.
- A model-family-neutral adapter protocol, including unknown and hybrid architectures.
- A reproducibility ledger recording exact model, tokenizer, runtime, data, patch and evaluation hashes.

## Critical epistemic rule

Treat claims about any named paper, model, benchmark gain, parameter count or demonstration as unverified until the primary paper, official repository, model artifact and evaluation protocol have been inspected. The user-provided narrative is a proposed workflow, not evidence that a particular result occurred. Do not claim that a specific model or research team performed a procedure unless independently verified.

## Install/use

1. Read `../SKILL.md`, then this directory's `playbooks/00-orchestrator.md`.
2. Run `tools/environment_report.py` and the existing bundle's checkpoint manifest utility.
3. Begin in DISCOVER/DIAGNOSE read-only mode. Do not edit source checkpoints.
4. Select or implement an adapter. Run conformance tests.
5. Establish baseline and holdout datasets before proposing interventions.
6. Execute bounded experiments in disposable copies. Keep every patch reversible.
7. Promote only after the acceptance gates pass.

## Terminology

- **Weight-space programming:** deliberate direct manipulation of parameters or parameter-space structures, with no optimizer step required for the intervention itself.
- **Activation-space intervention:** temporary manipulation of intermediate activations. This is diagnostic evidence, not a weight edit.
- **Gradient-free search:** candidate selection by evaluation (random search, coordinate search, evolutionary search, Bayesian optimization, etc.). It is not necessarily zero-shot and can be expensive.
- **Gradient-based optimization:** uses derivatives and an optimizer; useful but not direct gradient-free WSP.
- **Circuit hypothesis:** a testable claim about a component/direction contributing to behavior, not a claim that a neuron has a single human-readable meaning.
