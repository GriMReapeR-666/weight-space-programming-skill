# Gradient-free search and surgical delta optimization

## Separation of methods

This playbook covers search methods that select parameter changes using measured objective values without using gradient descent for the patch itself. It does not claim the search is zero-compute or zero-training in every broad sense.

## Candidate parameterizations

Prefer a compact parameterization:
- scalar scale on a validated component;
- sparse coordinate set;
- blockwise scale;
- low-rank delta `ΔW = A B`;
- subspace edit `ΔW = U C Vᵀ`;
- structured mask;
- expert/router bias adjustment;
- adapter coefficients;
- quantization scale/codebook parameters where supported.

Avoid unconstrained full-matrix rewrites by an LLM.

## Search methods

Select based on dimensionality and evaluation cost:
- bounded grid/coordinate search for very few parameters;
- random search for robust low-dimensional exploration;
- evolution strategies/CMA-ES for continuous low-dimensional parameterizations;
- simulated annealing for discrete or mixed search spaces;
- Bayesian optimization for expensive, low-dimensional black-box objectives;
- evolutionary or population-based methods for mixed structured candidates;
- greedy group selection for sparse edits.

Use a fixed budget and log every candidate. Do not run an unbounded search.

## Objective

Define a multi-objective score, for example:
`J = task_quality - λ1 * regression_penalty - λ2 * latency_penalty - λ3 * memory_penalty - λ4 * safety_penalty`.

Keep raw metric values as well as any scalarized score. Never hide a regression inside a single aggregate.

## Search protocol

1. Define editable parameterization and allowed bounds.
2. Establish no-op and random-control baselines.
3. Generate candidate IDs deterministically.
4. Apply candidate to disposable copy/overlay.
5. Evaluate on search split.
6. Reject invalid, unstable or clearly regressive candidates.
7. Re-evaluate finalists on validation split.
8. Select one candidate using a predeclared criterion.
9. Evaluate once on held-out test set.
10. Rebuild from clean base and reproduce.

## Important caution

A high-performing patch may overfit a small benchmark, exploit test artifacts, or damage unmeasured capabilities. A small benchmark gain is not proof of general improvement. Report uncertainty and all important regressions.
