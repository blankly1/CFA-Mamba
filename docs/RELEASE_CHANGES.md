# Public-release transformations

The release is assembled from the maintained journal-experiment workspace and the additional
fixed-budget experiment runner. The archived submission and reported results are unchanged.

1. Copy code, CUDA sources, model configurations and launch scripts; omit datasets, checkpoints,
   probability caches, logs, server records and compiled binaries.
2. Fix the repository-root lookup in legacy adaptation entry points (`parents[2]` to `parents[1]`).
3. Set public CLIP defaults to OpenCLIP `ViT-B-16`, pretrained `laion2b_s34b_b88k`.
   Explicit command-line choices remain supported.
4. Correct loss docstrings to match the implemented KL directions; no loss arithmetic changed.
5. Keep historical Python class/checkpoint identifiers to preserve loading compatibility.
6. Relocate the additional runner to `experiments/fixed_budget`, auto-resolve this repository,
   replace output-stage names with `layer1`, `layer2`, `layer3`, and use configurable relative paths.
7. Give CLIP caches a weight/temperature identifier; remove inherited PYTHONPATH from the main launcher.
8. Let single-seed VisDA/Office-31/DomainNet launchers accept a caller-provided SEED (default 0).
9. Expose the existing endpoint-saving entry via TRAIN_ENTRY; the default main training loop is unchanged.
10. Remove an unused source-only local smoke script, machine-specific documentation and stale manifests.

The original additional-runner protocol string remains for provenance; a relocated public run
has its own code hashes and is not a byte-identical re-execution of a historical code snapshot.

11. Add the inherited reference entry's missing wandb dependency; no online tracking is enabled by default.
12. Add an output-only Office-Home launcher with a distinct RUN_LABEL to avoid overwriting scan logs.

13. Replace the inherited missing derivation-image link with the local CUDA install guide.
