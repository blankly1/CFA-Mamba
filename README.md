# CFA-Mamba

Implementation of CFA-Mamba for source-free domain adaptation with frozen external CLIP
supervision and a semantically conditioned feature adapter. The adapter performs a
bidirectional recurrence on the final spatial feature map before classification.

**CLIP:** OpenCLIP `ViT-B-16`, pretrained **`laion2b_s34b_b88k`**, temperature `0.01`, five fixed prompts.
CLIP, the source reference and source classifier are frozen during adaptation.
The target feature extractor, its fusion blocks and the adapter are optimized.

[中文说明](README_zh.md) · [Installation](INSTALL.md) ·
[Data and weights](docs/DATA_AND_WEIGHTS.md) · [Protocols](docs/REPRODUCIBILITY.md) ·
[Additional study](experiments/fixed_budget/README.md) · [Validation](docs/VALIDATION.md)

**Paper:** CFA-Mamba: Frozen CLIP Semantics for Source-Free Domain Adaptation with Mamba.

## Repository layout

```text
repository/
  cfa_mamba/                Adapter, objectives, CLIP cache and adaptation entry points
  models/                  SfMamba backbone and inherited Mamba kernels
  configs/                 Small/Tiny backbone configurations
  datautil/, utils/        Data loading, optimization and evaluation
  main_source.py           Inherited source-training entry
  main_target.py           Inherited SfMamba reference implementation
third_party/selective_scan/ CUDA extension source
scripts/                   Main experiment launchers and endpoint extraction
experiments/fixed_budget/  Controlled adapters, fixed references, statistics and timing
  common/                  Runtime, controls, provenance and aggregation
  layer1/                  CLIP cache, frozen references and prediction analysis
  layer2/                  Office-Home structural controls
  layer3/                  VisDA-C controls and runtime measurements
  tests/                   Existing planning and adapter-math tests
```

Datasets, checkpoints, probability caches and measured outputs are external resources,
and are excluded from this repository. This code release does not invent missing artifacts.

## Quick start

Install on Linux with an NVIDIA GPU, using [INSTALL.md](INSTALL.md). From the repository root:

```bash
export REPO_ROOT="$PWD"
export DATA_ROOT=/path/to/officeHome/OfficeHome
export SOURCE_MODEL_ROOT=/path/to/source_models
export OUTPUT_ROOT=/path/to/a_new_output_directory
export CLIP_PRETRAINED=laion2b_s34b_b88k

# One Office-Home transfer, Art -> Clipart, adaptation seed 0:
DATASET=office-home SOURCE_ENV=0 TARGET_ENV=1 TARGET_NAME=Clipart \
  CLIP_WARMUP_EPOCHS=0 CLIP_MIN_WEIGHT=1.0 bash scripts/train_target.sh
```

Alternatively, set `CLIP_PRETRAINED` to the matching local checkpoint file. For another
weight or preprocessing protocol, choose a new `CLIP_CACHE_ID` and rebuild its cache.
Do not reuse a cache across changed weights, prompts, temperature, class order or image manifests.

## Main experiment launchers

Set `DATA_ROOT` to the relevant dataset directory before each command.

| Script | Dataset | Default adaptation seeds | Epochs |
|---|---|---|---|
| `scripts/reproduce_officehome.sh` | Office-Home, 12 transfers | 0, 1, 2 | 15 |
| `scripts/reproduce_officehome_output_only.sh` | Office-Home output-only, 12 transfers | 0, 1, 2 | 15 |
| `scripts/reproduce_office31.sh` | Office-31, 6 transfers | 0 | 15 |
| `scripts/reproduce_domainnet126.sh` | DomainNet-126, 12 available transfers | 0 | 5 |
| `scripts/reproduce_visda.sh` | VisDA-C train -> validation | 0 | 15 |
| `scripts/reproduce_office31_gate_scan.sh` | Office-31 gate/scan comparison | 0, 1, 2 | 15 |

The DomainNet launcher provides the full available transfer matrix; use the corresponding
reported subset when comparing with a paper table. Single-seed launchers accept `SEED` overrides.
Run scripts into a fresh `OUTPUT_ROOT` to avoid appending new runs to old logs.
See [protocol details](docs/REPRODUCIBILITY.md) for losses, LR factors and result selection.

The controlled additional study uses its own source bundles, cache checks and endpoint runtime.
Start with [its README](experiments/fixed_budget/README.md); do not mix its runs with main-result averages.

## Results and checkpoints

Reported accuracies use the **final evaluation after the fixed epoch budget**, not the maximum
validation accuracy. Main training writes `log_rank0.txt`; extract its final result with:

```bash
python scripts/evaluate_fixed_endpoint.py /path/to/log_rank0.txt --expected-evaluations 15
```

The generic `train.py --issave` branch saves a best checkpoint and is not a final-endpoint export.
For future endpoint analyses, set `TRAIN_ENTRY=train_cg_mamba_save_endpoint.py` when running
`train_target.sh`; the existing exporter writes `endpoint_checkpoint/` after the last epoch.
Do not substitute a best checkpoint for an endpoint. Additional-study runs save `endpoint.pt`
and the full comparison contract automatically.

Office-Home, Office-31 and DomainNet use top-1 accuracy; VisDA-C uses mean per-class accuracy.
Three-seed error bars are sample standard deviations, not confidence intervals.

## Checks

```bash
python scripts/verify_release.py
python -m unittest discover -s experiments/fixed_budget/tests -p 'test_planning.py' -v
# In an environment with PyTorch (CUDA is not needed for this unit test):
python -m unittest discover -s experiments/fixed_budget/tests -p 'test_controls_torch.py' -v
```

See [release changes](docs/RELEASE_CHANGES.md) and [file origins](docs/FILE_ORIGINS.json).
Historical internal identifiers such as `ClipSemanticTeacher` are retained for API/checkpoint
compatibility; CLIP is used as fixed external supervision, without asserting a teacher-student architecture.

## Acknowledgements and licensing

This implementation builds on SfMamba, VMamba, Mamba/Mamba-2 and OpenCLIP. Existing attribution
headers are preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream notices
and license scope.

This repository is provided for reference only. No license is granted for the authors'
original contributions. All rights are reserved. Third-party components remain subject
to their respective licenses; this notice does not restrict rights granted by those licenses.
Rights available under applicable law and GitHub's terms remain unaffected.
This repository is not offered under an open-source license.
