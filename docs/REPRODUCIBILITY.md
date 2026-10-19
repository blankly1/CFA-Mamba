# Experiment protocols

## Main-result launchers

The Small backbone, batch size 32, test batch size 16, AdamW, frozen source classifier,
frozen source anchor and frozen CLIP are shared. Loss parameters below follow the included
launch scripts; dataset-specific source bundles are external inputs.

| Setting | Office-Home | Office-31 | DomainNet-126 | VisDA-C |
|---|---:|---:|---:|---:|
| Epochs | 15 | 15 | 5 | 15 |
| Base LR | 3e-5 | 3e-5 | 3e-5 | 3e-6 |
| Feature LR factor | 0.1 | 0.1 | 0.1 | 0.1 |
| Fusion LR factor | 1.0 | 1.0 | 1.0 | 0.1 |
| Adapter LR factor | 1.0 | 1.0 | 1.0 | 0.1 |
| CLIP confidence threshold | 0.78 | 0.78 | 0.78 | 0.55 |
| Pseudo / consistency weights | 0.5 / 0.3 | 0.5 / 0.3 | 0.5 / 0.3 | 1.0 / 1.0 |
| Entropy / diversity / anchor | 0.1 / 1 / 0.1 | 0.1 / 1 / 0.1 | 0.1 / 1 / 0.1 | 0.05 / 1 / 0.1 |
| CLIP output schedule warmup / minimum | 0 / 1.0 | 1 / 0.05 | 1 / 0.05 | 0 / 1.0 |
| Default adaptation seeds | 0,1,2 | 0 | 0 | 0 |

The source model is source-trained; freezing its classifier does not imply the target
extractor is frozen. The adapter operates along spatial tokens, not across class indices.
The source/CLIP weighting uses KL(CLIP || source), consistency uses KL(CLIP || target),
and the source anchor uses KL(source || target).

Main launchers default to NUM_WORKERS=1 and CLIP_NUM_WORKERS=0 for portability. Preserve
recorded workers when reconstructing a particular training trajectory. The additional
runner defines its own workers (Office-Home 8, VisDA-C 2). No claim of identical optimizer
trajectories is made merely because seeds and nominal hyperparameters match.

## Fixed endpoints

Main `train.py` evaluates each epoch and logs accuracy to one decimal place. Paper summaries
use the final logged evaluation; use the unrounded saved predictions/records when available
for higher-precision derived statistics. `--issave` in that generic entry saves the best
model and must not be used as a substitute for a final model.

The existing `train_cg_mamba_save_endpoint.py` additionally exports the final training state
under `endpoint_checkpoint/`, regardless of whether the best-save flag is enabled. Set
`TRAIN_ENTRY=train_cg_mamba_save_endpoint.py` to use it with the main launcher. It preserves
the scan/gate entry's model and loss computation. A future exported endpoint does not restore
a historical checkpoint that was never saved.

Use fresh output directories: main logs append, whereas the additional runner rejects
an existing run directory and records each explicit attempt separately.

## Additional controlled study

The five CFA variants are output-only, pointwise gate, conditional MLP, state-reset and scan.
The optional original SfMamba objective is a complete algorithm reference, not a pure module
ablation. Office-Home uses A->C, C->P, P->R, R->A; VisDA-C uses train->validation.
Seeds are 0,1,2. The fixed-budget default is 15 epochs; 30-epoch jobs are a separate budget.

The runner binds source/checkpoint hashes, class order, data and cache identity, sampled
index-sequence hashes, losses, final endpoint and successful mixed-precision updates.
Frozen CLIP and frozen source are also evaluated individually and as a fixed equal-weight
probability average; this ensemble has no trainable fusion weights.

Within each task and seed, compute scan minus comparator; average transfers within each
seed, then compute the mean and sample standard deviation over the three seeds. VisDA-C
benchmark accuracy is mean per-class; subgroup CLIP-correct/wrong analyses use sample-weighted
top-1. A subgroup top-1 is not interchangeable with the mean per-class benchmark.

## Timing scope

Additional `layer3/benchmark.py` measures cached inference, inference including the CLIP
forward, or adaptation training steps (including the frozen source anchor). It uses resident
real-image batches, AMP and five timing blocks. Inference batch size defaults to 16;
training uses 32. Image decoding and host-to-device transfer are outside these measurements.
Run on the intended GPU and keep hardware/software metadata with results. Timing reports
from different scopes or batches should not be merged.

## Reproduction boundary

This release contains executable procedures and protocol descriptions, not datasets or all
historical weights/logs. Main results, structural comparisons, same-checkpoint interventions
and local hyperparameter studies remain distinct experiments. The bundle has not been
retrained as part of this release preparation. Static/CPU validation is documented separately.
