# Additional fixed-budget study

Run commands below from the **repository root**. These entry points are independent from
main-result shell matrices. Default paths are relative to this repository, so no private
workspace is required. Read `docs/DATA_AND_WEIGHTS.md` first.

## Configure external inputs

`settings.json` supplies relative example paths. For another machine, copy it to
`settings.local.json`, update the data/weight/output paths, and use `--settings` on commands.
Use a separate local source registry and class-alias registry if multiple protocols are needed.
Both checked-in registries are empty; no checkpoint provenance has been certified on behalf
of the person running the code.

Required source bundle registration (after verifying checkpoint provenance and classifier row order):

```bash
python experiments/fixed_budget/common/register_source.py \
  --dataset office-home --source Art \
  --class-names-json /path/to/verified_class_names_in_row_order.json \
  --provenance 'description of source-only training and checkpoint selection' --confirm-reviewed
```

The class-names file is a JSON array of 65 names for Office-Home (12 for VisDA-C). Repeat
for Clipart, Product and RealWorld, or `--dataset VISDA-C --source train`. Registration records
hashes and does not download or train a source model. Do not mark unknown class orders as reviewed.
If the class-alias audit fails, verify aliases and record the reported mapping hash plus
`reviewed: true` and a semantic reason under the reported dataset/domain key in the registry.

The registered SfMamba baseline profile follows the released algorithm's Office-Home settings;
it is a complete algorithm reference, and is restricted to Office-Home/15 epochs. Its provenance
is documented in `sfmamba_officehome.released_code.json`.

## Inspect a plan without starting a GPU

```bash
python experiments/fixed_budget/common/queue_runs.py --layer 2
python experiments/fixed_budget/common/queue_runs.py --layer 3
```

Layer 2 defaults to six methods x four transfers x three seeds (72 runs), including the
original SfMamba reference. To plan the five CFA variants only:

```bash
python experiments/fixed_budget/common/queue_runs.py --layer 2 \
  --methods output_only pointwise_gate conditional_mlp reset_state scan
```

## Build a checked CLIP cache

After installing the runtime and placing the matching CLIP file and source weights:

```bash
python experiments/fixed_budget/common/launch.py --layer 1 --action cache \
  --dataset office-home --target Clipart
python experiments/fixed_budget/common/launch.py --layer 1 --action cache \
  --dataset VISDA-C --target validation
```

Repeat for all Office-Home target domains used by the matrix. The cache manifest binds class
order, image identity, preprocessing and frozen CLIP weights.

## Frozen references and fixed fusion

```bash
python experiments/fixed_budget/common/launch.py --layer 1 --action references \
  --dataset office-home --source Art --target Clipart
```

This evaluates CLIP, the frozen source and their equal probability average. It does not
train a fusion network.

## One controlled training run

```bash
python experiments/fixed_budget/common/launch.py --layer 2 --action check \
  --source Art --target Clipart --method scan --seed 0 --epochs 15
python experiments/fixed_budget/common/launch.py --layer 2 --action train \
  --source Art --target Clipart --method scan --seed 0 --epochs 15
```

For VisDA-C, use `--layer 3 --source train --target validation`.
For a complete queue, add `--execute` to the queue command. Without it, only plans are printed.
The queue first validates inputs and runs adapter unit tests. All attempts require new output
directories. Endpoints and results are written under `outputs/fixed_budget/layer2/runs/`
or `layer3/runs/`, not into the repository's tracked source files.

## Aggregate three seeds

```bash
python experiments/fixed_budget/common/summarize.py \
  --runs-root outputs/fixed_budget/layer2/runs --dataset office-home \
  --methods output_only pointwise_gate conditional_mlp reset_state scan \
  --epochs 15 --output outputs/fixed_budget/summary_officehome
```

It requires complete, matching runs. VisDA-C uses `--dataset VISDA-C` and the appropriate
three methods and run root. Outputs include raw/summary CSV, paired differences and audit metadata.

## Measure runtime for a completed endpoint

```bash
python experiments/fixed_budget/layer3/benchmark.py \
  --run-dir /path/to/completed/attempt_01 --mode inference
# Add --execute to measure; choose uncached_inference or train_steps for other scopes.
```

The default is a plan. GPU measurements start only with `--execute`; result directories
and checkpoints are checked before measurement. `layer1/analyze_predictions.py --help`
documents aligned before/after prediction analysis.

`common/variants.py` implements conditional MLP, state-reset and shared-condition controls.
`common/runtime.py` retains historical function names for compatibility with the recorded
protocol; CLIP is a frozen external supervisor. See root documentation for scientific scope.
