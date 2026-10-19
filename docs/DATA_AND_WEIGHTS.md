# Data and weights

No datasets, images or weights are bundled. Obtain benchmarks under their original usage terms.
Training consumes only target images and existing source-model parameters during adaptation.
Evaluation labels are used for metrics and fixed data/class-order audits, not for target tuning,
checkpoint selection or early stopping.

## Dataset layouts

For main launchers, `DATA_ROOT` points directly to the selected dataset directory.
For the additional study, `dataset_root` contains the four directories shown below:

```text
data/
  officeHome/OfficeHome/
    Art/<class_name>/*.jpg
    Clipart/<class_name>/*.jpg
    Product/<class_name>/*.jpg
    RealWorld/<class_name>/*.jpg
  VISDA-C/
    train/<class_name>/*
    validation/<class_name>/*
  office31/
    amazon_list.txt
    dslr_list.txt
    webcam_list.txt
    amazon/...
    dslr/...
    webcam/...
  Domainnet/
    clipart_list.txt
    painting_list.txt
    real_list.txt
    sketch_list.txt
    clipart/...
    painting/...
    real/...
    sketch/...
```

Office-Home and VisDA-C use torchvision ImageFolder ordering. Office-31 and DomainNet-126 use
list entries `relative/path/to/image integer_class_index`. Paths are relative to DATA_ROOT;
use forward slashes, and keep lists aligned with the trained classifier's output-row order.
DomainNet-126 requires the 126-class subset and matching lists, not the full 345-class benchmark.

`repository/cfa_mamba/class_names.py` defines the semantic names; do not infer checkpoint
compatibility from class counts alone. The additional study rejects unreviewed class aliases.

## Source model bundles

```text
weights/source/
  office-home/Art/source_F.pt
  office-home/Art/source_B.pt
  office-home/Art/source_C.pt
  office-home/Clipart/...
  office-home/Product/...
  office-home/RealWorld/...
  office/amazon/...
  office/dslr/...
  office/webcam/...
  domainnet126/clipart/...
  domainnet126/painting/...
  domainnet126/real/...
  domainnet126/sketch/...
  VISDA-C/train/...
```

Use source-trained SfMamba Small checkpoints matching `bn_relu`, `ssm` fusion and the class map.
The inherited F format contains the union of patch-embedding and layer state dictionaries;
B contains fusion blocks and C the classifier. Verify module keys/shapes and class names,
rather than renaming unrelated checkpoints. Source checkpoints and target endpoints are different artifacts.
The inherited `repository/main_source.py` is included for source training; its result is not
claimed to recreate every historical source checkpoint without the original source-training protocol.

## CLIP checkpoint

- Library/model: OpenCLIP `ViT-B-16`.
- Pretrained identifier: `laion2b_s34b_b88k` (not the distinct `openai` checkpoint).
- Additional-study local file: `weights/clip/ViT-B-16-laion2b_s34b_b88k/open_clip_pytorch_model.bin`.
- Recorded file SHA-256: `1f16b1274ca25493b1f36225efe81d05f66a3f33c487acd1a07c5ba87920de6f`.
- Official checkpoint repository: https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K

Main training supports the identifier (downloaded by OpenCLIP) or a local file through
`CLIP_PRETRAINED`. The additional runner requires the recorded local-file hash for strict
comparison; an otherwise equivalent reserialized checkpoint will not pass that byte check.

Probability caches are indexed by dataset index. Main-launcher cache reuse is deliberately
simple: filenames are not a substitute for checking image/class/weight identity. Use a fresh
output/cache directory whenever those inputs change. The additional runner checks image
manifests, class order, weight hashes and cache metadata before training.
