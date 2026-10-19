# Installation

The recorded training environment is Linux x86_64, Python 3.10.12, PyTorch 2.1.0+cu121,
torchvision 0.16.0 and Triton 2.1.0. Exact Python dependencies are listed in
`environment/requirements.txt`; `environment/runtime.txt` records the original GPU runtime.
A compatible NVIDIA driver, CUDA 12.1 toolkit (including nvcc) and C++ build toolchain are needed.

Run from the repository root:

```bash
conda create -n cfa-mamba python=3.10 -y
conda activate cfa-mamba
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -v --no-build-isolation ./third_party/selective_scan
export PYTHONPATH="$PWD/repository:$PWD/third_party/selective_scan"
python repository/cfa_mamba/train.py --help
python -c 'import torch, selective_scan_cuda_oflex; print(torch.__version__, torch.cuda.is_available())'
```

Keep the CUDA toolkit consistent with the selected PyTorch build. A CPU-only install can run
planning checks, but cannot train the SfMamba CUDA backbone. Native Windows GPU training has
not been validated. Dataset and source checkpoints are required before a real training smoke run.

The repository supplies bundled CUDA sources rather than precompiled binaries. Do not import
an older SfMamba checkout through PYTHONPATH. If a different environment is used, record its
package versions and evaluate reproducibility rather than assuming numerical equivalence.

`environment/environment.yaml` is an optional Python-environment skeleton; the explicit commands
above are the primary install route because they specify the CUDA PyTorch wheel index.
