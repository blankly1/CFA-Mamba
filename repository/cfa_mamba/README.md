# CFA-Mamba implementation

`model.py` implements semantic conditioning, bidirectional scanning, gating and residual adaptation.
`losses.py` implements the five objectives; `clip_teacher.py` constructs frozen CLIP distributions.
The historical identifiers `ClipSemanticTeacher`, `CGMambaLoss` and `ClipGuidedSfMamba` are retained
for checkpoint/API compatibility. They do not assert a teacher-student distillation architecture.

See the root README and `docs/REPRODUCIBILITY.md` before launching experiments.
