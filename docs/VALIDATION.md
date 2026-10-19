# Release validation

Prepared 2026-10-17. Validation below concerns packaging, not experimental accuracy.

- Python source syntax and checked-in JSON parsed successfully.
- Seven shell launchers passed syntax parsing without execution.
- Seven existing dependency-free experiment-planning tests passed.
- Default additional-study settings resolve to this public repository after relocation.
- Office-Home and VisDA-C dry-run matrices produce 72 and 9 jobs respectively, without GPU execution.
- Runtime benchmark dry-run reports five timing blocks and no GPU execution.
- Core model, loss and class-mapping ASTs match the maintained source after ignoring docstrings.
- The release includes SHA-256 file integrity metadata and a dependency-free verification command.
- Published source was scanned for known private Windows/server paths and credential patterns;
  no matching credentials were found. Weights, caches, images, outputs and compiled binaries are omitted.

The preparation machine does not have PyTorch, so adapter tensor tests, CUDA extension compilation,
training, accuracy reproduction and GPU timing were not executed here. Existing tensor tests are
included for the training environment. Shell syntax is also checked by the supplied Linux
GitHub Actions workflow; the workflow itself has not run until the repository is uploaded.

Before a full matrix on the server, run the included CPU adapter tests and one GPU transfer to
verify cache indices, frozen parameters, loss finiteness, output shape and endpoint saving.
Do not interpret a static check as confirmation that all paper metrics have been reproduced.
