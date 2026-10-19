# Third-party notices and license scope

Existing copyright headers are retained in the copied source files. This release does not
relicense third-party code or pretrained weights.

| Component | Source / relationship | Included notice |
|---|---|---|
| SfMamba backbone, data/config/training utilities | Inherited from the SfMamba implementation used by the experiments | Preserve the source headers; no standalone SfMamba license file was present in the local source snapshot |
| VMamba-derived visual scanning code | https://github.com/MzeroMiko/VMamba | Upstream MIT text in `licenses/VMamba-LICENSE.txt` |
| Mamba / Mamba-2 kernels and Triton ops | https://github.com/state-spaces/mamba ; `repository/models/mamba2/__init__.py` identifies its upstream source | Upstream Apache-2.0 text in `licenses/Mamba-LICENSE.txt` |
| OpenCLIP | https://github.com/mlfoundations/open_clip | Installed dependency; not vendored; its code/weight licenses apply separately |

License texts were retrieved from the named upstream repositories on 2026-10-17. They document
the identified upstream components, not an independently verified license for every historical
modification in the local SfMamba snapshot. Existing Selective Scan setup metadata mentions BSD,
while current Mamba upstream supplies Apache-2.0; that inherited metadata is not a project-wide license.

## Authors' original contributions

This repository is provided for reference only. No license is granted for the authors'
original contributions. All rights are reserved. It is not offered under an open-source
license. Rights available under applicable law and GitHub's terms remain unaffected.

## Third-party scope

Third-party components remain subject to their respective licenses. The reference-only
notice above does not replace those licenses, relicense upstream code, or restrict rights
already granted by upstream copyright holders. Existing copyright and attribution notices
and included upstream license texts must be retained as required by those licenses.

The redistribution terms for the inherited SfMamba snapshot still require confirmation;
the reference-only notice does not establish permission to redistribute that code.
