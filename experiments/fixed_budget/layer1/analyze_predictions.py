"""Offline paired predictions or historical same-checkpoint intervention analysis."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from contracts import sha, write_json


def extract(path, prefix=None):
    with np.load(path, allow_pickle=False) as f:
        ykey = 'labels' if 'labels' in f else 'label'
        y = f[ykey].reshape(-1)
        idx = f['dataset_index'].reshape(-1)
        if prefix:
            scores = f[prefix]
        else:
            scores = f['logits'] if 'logits' in f else f['probs']
        if len(set(idx.tolist())) != len(idx) or len(y) != len(idx) or len(scores) != len(idx):
            raise ValueError('Invalid prediction indices/shapes')
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite predictions')
        order = np.argsort(idx)
        return idx[order], y[order], scores[order]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', required=True)
    p.add_argument('--after', help='Omit for a historical mechanism_arrays.npz file')
    p.add_argument('--teacher', help='teacher_predictions.npz for independent model comparisons')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.after:
        if not args.teacher:
            p.error('--teacher required for independently trained model comparison')
        before = extract(args.before)
        after = extract(args.after)
        teacher = extract(args.teacher)
        kind = 'independently_trained_predictions_requires_external_protocol_audit'
    else:
        before = extract(args.before, 'logits_before')
        after = extract(args.before, 'logits_after')
        teacher = extract(args.before, 'clip_probs')
        kind = 'same_checkpoint_intervention_not_output_only'
    for other in (after, teacher):
        if not np.array_equal(before[0], other[0]) or not np.array_equal(before[1], other[1]):
            raise ValueError('Sample indices or labels differ')
    y = before[1]
    b, a, t = [item[2].argmax(1) == y for item in (before, after, teacher)]
    confidence = teacher[2].max(1)
    if not np.allclose(teacher[2].sum(1), 1, atol=1e-5):
        raise ValueError('Teacher file must contain probabilities')
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, 6)))
    groups = {'all': np.ones(len(y), dtype=bool), 'clip_correct': t, 'clip_wrong': ~t}
    for i in range(len(edges)-1):
        groups[f'confidence_bin_{i}'] = ((confidence >= edges[i]) &
                                        (confidence <= edges[i+1] if i==len(edges)-2 else confidence < edges[i+1]))
    rows = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        rows[name] = dict(n=n, before_accuracy=100*float(b[mask].mean()) if n else None,
                          after_accuracy=100*float(a[mask].mean()) if n else None,
                          corrected=int((~b & a & mask).sum()), broken=int((b & ~a & mask).sum()))
    write_json(output, dict(provenance='measured_archive_analysis', comparison_kind=kind,
                            before_sha256=sha(args.before), after_sha256=sha(args.after or args.before),
                            teacher_sha256=sha(args.teacher or args.before), groups=rows,
                            confidence_edges=edges.tolist(),
                            warning='Dataset indices alone do not establish cross-run manifest equality; audit manifests before publication.'))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
