"""Aggregate only complete, matched runs. Original SfMamba is an algorithm reference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from contracts import CORE_PAIRS, csv_write, read_json, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', required=True, help='Layer-specific runs folder')
    p.add_argument('--methods', nargs='+', required=True)
    p.add_argument('--dataset', choices=['office-home', 'VISDA-C'], required=True)
    p.add_argument('--epochs', type=int, choices=[15,30], default=15)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    selected = {}
    for path in Path(args.runs_root).glob('*/attempt_*/result.json'):
        record = read_json(path)
        if record['dataset'] != args.dataset or record['epochs'] != args.epochs or record['method'] not in args.methods:
            continue
        if record['status'] != 'completed' or record['completed_epochs'] != args.epochs:
            raise ValueError(f'Incomplete run: {path}')
        key = (record['source'], record['target'], record['seed'], record['method'])
        if key in selected:
            raise ValueError(f'Multiple completed attempts; resolve provenance explicitly: {key}')
        if sha(path.parent/'endpoint.pt') != record['endpoint_sha256']:
            raise ValueError(f'Endpoint hash mismatch: {path}')
        selected[key] = record
    pairs = CORE_PAIRS if args.dataset=='office-home' else [('train','validation')]
    expected = {(s,t,seed,m) for s,t in pairs for seed in (0,1,2) for m in args.methods}
    if set(selected) != expected:
        raise ValueError(f'Missing={sorted(expected-set(selected))}; unexpected={sorted(set(selected)-expected)}')
    for source, target in pairs:
        for seed in (0,1,2):
            records = [selected[(source,target,seed,m)] for m in args.methods]
            if len({r['comparison_contract_sha256'] for r in records}) != 1:
                raise ValueError(f'Comparison contract mismatch: {source}->{target} seed{seed}')
            if len({tuple(r['sample_order_hashes']) for r in records}) != 1:
                raise ValueError('Observed sample orders differ')
    summaries, raw = [], []
    for method in args.methods:
        seed_means = [statistics.mean(selected[(s,t,seed,method)]['result']['accuracy_percent'] for s,t in pairs)
                      for seed in (0,1,2)]
        summaries.append(dict(method=method, mean=statistics.mean(seed_means),
                              sample_std=statistics.stdev(seed_means), seed0=seed_means[0],
                              seed1=seed_means[1],seed2=seed_means[2]))
        for source,target in pairs:
            for seed in (0,1,2):
                r=selected[(source,target,seed,method)]
                raw.append(dict(method=method,source=source,target=target,seed=seed,
                                accuracy_percent=r['result']['accuracy_percent'],
                                successful_updates=r['successful_updates'],attempted_steps=r['attempted_steps']))
    paired=[]
    if 'scan' in args.methods:
        for method in args.methods:
            if method == 'scan':
                continue
            deltas=[statistics.mean(selected[(s,t,seed,'scan')]['result']['accuracy_percent']-
                                    selected[(s,t,seed,method)]['result']['accuracy_percent'] for s,t in pairs)
                    for seed in (0,1,2)]
            paired.append(dict(comparison='scan-minus-'+method,mean_pp=statistics.mean(deltas),
                               std_pp=statistics.stdev(deltas),seed_deltas=deltas,
                               role='algorithm_difference_not_pure_CLIP_loss_gain' if method=='sfmamba_original' else 'structure_comparison'))
    out.mkdir(parents=True)
    csv_write(out/'summary.csv',summaries)
    csv_write(out/'raw.csv',raw)
    write_json(out/'paired.json',paired)
    write_json(out/'audit.json',dict(status='contract_and_sample_order_passed',
                                    runs=len(selected),seeds=[0,1,2],directions=pairs,
                                    manual_checks_remaining=['actual LR traces for shared parameter groups','model-control gradient verification on GPU'],
                                    paper_ready=False,
                                    note='No automatic significance claim; finish listed checks before publication.'))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
