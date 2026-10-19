"""Plan a layer matrix; --execute explicitly launches isolated GPU processes."""
import argparse
import importlib.util
import json
import subprocess
import sys

from contracts import PACKAGE, CORE_PAIRS, METHODS, jobs, preflight, settings, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', default=str(PACKAGE/'settings.json'))
    p.add_argument('--layer', choices=['2', '3'], required=True)
    p.add_argument('--methods', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, choices=[15, 30], default=15)
    p.add_argument('--pair', help='Example Art:Clipart; restricts to one pair')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    args = p.parse_args()
    if args.attempt < 1:
        p.error('attempt must be positive')
    s = settings(args.settings)
    dataset = 'office-home' if args.layer == '2' else 'VISDA-C'
    methods = args.methods or (METHODS if args.layer == '2' else ['output_only', 'pointwise_gate', 'scan'])
    pairs = [args.pair.split(':')] if args.pair else None
    rows = jobs(dataset, methods, args.seeds, args.epochs, pairs)
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'runs': rows, 'count': len(rows),
                          'settings': str(s['settings_file']), 'gpu_training_started': False},
                         indent=2, ensure_ascii=False))
        return
    # Validate the whole selected queue before starting the first job.
    preflight(s, rows)
    if importlib.util.find_spec('torch') is None:
        raise SystemExit('PyTorch is unavailable; activate the training environment first')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(PACKAGE/'tests'),
                    '-p', 'test_controls_torch.py', '-v'], check=True,
                   env={**__import__('os').environ, 'CFA_TEST_SETTINGS': str(s['settings_file'])})
    stage = 'layer2' if args.layer == '2' else 'layer3'
    for row in rows:
        out = s['output_root']/stage/'runs'/row['run_id']/f'attempt_{args.attempt:02d}'
        if out.exists():
            raise SystemExit(f'Existing run: {out}. No automatic skip; audit completed runs and select missing jobs explicitly.')
    for index, row in enumerate(rows, 1):
        command = [sys.executable, str(PACKAGE/'common/launch.py'), '--settings', str(s['settings_file']),
                   '--layer', args.layer, '--action', 'train', '--dataset', dataset,
                   '--source', row['source'], '--target', row['target'], '--method', row['method'],
                   '--seed', str(row['seed']), '--epochs', str(row['epochs']), '--attempt', str(args.attempt)]
        print(f'[{index}/{len(rows)}] {row["run_id"]}', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
