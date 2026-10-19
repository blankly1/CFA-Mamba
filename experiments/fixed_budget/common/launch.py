"""Command line dispatcher for the three experiment layers."""
import argparse
import json
from pathlib import Path

from contracts import PACKAGE, CORE_PAIRS, METHODS, jobs, preflight, read_json, settings, write_json, csv_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', default=str(PACKAGE / 'settings.json'))
    parser.add_argument('--layer', choices=['1', '2', '3'], required=True)
    parser.add_argument('--action', choices=['check', 'cache', 'references', 'train'], default='check')
    parser.add_argument('--dataset', choices=['office-home', 'VISDA-C', 'office', 'domainnet126'], default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--method', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.layer != '1' and bool(args.source) != bool(args.target):
        parser.error('--source and --target must be supplied together')
    if args.layer == '1' and args.action not in ('check', 'cache', 'references'):
        parser.error('Layer 1 supports check, cache, and references only')
    s = settings(args.settings)

    if args.layer == '1':
        dataset = args.dataset or 'office-home'
        from contracts import DOMAINS, DATA_SUBDIR
        targets = [args.target] if args.target else DOMAINS[dataset]
        if dataset == 'VISDA-C':
            targets = [args.target or 'validation']
        for target in targets:
            if target not in DOMAINS[dataset]:
                raise SystemExit('Invalid target domain')
            if args.action == 'cache':
                from runtime import cache_build
                print(json.dumps(cache_build(s, dataset, target), indent=2))
            elif args.action == 'references':
                from runtime import references
                if not args.source:
                    raise SystemExit('--source is required for references')
                preflight(s, [dict(dataset=dataset, source=args.source, target=target, method='output_only')])
                print(json.dumps(references(s, dataset, args.source, target), indent=2))
            else:
                root = s['dataset_root']/DATA_SUBDIR[dataset]
                print(json.dumps({'layer': 1, 'dataset': dataset, 'target': target,
                                  'action': args.action, 'status': 'path_check_only',
                                  'data_root': str(root), 'data_exists': root.is_dir(),
                                  'weights_exist': s['clip_weights'].is_file(),
                                  'cache_validated': False}, indent=2))
        return

    if args.layer == '2':
        if args.dataset and args.dataset != 'office-home':
            raise SystemExit('Layer 2 is Office-Home only')
        selected_methods = [args.method] if args.method else METHODS
        pairs = [(args.source, args.target)] if args.source and args.target else CORE_PAIRS
        rows = jobs('office-home', selected_methods, [args.seed] if args.seed is not None else [0, 1, 2], args.epochs, pairs)
    else:
        if args.dataset and args.dataset != 'VISDA-C':
            raise SystemExit('Layer 3 training is VisDA only')
        selected_methods = [args.method] if args.method else ['output_only', 'pointwise_gate', 'scan']
        pairs = [(args.source, args.target)] if args.source and args.target else [('train', 'validation')]
        rows = jobs('VISDA-C', selected_methods, [args.seed] if args.seed is not None else [0, 1, 2], args.epochs, pairs)
    if args.action == 'check':
        preflight(s, rows, require_cache=True)
        print(json.dumps({'status': 'preflight_passed', 'runs': len(rows),
                          'warning': 'No training started by check'}, indent=2))
        return
    if args.action != 'train':
        raise SystemExit('For layer 2/3 use --action check or train')
    preflight(s, rows, require_cache=True)
    if len(rows) != 1:
        raise SystemExit('Training launcher requires one explicit run: source, target, method, seed')
    from runtime import train
    if args.attempt < 1:
        raise SystemExit('attempt must be positive')
    output = train(s, rows[0], args.attempt)
    print(output)


if __name__ == '__main__':
    main()
