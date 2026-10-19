"""Non-GPU path and dependency check; no source loading and no training."""
import argparse
import json
from contracts import PACKAGE, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--settings', default=str(PACKAGE / 'settings.json'))
    args = p.parse_args()
    s = settings(args.settings)
    report = {'workspace': str(s['workspace']), 'code_root': str(s['code_root']),
              'dataset_root': str(s['dataset_root']), 'source_root': str(s['source_root']),
              'clip_weights': str(s['clip_weights']),
              'code_root_exists': s['code_root'].is_dir(),
              'dataset_root_exists': s['dataset_root'].is_dir(),
              'clip_weights_exists': s['clip_weights'].is_file(),
              'gpu_training_started': False}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['code_root_exists'] or not report['dataset_root_exists']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
