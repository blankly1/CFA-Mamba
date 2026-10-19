"""Dependency-free source/config/manifest validation. Does not start training."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(part in ('.git', '__pycache__', '.venv', 'venv') for part in p.relative_to(ROOT).parts))
    python_files = [p for p in files if p.suffix == '.py']
    for p in python_files:
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as e:
            errors.append(str(e))
    for p in files:
        if p.suffix == '.json' and not p.name.endswith('.local.json'):
            try:
                json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError) as e:
                errors.append(f'{p.relative_to(ROOT)}: {e}')
        if p.suffix == '.sh' and b'\r\n' in p.read_bytes():
            errors.append(f'Shell script requires LF: {p.relative_to(ROOT)}')
    settings = json.loads((ROOT/'experiments/fixed_budget/settings.json').read_text(encoding='utf-8'))
    if settings['workspace'] != 'auto' or settings['code_root'] != 'auto':
        errors.append('Portable default settings must auto-resolve the repository')
    sys.path.insert(0, str(ROOT/'experiments/fixed_budget/common'))
    from contracts import settings as resolve_settings, jobs, METHODS
    if resolve_settings(ROOT/'experiments/fixed_budget/settings.json')['code_root'] != ROOT:
        errors.append('Additional runner resolves to a different repository')
    if len(jobs('office-home', METHODS, [0,1,2], 15)) != 72:
        errors.append('Unexpected Office-Home matrix')
    if len(jobs('VISDA-C', ['output_only','pointwise_gate','scan'], [0,1,2], 15)) != 9:
        errors.append('Unexpected VisDA matrix')
    for p in (ROOT/'repository/cfa_mamba').glob('train*.py'):
        text = p.read_text(encoding='utf-8')
        if 'Path(__file__).resolve().parents[2]' in text:
            errors.append(f'Legacy root lookup: {p.name}')
        if 'default="openai"' in text:
            errors.append(f'Unexpected CLIP default: {p.name}')
    launcher=(ROOT/'scripts/train_target.sh').read_text(encoding='utf-8')
    if 'CLIP_PRETRAINED:-laion2b_s34b_b88k' not in launcher:
        errors.append('CLIP launcher default differs from the paper')
    manifest=ROOT/'MANIFEST.sha256'
    verified=0
    if manifest.is_file():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            checksum, rel = line.split('  ', 1)
            path=ROOT/rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:
                errors.append(f'Manifest mismatch: {rel}')
            verified+=1
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'status':'passed', 'python_files':len(python_files),
                      'manifest_entries_verified':verified, 'officehome_planned_runs':72,
                      'visda_planned_runs':9, 'gpu_training_started':False}, indent=2))


if __name__ == '__main__':
    main()
