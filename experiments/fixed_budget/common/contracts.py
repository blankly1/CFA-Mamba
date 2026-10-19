"""Dependency-free plan resolution and provenance helpers."""
import csv
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
CLIP_SHA = '1f16b1274ca25493b1f36225efe81d05f66a3f33c487acd1a07c5ba87920de6f'
DOMAINS = {'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'],
           'VISDA-C': ['train', 'validation'], 'office': ['amazon', 'dslr', 'webcam'],
           'domainnet126': ['clipart', 'painting', 'real', 'sketch']}
DATA_SUBDIR = {'office-home': 'officeHome/OfficeHome', 'VISDA-C': 'VISDA-C',
               'office': 'office31', 'domainnet126': 'Domainnet'}
CLASSES = {'office-home': 65, 'VISDA-C': 12, 'office': 31, 'domainnet126': 126}
CORE_PAIRS = [('Art', 'Clipart'), ('Clipart', 'Product'),
              ('Product', 'RealWorld'), ('RealWorld', 'Art')]
METHODS = ['sfmamba_original', 'output_only', 'pointwise_gate', 'conditional_mlp', 'reset_state', 'scan']
CFA_METHODS = METHODS[1:] + ['shared_condition_scan', 'no_clip_output']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def settings(path):
    path = Path(path).resolve()
    s = read_json(path)
    workspace = Path(s['workspace']) if s['workspace'] != 'auto' else PACKAGE.parents[1]
    workspace = workspace.resolve()
    code = s['code_root']
    if code == 'auto':
        options = [workspace]
        found = [p for p in options if (p / 'repository/cfa_mamba/train.py').is_file()]
        if len(found) != 1:
            raise ValueError(f'Expected exactly one code root; found {found}. Set code_root explicitly.')
        code = found[0]
    else:
        code = Path(code)
        if not code.is_absolute():
            code = workspace / code
    result = {'workspace': workspace, 'code_root': Path(code).resolve(), 'settings_file': path}
    for key in ('dataset_root', 'source_root', 'clip_weights', 'output_root'):
        p = Path(s[key])
        result[key] = (p if p.is_absolute() else workspace / p).resolve()
    for key in ('baseline_profile', 'source_bundles', 'class_mapping_attestations'):
        value = s.get(key)
        p = Path(value) if value else None
        result[key] = (p if p is None or p.is_absolute() else path.parent / p)
    return result


def profile(dataset, epochs=15):
    if dataset not in ('office-home', 'VISDA-C') or epochs not in (15, 30):
        raise ValueError('Training is restricted to the specified OH/VisDA 15/30 epoch protocols.')
    visda = dataset == 'VISDA-C'
    return dict(dataset=dataset, epochs=epochs, batch_size=32, test_batch_size=16,
                workers=2 if visda else 8, lr=3e-6 if visda else 3e-5,
                lr_factor1=0.1, lr_factor2=0.1 if visda else 1.0,
                gate_lr_factor=0.1 if visda else 1.0,
                lambda_pseudo=1.0 if visda else 0.5,
                lambda_consistency=1.0 if visda else 0.3,
                lambda_entropy=0.05 if visda else 0.1,
                lambda_diversity=1.0, lambda_anchor=0.1,
                clip_conf_threshold=0.55 if visda else 0.78,
                metric='mean_class_accuracy_percent' if visda else 'top1_accuracy_percent',
                clip_temperature=0.01, clip_warmup_epochs=0, clip_min_weight=1.0,
                warmup_lr=3e-7, min_lr=1.5e-7 if visda else 1.5e-6,
                selection='fixed_final_endpoint', source_seed_policy='shared_verified_bundle')


def run_id(dataset, source, target, method, seed, epochs):
    prefix = ('OH' if dataset == 'office-home' else 'VISDA') + str(epochs)
    return f'{prefix}__{source}-to-{target}__{method}__seed{seed}'


def jobs(dataset, methods, seeds, epochs, pairs=None):
    profile(dataset, epochs)
    if 'sfmamba_original' in methods and (dataset != 'office-home' or epochs != 15):
        raise ValueError('Audited SfMamba baseline is restricted to Office-Home, 15 epochs')
    if not methods or not set(methods) <= set(METHODS + CFA_METHODS):
        raise ValueError('Unknown or empty methods')
    if not seeds or not set(seeds) <= {0, 1, 2}:
        raise ValueError('Only preregistered seeds 0/1/2 are supported')
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate methods or seeds are forbidden')
    pairs = pairs or (CORE_PAIRS if dataset == 'office-home' else [('train', 'validation')])
    rows = []
    for seed in seeds:
        for i, (source, target) in enumerate(pairs):
            if source == target or source not in DOMAINS[dataset] or target not in DOMAINS[dataset]:
                raise ValueError('Invalid domain pair')
            offset = (seed + i) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                rows.append(dict(dataset=dataset, source=source, target=target, method=method,
                                 seed=seed, epochs=epochs,
                                 run_id=run_id(dataset, source, target, method, seed, epochs)))
    return rows


def preflight(s, rows, require_cache=True):
    errors = []
    for dataset in sorted({r['dataset'] for r in rows}):
        p = s['dataset_root'] / DATA_SUBDIR[dataset]
        if not p.is_dir():
            errors.append(f'Missing dataset: {p}')
    if not s['clip_weights'].is_file():
        errors.append(f'Missing CLIP weights: {s["clip_weights"]}')
    elif sha(s['clip_weights']) != CLIP_SHA:
        errors.append('CLIP weight hash differs from historical teacher')
    bundles = read_json(s['source_bundles']) if s['source_bundles'].is_file() else {}
    for dataset, source in sorted({(r['dataset'], r['source']) for r in rows}):
        key = f'{dataset}/{source}'
        attested = bundles.get(key, {})
        if not attested.get('reviewed') or not attested.get('class_names') or not attested.get('provenance'):
            errors.append(f'Unregistered source bundle {key}; see source_bundles instructions')
        elif len(attested['class_names']) != CLASSES[dataset]:
            errors.append(f'Source bundle class count mismatch: {key}')
        for part in ('F', 'B', 'C'):
            p = s['source_root'] / dataset / source / f'source_{part}.pt'
            if not p.is_file():
                errors.append(f'Missing source part: {p}')
            elif attested.get('sha256', {}).get(p.name) != sha(p):
                errors.append(f'Source hash not attested: {p}')
    if require_cache:
        for dataset, target in sorted({(r['dataset'], r['target']) for r in rows}):
            p = s['output_root'] / 'layer1/cache' / dataset / target
            if not (p / 'cache.pt').is_file() or not (p / 'manifest.json').is_file():
                errors.append(f'Missing audited cache: {p}; run layer1 cache first')
    if any(r['method'] == 'sfmamba_original' for r in rows):
        p = s['baseline_profile']
        if p is None or not p.is_file():
            errors.append('Provide baseline_profile for original SfMamba; no default fallback')
        else:
            b = read_json(p)
            expected = {'par_ent', 'par_cls', 'par_st', 'div_loss', 'pseu_target', 'sel',
                        'clean_ratio', 'su_cl_t', 'add_contrast', 'reg_weight'}
            if not b.get('reviewed') or not b.get('evidence') or not expected <= b.keys():
                errors.append('SfMamba profile is not fully reviewed')
            if set(b) - expected - {'reviewed', 'evidence'}:
                errors.append('Unknown SfMamba profile fields; shared training settings cannot be overridden')
            if b.get('pseu_target') == 'target':
                errors.append('Target-label pseudo targets are forbidden')
            if b.get('par_cls', 0) <= 0 or not b.get('sel'):
                errors.append('Original runner requires par_cls>0 and sel=true; audit other profiles separately')
    if errors:
        raise RuntimeError('\n'.join(errors))


def csv_write(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
