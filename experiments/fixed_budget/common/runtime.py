"""Measured experiment runtime. Imported only by explicit GPU commands."""
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from contracts import (CLASSES, CLIP_SHA, DATA_SUBDIR, DOMAINS, PACKAGE, digest,
                       profile, read_json, sha, write_json)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class AuditSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)
        self.hashes = []

    def __iter__(self):
        indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        self.hashes.append(digest(indices))
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def loader(dataset, batch, workers, seed, training=False):
    return DataLoader(dataset, batch_size=batch, shuffle=False,
                      sampler=AuditSampler(dataset, seed) if training else None,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed + 10000), drop_last=False)


def initialize(s):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. Use --plan/--check without GPU instead.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(s['code_root'] / 'repository'))
    sys.path.insert(1, str(s['code_root'] / 'third_party/selective_scan'))
    from cfa_mamba import train as original
    for module in ('cfa_mamba.model', 'cfa_mamba.losses', 'models', 'config'):
        imported = __import__(module, fromlist=['*'])
        if not Path(imported.__file__).resolve().is_relative_to(s['code_root']):
            raise RuntimeError(f'Contaminated import: {module}: {imported.__file__}')
    return original


def configure(s, dataset, source, target, seed=0, epochs=15, output=None):
    original = initialize(s)
    p = profile(dataset, epochs) if dataset in ('office-home', 'VISDA-C') else profile('office-home')
    data_root = s['dataset_root'] / DATA_SUBDIR[dataset]
    argv = ['runner', '--cfg', str(s['code_root'] / 'repository/configs/vmambav2_small_224.yaml'),
            '--dataset', dataset, '--source_env', str(DOMAINS[dataset].index(source)),
            '--target_env', str(DOMAINS[dataset].index(target)),
            '--data_path', str(data_root) + '/', '--output', str(output or s['output_root'] / '_eval'),
            '--batch_size', '32', '--test_batch_size', '16', '--dg_aug', 'True',
            '--neck', 'bn_relu', '--fusion_module', 'ssm', '--clip_backend', 'open_clip',
            '--clip_model', 'ViT-B-16', '--clip_pretrained', str(s['clip_weights']),
            '--clip_temperature', '0.01', '--clip_batch_size', '128', '--clip_num_workers', '2',
            '--gate_hidden_dim', '0', '--gate_dropout', '0.0', '--adapter_type', 'scan']
    for key in ('lr', 'lr_factor1', 'lr_factor2', 'gate_lr_factor', 'lambda_pseudo',
                'lambda_consistency', 'lambda_entropy', 'lambda_diversity', 'lambda_anchor',
                'clip_conf_threshold', 'clip_warmup_epochs', 'clip_min_weight'):
        argv += ['--' + key, str(p[key])]
    argv += ['MODEL.SOURCE_DIR', str(s['source_root']), 'DATA.NUM_WORKERS', str(p['workers']),
             'TRAIN.EPOCHS', str(epochs), 'TRAIN.WARMUP_EPOCHS', '1',
             'TRAIN.WARMUP_LR', str(p['warmup_lr']), 'TRAIN.MIN_LR', str(p['min_lr']),
             'TRAIN.ACCUMULATION_STEPS', '1', 'SEED', str(seed)]
    saved = sys.argv
    try:
        sys.argv = argv
        args, config = original.parse_option()
    finally:
        sys.argv = saved
    config.defrost()
    if output is not None:
        config.OUTPUT = str(output)
    config.freeze()
    return original, args, config, argv


def datasets(config):
    from datautil.imgdata.imgdataload import ImageDataset
    from datautil.imgdata.util import image_train, img_test
    domain = config.img_dataset[config.target_env]
    common = (config.dataset, config.DATA.DATA_PATH, domain)
    return (ImageDataset(*common, transform=image_train(True), return_idx=True),
            ImageDataset(*common, transform=img_test(True), return_idx=True))


def data_contract(s, config, dataset, full_hash=True):
    from cfa_mamba.class_names import infer_class_names
    names = infer_class_names(config)
    root = Path(config.DATA.DATA_PATH).resolve()
    records, observed = [], {}
    for i, (path, label) in enumerate(zip(dataset.x, dataset.labels)):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid image path: {path}')
        label = int(label)
        if label < 0 or label >= len(names):
            raise ValueError('Class index out of range')
        observed.setdefault(label, set()).add(path.parent.name)
        records.append(dict(index=i, path=path.relative_to(root).as_posix(),
                            size=path.stat().st_size, sha256=sha(path) if full_hash else None))
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Duplicate sample paths')
    if set(observed) != set(range(len(names))):
        raise ValueError('Missing target class or incorrect class mapping')
    def normalize(name):
        return ''.join(c.lower() for c in name if c.isalnum())
    mismatches = {str(i): {'observed': sorted(v), 'teacher': names[i]}
                  for i, v in observed.items() if {normalize(n) for n in v} != {normalize(names[i])}}
    observed_map = {str(i): sorted(v) for i, v in observed.items()}
    mapping_id = digest({'teacher': names, 'observed': observed_map})
    key = f'{config.dataset}/{config.img_dataset[config.target_env]}'
    if mismatches:
        approval = read_json(s['class_mapping_attestations']).get(key, {})
        if approval.get('mapping_sha256') != mapping_id or not approval.get('reviewed') or not approval.get('reason'):
            raise ValueError(f'Class-name aliases need semantic review for {key}. '
                             f'mapping_sha256={mapping_id}; mismatches={mismatches}')
    return dict(dataset=config.dataset, domain=config.img_dataset[config.target_env],
                records=records, names=names, mapping_sha256=mapping_id,
                images_sha256=digest(records), labels_sha256=digest(dataset.labels.tolist()),
                classes_sha256=digest(names), sample_count=len(records), full_hash=full_hash)


def environment():
    versions = {}
    for name in ('torch', 'torchvision', 'timm', 'open_clip_torch', 'numpy', 'yacs'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), versions=versions,
                tf32=False, cudnn_deterministic=True,
                note='Custom selective_scan may not be bitwise deterministic')


def cache_location(s, dataset, target):
    return s['output_root'] / 'layer1/cache' / dataset / target


def tensor_hash(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def cache_build(s, dataset, target):
    source = next(d for d in DOMAINS[dataset] if d != target)
    original, args, config, argv = configure(s, dataset, source, target)
    folder = cache_location(s, dataset, target)
    if folder.exists():
        raise FileExistsError(f'Cache output already exists; inspect it, do not overwrite: {folder}')
    if sha(s['clip_weights']) != CLIP_SHA:
        raise ValueError('Wrong CLIP weights')
    from cfa_mamba.clip_teacher import ClipSemanticTeacher, DEFAULT_PROMPT_TEMPLATES
    _, evaluation = datasets(config)
    contract = data_contract(s, config, evaluation)
    seed_all(0)
    teacher = ClipSemanticTeacher(contract['names'], model_name='ViT-B-16', backend='open_clip',
                                  pretrained=str(s['clip_weights']), temperature=0.01, device='cuda')
    clip_dataset, clip_loader = original.build_clip_cache_loader(config, args, teacher)
    if clip_dataset.x != evaluation.x or not np.array_equal(clip_dataset.labels, evaluation.labels):
        raise ValueError('Teacher and evaluation sample order differ')
    started = time.perf_counter()
    probs = teacher.build_cache(clip_loader, len(clip_dataset), cache_path='', amp=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    validate_probs(probs, len(evaluation), len(contract['names']))
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(probs, folder / 'cache.pt')
    write_json(folder / 'samples.json', contract)
    manifest = dict(contract_sha256=digest(contract), images_sha256=contract['images_sha256'],
                    labels_sha256=contract['labels_sha256'], classes_sha256=contract['classes_sha256'],
                    mapping_sha256=contract['mapping_sha256'], tensor_sha256=tensor_hash(probs),
                    cache_file_sha256=sha(folder / 'cache.pt'), weights_sha256=CLIP_SHA,
                    temperature=0.01, prompts=DEFAULT_PROMPT_TEMPLATES,
                    preprocess=repr(teacher.preprocess), environment=environment(),
                    teacher_code_sha256=sha(s['code_root']/'repository/cfa_mamba/clip_teacher.py'),
                    measured_cache_seconds=elapsed, provenance='measured_cache_generation')
    write_json(folder / 'manifest.json', manifest)
    write_json(folder / 'class_names.json', contract['names'])
    result = evaluate_probabilities(probs.numpy(), evaluation.labels, dataset)
    write_json(folder / 'teacher_result.json', dict(result, cache_manifest_sha256=digest(manifest)))
    np.savez_compressed(folder / 'teacher_predictions.npz', dataset_index=np.arange(len(evaluation)),
                        probs=probs.numpy(), labels=evaluation.labels)
    return result


def validate_probs(probs, n, c):
    if tuple(probs.shape) != (n, c) or not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError('Invalid cache dimensions or probabilities')
    if not torch.allclose(probs.sum(1), torch.ones(n), atol=1e-5, rtol=0):
        raise ValueError('Cache probabilities do not sum to one')


def cache_load(s, config, evaluation):
    contract = data_contract(s, config, evaluation)
    folder = cache_location(s, config.dataset, config.img_dataset[config.target_env])
    meta = read_json(folder / 'manifest.json')
    for name in ('images_sha256', 'labels_sha256', 'classes_sha256', 'mapping_sha256'):
        if meta[name] != contract[name]:
            raise ValueError(f'Cache contract mismatch: {name}')
    if meta['weights_sha256'] != CLIP_SHA or meta['temperature'] != 0.01:
        raise ValueError('Teacher contract mismatch')
    if meta['environment']['versions'] != environment()['versions']:
        raise ValueError('Cache environment differs; do not silently mix library versions')
    if meta['teacher_code_sha256'] != sha(s['code_root']/'repository/cfa_mamba/clip_teacher.py'):
        raise ValueError('Teacher implementation changed')
    if sha(folder / 'cache.pt') != meta['cache_file_sha256']:
        raise ValueError('Cache file hash mismatch')
    probs = torch.load(folder / 'cache.pt', map_location='cpu', weights_only=True)
    validate_probs(probs, len(evaluation), config.MODEL.NUM_CLASSES)
    if tensor_hash(probs) != meta['tensor_sha256']:
        raise ValueError('Cache tensor changed')
    return probs, meta, contract


def load_source(s, config, original, names):
    model = original.build_model(config).cuda()
    key = f'{config.dataset}/{config.domains[config.source_env[0]]}'
    registry = read_json(s['source_bundles']).get(key, {})
    if not registry.get('reviewed') or registry.get('class_names') != names:
        raise ValueError(f'Source class order not explicitly verified: {key}')
    parts = {}
    for part in ('F', 'B', 'C'):
        path = Path(getattr(config.MODEL, 'SOURCE_' + part))
        if sha(path) != registry.get('sha256', {}).get(path.name):
            raise ValueError(f'Source bundle hash mismatch: {path}')
        parts[part] = torch.load(path, map_location='cpu', weights_only=True)
    expected_f = set(model.patch_embed.state_dict()) | set(model.layers.state_dict())
    if set(parts['F']) != expected_f:
        raise ValueError('Source F keys differ from complete patch_embed/layers union')
    for module in (model.patch_embed, model.layers):
        module.load_state_dict({k: parts['F'][k] for k in module.state_dict()}, strict=True)
    model.fusion_blocks.load_state_dict(parts['B'], strict=True)
    model.classifier.load_state_dict(parts['C'], strict=True)
    del parts
    return model, digest(registry)


def state_hash(model):
    return digest({k: tensor_hash(v) for k, v in model.state_dict().items()})


def make_student(original, base, method, probs, seed):
    from variants import NoAdapter, ConditionalMLP, ResetState, SharedCondition
    seed_all(20000 + seed)
    if method == 'sfmamba_original':
        return base
    student = original.ClipGuidedSfMamba(base, num_classes=probs.shape[1], adapter_type='scan',
                                       gate_hidden_dim=None, gate_dropout=0.0)
    reference = student.semantic_gate.state_dict()
    kwargs = dict(num_classes=probs.shape[1], feature_dim=base.num_features,
                  hidden_dim=base.num_features, dropout=0.0)
    seed_all(20000 + seed)
    if method == 'output_only':
        student.semantic_gate = NoAdapter()
    elif method == 'pointwise_gate':
        from cfa_mamba.model import SemanticGatedMambaAdapter
        student.semantic_gate = SemanticGatedMambaAdapter(**kwargs)
    elif method == 'conditional_mlp':
        student.semantic_gate = ConditionalMLP(**kwargs)
    elif method == 'reset_state':
        student.semantic_gate = ResetState(**kwargs)
    elif method == 'shared_condition_scan':
        student.semantic_gate = SharedCondition(probs.mean(0), **kwargs)
    elif method not in ('scan', 'no_clip_output'):
        raise ValueError(method)
    own = student.semantic_gate.state_dict()
    common = {k: v for k, v in reference.items() if k in own and own[k].shape == v.shape}
    student.semantic_gate.load_state_dict(common, strict=False)
    return student.cuda()


def evaluate_probabilities(probs, labels, dataset):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(probs, axis=1)
    c = CLASSES[dataset]
    counts = np.bincount(labels, minlength=c)
    if len(counts) != c or np.any(counts == 0):
        raise ValueError('Evaluation must contain all declared classes')
    correct = np.bincount(labels[preds == labels], minlength=c)
    per_class = 100 * correct / counts
    top1 = 100 * float((preds == labels).mean())
    primary = float(per_class.mean()) if dataset == 'VISDA-C' else top1
    return dict(accuracy_percent=primary, top1_percent=top1,
                metric='mean_class_accuracy_percent' if dataset == 'VISDA-C' else 'top1_accuracy_percent',
                eval_count=len(labels), class_accuracy_percent=per_class.tolist())


@torch.no_grad()
def evaluate(model, evaluation_loader, probs, dataset, method, path=None):
    state = rng_state()
    was_training = model.training
    try:
        model.eval()
        n, c = probs.shape
        logits = torch.empty(n, c)
        labels = torch.empty(n, dtype=torch.long)
        seen = torch.zeros(n, dtype=torch.int32)
        for images, y, indices in evaluation_loader:
            indices = indices.long()
            with torch.cuda.amp.autocast(enabled=True):
                output = (model(images.cuda(non_blocking=True)) if method == 'sfmamba_original'
                          else model(images.cuda(non_blocking=True), clip_probs=probs[indices].cuda()))
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite evaluation logits')
            logits[indices] = output.float().cpu()
            labels[indices] = y
            seen[indices] += 1
        if not (seen == 1).all():
            raise ValueError('Evaluation indices are not a permutation of the target set')
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, dataset_index=np.arange(n), logits=logits.numpy(), labels=labels.numpy())
        return evaluate_probabilities(logits.numpy(), labels.numpy(), dataset), logits, labels
    finally:
        model.train(was_training)
        restore_rng(state)


def references(s, dataset, source, target):
    original, args, config, argv = configure(s, dataset, source, target)
    _, evaluation = datasets(config)
    probs, cache_meta, contract = cache_load(s, config, evaluation)
    seed_all(0)
    base, source_id = load_source(s, config, original, contract['names'])
    output = s['output_root'] / 'layer1/references' / dataset / f'{source}_{target}'
    output.mkdir(parents=True, exist_ok=False)
    result, logits, labels = evaluate(base, loader(evaluation, 16, config.DATA.NUM_WORKERS, 0),
                                      probs, dataset, 'sfmamba_original', output/'source_predictions.npz')
    ensemble = 0.5 * logits.softmax(1) + 0.5 * probs
    ensemble_result = evaluate_probabilities(ensemble.numpy(), labels.numpy(), dataset)
    np.savez_compressed(output/'ensemble_predictions.npz', dataset_index=np.arange(len(evaluation)),
                        probs=ensemble.numpy(), labels=labels.numpy())
    write_json(output/'results.json', dict(source=result, ensemble=ensemble_result,
                                           source_bundle_sha256=source_id, cache=cache_meta,
                                           data_sha256=contract['images_sha256'], provenance='measured'))
    return dict(source=result, ensemble=ensemble_result, output=str(output))


def code_manifest(s):
    roots = [('package', PACKAGE), ('repository', s['code_root'] / 'repository'),
             ('selective_scan', s['code_root'] / 'third_party/selective_scan')]
    files = {}
    for label, root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.yaml', '.cpp', '.cu', '.h', '.cuh', '.so', '.pyd'):
                if '__pycache__' not in path.parts and path.name not in ('source_bundles.json', 'class_mapping_attestations.json'):
                    files[label+'/'+path.relative_to(root).as_posix()] = sha(path)
    return files


class TrackingScaler:
    def __init__(self, original_scaler, stream):
        self.inner = original_scaler
        self.stream = stream
        self.attempted = 0
        self.successful = 0

    def __call__(self, loss, optimizer, **kwargs):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        before = self.inner.state_dict()['scale']
        lrs = [dict(lr=g['lr'], weight_decay=g['weight_decay']) for g in optimizer.param_groups]
        started = time.perf_counter()
        result = self.inner(loss, optimizer, **kwargs)
        after = self.inner.state_dict()['scale']
        self.attempted += 1
        self.successful += int(after >= before)
        self.stream.write(json.dumps(dict(step=self.attempted, loss=float(loss.detach()),
                                         scaler_before=before, scaler_after=after,
                                         update_skipped=after < before, groups=lrs,
                                         seconds=time.perf_counter()-started))+'\n')
        return result

    def state_dict(self):
        return self.inner.state_dict()


def train(s, row, attempt):
    stage = 'layer2' if row['dataset'] == 'office-home' else 'layer3'
    out = s['output_root'] / stage / 'runs' / row['run_id'] / f'attempt_{attempt:02d}'
    if out.exists():
        raise FileExistsError(f'No overwrite or best-checkpoint resume: {out}')
    original, args, config, argv = configure(s, row['dataset'], row['source'], row['target'],
                                             row['seed'], row['epochs'], out)
    training, evaluation = datasets(config)
    probs, cache_meta, contract = cache_load(s, config, evaluation)
    if training.x != evaluation.x:
        raise ValueError('Train/eval sample order differs')
    seed_all(0)
    base, source_id = load_source(s, config, original, contract['names'])
    initial_hash = state_hash(base)
    anchor = copy.deepcopy(base).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    anchor_hash = state_hash(anchor)
    student = make_student(original, base, row['method'], probs, row['seed'])
    initial_classifier = state_hash(base.classifier)
    out.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(row['run_id'])
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(out/'stdout.log', encoding='utf-8'))
    logger.addHandler(logging.StreamHandler())
    seed_all(30000 + row['seed'])
    train_loader = loader(training, 32, config.DATA.NUM_WORKERS, row['seed'], True)
    eval_loader = loader(evaluation, 16, config.DATA.NUM_WORKERS, row['seed'])
    baseline = None
    if row['method'] == 'sfmamba_original':
        import main_target as baseline
        baseline.logger = logger
        b = read_json(s['baseline_profile'])
        for key in ('par_ent', 'par_cls', 'par_st', 'div_loss', 'pseu_target',
                    'sel', 'clean_ratio', 'su_cl_t', 'add_contrast', 'reg_weight'):
            setattr(args, key, b[key])
        optimizer = baseline.build_optimizer(config, student, logger)
        criterion = baseline.myCriterion(div=args.div_loss, par_ent=args.par_ent,
                                         par_cls=args.par_cls, par_st=args.par_st)
    else:
        optimizer = original.build_cg_optimizer(config, args, student, logger)
        criterion = original.CGMambaLoss(
            lambda_pseudo=0 if row['method']=='no_clip_output' else args.lambda_pseudo,
            lambda_consistency=0 if row['method']=='no_clip_output' else args.lambda_consistency,
            lambda_entropy=args.lambda_entropy, lambda_diversity=args.lambda_diversity,
            lambda_anchor=args.lambda_anchor, confidence_threshold=args.clip_conf_threshold)
    scheduler = original.build_scheduler(config, optimizer, len(train_loader))
    names = {id(p): name for name, p in student.named_parameters()}
    groups = [dict(parameters=[names[id(p)] for p in g['params']],
                   lr=g['lr'], weight_decay=g['weight_decay']) for g in optimizer.param_groups]
    code = code_manifest(s)
    common = dict(protocol='NC_FIXED_20260930_RUNTIME_v2', profile=profile(row['dataset'], row['epochs']),
                  source_bundle_sha256=source_id, initial_base_sha256=initial_hash,
                  data_sha256=contract['images_sha256'], class_sha256=contract['classes_sha256'],
                  cache_sha256=cache_meta['tensor_sha256'], code_sha256=digest(code),
                  environment=environment(), source_anchor_convention='same_augmented_training_image',
                  selection='fixed_final_endpoint', data_order='independent_sampler_and_worker_generators')
    record = dict(row, status='running', comparison_contract=common,
                  comparison_contract_sha256=digest(common), provenance='measured_run_in_progress',
                  adapter_parameters=sum(p.numel() for p in student.semantic_gate.parameters()) if baseline is None else 0,
                  effective_adapter_parameters=sum(p.numel() for p in student.semantic_gate.parameters() if p.requires_grad) if baseline is None else 0)
    if baseline:
        record['baseline_profile'] = read_json(s['baseline_profile'])
    write_json(out/'run_manifest.json', record)
    write_json(out/'code_hashes.json', code)
    write_json(out/'optimizer_groups.json', groups)
    write_json(out/'args.resolved.json', vars(args))
    (out/'config.resolved.yaml').write_text(config.dump(), encoding='utf-8')
    write_json(out/'command.json', dict(canonical_argv=argv, launcher_argv=sys.argv))
    epoch_metrics = []
    started = time.perf_counter()
    with (out/'train_steps.jsonl').open('w', encoding='utf-8') as step_file:
        scaler = TrackingScaler(original.NativeScalerWithGradNormCount(), step_file)
        try:
            seed_all(30000 + row['seed'])
            evaluate(student, eval_loader, probs, row['dataset'], row['method'], out/'predictions/epoch_00.npz')
            for epoch in range(row['epochs']):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                if baseline:
                    baseline.train_one_epoch(config, args, student, criterion, [train_loader],
                                              [eval_loader], optimizer, epoch, scheduler, scaler)
                else:
                    original.train_one_epoch(config, args, student, anchor, criterion,
                                              [train_loader], optimizer, epoch, scheduler, scaler, probs, logger)
                torch.cuda.synchronize()
                train_seconds = time.perf_counter() - t0
                pred = out/f'predictions/epoch_{epoch+1:02d}.npz'
                keep = epoch+1 in {1,3,5,10,15,30} or epoch+1 == row['epochs']
                result, _, _ = evaluate(student, eval_loader, probs, row['dataset'], row['method'], pred if keep else None)
                epoch_metrics.append(dict(epoch_completed=epoch+1, train_seconds=train_seconds,
                                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, **result))
                write_json(out/'metrics_by_epoch.json', epoch_metrics)
            if state_hash(base.classifier) != initial_classifier or state_hash(anchor) != anchor_hash:
                raise ValueError('Frozen classifier or source anchor changed')
            endpoint = dict(model=student.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            epoch=row['epochs']-1, completed_epochs=row['epochs'], rng=rng_state(),
                            row=row, contract=common,
                            resume_supported=False, note='Worker prefetch state not serialized; restart failed runs from source')
            torch.save(endpoint, out/'endpoint.pt')
            # Reload validation detects incomplete serialization without selecting an epoch.
            checkpoint = torch.load(out/'endpoint.pt', map_location='cpu', weights_only=False)
            student.load_state_dict(checkpoint['model'], strict=True)
            reloaded, _, _ = evaluate(student, eval_loader, probs, row['dataset'], row['method'])
            if reloaded != result:
                raise ValueError('Endpoint reload evaluation differs')
            record.update(status='completed', provenance='measured', result=result,
                          completed_epochs=row['epochs'], attempted_steps=scaler.attempted,
                          successful_updates=scaler.successful, endpoint_sha256=sha(out/'endpoint.pt'),
                          sample_order_hashes=train_loader.sampler.hashes,
                          elapsed_seconds=time.perf_counter()-started,
                          classifier_unchanged=True, source_anchor_unchanged=True)
            write_json(out/'result.json', record)
            inventory={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
                       if p.is_file() and p.name not in ('run_manifest.json','stdout.log','train_steps.jsonl')}
            write_json(out/'artifacts.sha256.json',inventory)
        except BaseException as error:
            record.update(status='failed', failure_reason=repr(error))
            raise
        finally:
            write_json(out/'run_manifest.json', record)
    logger.info('Completed fixed endpoint: %s', out)
    return out
