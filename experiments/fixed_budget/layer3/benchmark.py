"""Measure a completed model on the same GPU; never overwrites its checkpoint."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from contracts import PACKAGE, settings, read_json, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', default=str(PACKAGE/'settings.json'))
    p.add_argument('--run-dir', required=True)
    p.add_argument('--mode', choices=['inference', 'uncached_inference', 'train_steps'], default='inference')
    p.add_argument('--batch', type=int, choices=[1, 16], default=16,
                   help='Inference batch; train_steps always uses 32')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'mode':args.mode, 'run_dir':args.run_dir,
                          'batch':32 if args.mode=='train_steps' else args.batch,
                          'warmup':30 if args.mode=='train_steps' else 50,
                          'measured_iterations':100 if args.mode=='train_steps' else 200,
                          'blocks':5, 'gpu_started':False}, indent=2))
        return
    import torch
    from runtime import (configure, datasets, cache_load, load_source, make_student, loader,
                         seed_all, environment, state_hash)
    s = settings(args.settings)
    run_dir = Path(args.run_dir).resolve()
    record = read_json(run_dir/'result.json')
    if record['status'] != 'completed':
        raise ValueError('A completed fixed endpoint is required')
    if record['method'] == 'sfmamba_original':
        raise ValueError('This benchmark is for CFA structural variants, not the original algorithm')
    if sha(run_dir/'endpoint.pt') != record['endpoint_sha256']:
        raise ValueError('Checkpoint was modified')
    original, options, config, _ = configure(s, record['dataset'], record['source'], record['target'],
                                             record['seed'], record['epochs'])
    training, evaluation = datasets(config)
    probs, cache_meta, contract = cache_load(s, config, evaluation)
    seed_all(0)
    base, source_id = load_source(s, config, original, contract['names'])
    if source_id != record['comparison_contract']['source_bundle_sha256']:
        raise ValueError('Source bundle differs from measured run')
    import copy
    anchor = copy.deepcopy(base).eval()
    for v in anchor.parameters():
        v.requires_grad_(False)
    if args.mode != 'train_steps':
        del anchor
    student = make_student(original, base, record['method'], probs, record['seed'])
    checkpoint = torch.load(run_dir/'endpoint.pt', map_location='cpu', weights_only=False)
    batch = 32 if args.mode == 'train_steps' else args.batch
    source_loader = loader(training if args.mode=='train_steps' else evaluation,
                           batch, config.DATA.NUM_WORKERS, record['seed'])
    # Keep a bounded real-image bank; transfer is explicitly outside the timing scope.
    bank = []
    teacher = None
    teacher_iterator = None
    if args.mode == 'uncached_inference':
        from cfa_mamba.clip_teacher import ClipSemanticTeacher
        from datautil.imgdata.imgdataload import ImageDataset
        teacher = ClipSemanticTeacher(contract['names'], model_name='ViT-B-16', backend='open_clip',
                                      pretrained=str(s['clip_weights']), temperature=0.01, device='cuda')
        clip_dataset=ImageDataset(config.dataset,config.DATA.DATA_PATH,record['target'],
                                  transform=teacher.preprocess,return_idx=True)
        teacher_iterator=iter(loader(clip_dataset,batch,config.DATA.NUM_WORKERS,record['seed']))
    for images, labels, indices in source_loader:
        if teacher_iterator is not None:
            clip_images, _, clip_indices=next(teacher_iterator)
            if not torch.equal(indices,clip_indices):
                raise ValueError('Teacher/student timing inputs differ')
            condition=clip_images.cuda()
        else:
            condition=probs[indices].cuda()
        bank.append((images.cuda(), condition))
        if len(bank) == 4:
            break
    if len(bank) == 0:
        raise ValueError('Empty dataset')
    criterion = original.CGMambaLoss(lambda_pseudo=options.lambda_pseudo,
                                     lambda_consistency=options.lambda_consistency,
                                     lambda_entropy=options.lambda_entropy,
                                     lambda_diversity=options.lambda_diversity,
                                     lambda_anchor=options.lambda_anchor,
                                     confidence_threshold=options.clip_conf_threshold)
    if record['method'] == 'no_clip_output':
        criterion.lambda_pseudo = criterion.lambda_consistency = 0
    blocks = []
    for block in range(5):
        seed_all(record['seed'] + 40000)
        student.load_state_dict(checkpoint['model'], strict=True)
        student.train(args.mode == 'train_steps')
        optimizer = None
        if args.mode == 'train_steps':
            import logging
            optimizer = original.build_cg_optimizer(config, options, student, logging.getLogger('benchmark'))
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler = original.NativeScalerWithGradNormCount()
            scaler.load_state_dict(checkpoint['scaler'])
        warmup, measured = (30, 100) if optimizer else (50, 200)
        iterator = itertools.cycle(bank)

        def step():
            images, condition = next(iterator)
            if optimizer is None:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    if teacher is not None:
                        condition=teacher(condition)
                    student(images, clip_probs=condition)
            else:
                optimizer.zero_grad()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    source_logits = anchor(images)
                with torch.cuda.amp.autocast(enabled=True):
                    loss = criterion(student(images, clip_probs=condition), condition,
                                     clip_weight=1.0, source_logits=source_logits).total
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite timing loss')
                scaler(loss, optimizer, clip_grad=5.0, parameters=student.parameters(),
                       create_graph=False, update_grad=True)
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(measured):
            step()
        end.record()
        torch.cuda.synchronize()
        blocks.append(dict(block=block, cuda_ms_per_batch=start.elapsed_time(end)/measured,
                           wall_ms_per_batch=1000*(time.perf_counter()-wall)/measured,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    destination = s['output_root']/'layer3/benchmarks'/record['run_id']/f'{args.mode}_batch{batch}.json'
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, dict(provenance='measured', mode=args.mode, batch=batch,
                                 run_id=record['run_id'], endpoint_sha256=record['endpoint_sha256'],
                                 scope='resident real-image bank; excludes decoding, transfers and evaluation; uncached mode includes CLIP forward only',
                                 optimizer_mode='endpoint optimizer state; fixed LR during microbenchmark',
                                 memory_scope='resident bank; source anchor only for train_steps; CLIP only for uncached_inference',
                                 includes_source_anchor=args.mode=='train_steps', blocks=blocks,
                                 median_cuda_ms=statistics.median(b['cuda_ms_per_batch'] for b in blocks),
                                 mean_wall_ms=statistics.mean(b['wall_ms_per_batch'] for b in blocks),
                                 environment=environment()))
    print(destination)


if __name__ == '__main__':
    main()
