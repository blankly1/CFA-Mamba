from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from timm.utils import AverageMeter

ROOT = Path(__file__).resolve().parents[1]  # repository/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force cg_mamba package to resolve to THIS directory (CFA-Mamba/cg_mamba/),
# overriding the original SfMamba-master/cg_mamba/ that sys.path would otherwise find.
import types as _types
_cg_pkg = _types.ModuleType("cfa_mamba")
_cg_pkg.__path__ = [str(Path(__file__).resolve().parent)]
_cg_pkg.__package__ = "cfa_mamba"
sys.modules["cfa_mamba"] = _cg_pkg

from cfa_mamba.class_names import infer_class_names
from cfa_mamba.clip_teacher import ClipSemanticTeacher
from cfa_mamba.losses import CGMambaLoss
from cfa_mamba.model import ClipGuidedSfMamba
from config import get_config
from datautil.getdataloader import get_tgt_img_dataloader
from datautil.imgdata.imgdataload import ImageDataset
from models import build_model
from utils.dgutil import img_param_init
from utils.logger import create_logger
from utils.lr_scheduler import build_scheduler
from utils.optimizer import check_keywords_in_name
from utils.utils import NativeScalerWithGradNormCount


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_option():
    parser = argparse.ArgumentParser("CFA-Mamba target adaptation", add_help=True)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--optim", type=str)

    parser.add_argument("--dataset", type=str, default="office-home", choices=["office", "office-home", "VISDA-C", "domainnet126", "domainnet"])
    parser.add_argument("--target_env", type=int, default=0)
    parser.add_argument("--source_env", type=int, nargs="+", default=[0])
    parser.add_argument("--split_style", type=str, default="strat")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--issave", action="store_true", default=False)
    parser.add_argument("--dg_aug", type=str2bool, default=True)
    parser.add_argument(
        "--neck",
        type=str,
        default="bn_relu",
        choices=["", "bn", "bn_relu", "bn_ln_relu", "bn_relu_ln", "bottleneck", "eca", "se", "cbam"],
    )
    parser.add_argument("--fusion_module", type=str, default="ssm")
    parser.add_argument("--lr_factor1", type=float, default=0.1)
    parser.add_argument("--lr_factor2", type=float, default=1.0)
    parser.add_argument("--bg_ratio", type=float, default=0.2)

    parser.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "open_clip", "clip"])
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--clip_temperature", type=float, default=0.01)
    parser.add_argument("--clip_batch_size", type=int, default=128)
    parser.add_argument("--clip_num_workers", type=int, default=8)
    parser.add_argument("--clip_cache", type=str, default="")
    parser.add_argument("--force_clip_cache", action="store_true")
    parser.add_argument("--class_names_file", type=str, default="")

    parser.add_argument("--lambda_pseudo", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_diversity", type=float, default=1.0)
    parser.add_argument("--clip_conf_threshold", type=float, default=0.55)
    parser.add_argument("--soft_pseudo", action="store_true")
    parser.add_argument("--gate_hidden_dim", type=int, default=0)
    parser.add_argument("--gate_dropout", type=float, default=0.0)
    parser.add_argument("--gate_lr_factor", type=float, default=1.0)
    parser.add_argument("--adapter_type", type=str, default="scan", choices=["scan", "gate"])
    parser.add_argument("--freeze_backbone", action="store_true")

    # reliability-aware CLIP guidance
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--clip_warmup_epochs", type=int, default=1)
    parser.add_argument("--clip_min_weight", type=float, default=0.05)

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Yacs override options, e.g. MODEL.SOURCE_DIR outputs/vmamba_small TRAIN.EPOCHS 15",
    )
    args, _ = parser.parse_known_args()
    args.target_da = True
    args = img_param_init(args)
    config = get_config(args)
    return args, config


def clip_weight_schedule(epoch: int, total_epochs: int, warmup: int = 1, min_weight: float = 0.05) -> float:
    """Full CLIP weight for first `warmup` epochs, then linear decay to `min_weight`."""
    if epoch < warmup:
        return 1.0
    t = (epoch - warmup) / max(total_epochs - warmup, 1)
    return max(min_weight, 1.0 - (1.0 - min_weight) * t)


def safe_torch_load(path: str):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def load_cg_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    checkpoint_path = os.path.join(config.OUTPUT, "cfa_mamba_training_state.pt")
    training_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info(f"Resuming CFA-Mamba from epoch {training_state['epoch']}")
    model.load_state_dict(training_state["model"], strict=True)
    optimizer.load_state_dict(training_state["optimizer"])
    lr_scheduler.load_state_dict(training_state["lr_scheduler"])
    loss_scaler.load_state_dict(training_state["scaler"])
    config.defrost()
    config.TRAIN.START_EPOCH = training_state["epoch"] + 1
    config.freeze()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_source_model(config, args, model, logger):
    logger.info(f"Loading source feature extractor from {config.MODEL.SOURCE_F}")
    model.layers.load_state_dict(safe_torch_load(config.MODEL.SOURCE_F), strict=False)
    model.patch_embed.load_state_dict(safe_torch_load(config.MODEL.SOURCE_F), strict=False)
    if args.neck != "" and not (args.neck == "bn_relu" and config.MODEL.NECK_DEEP == 0):
        logger.info(f"Loading source fusion blocks from {config.MODEL.SOURCE_B}")
        model.fusion_blocks.load_state_dict(safe_torch_load(config.MODEL.SOURCE_B), strict=False)
    logger.info(f"Loading source classifier from {config.MODEL.SOURCE_C}")
    model.classifier.load_state_dict(safe_torch_load(config.MODEL.SOURCE_C), strict=True)


def build_cg_optimizer(config, args, model, logger):
    logger.info(f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}")
    for param in model.base_model.classifier.parameters():
        param.requires_grad_(False)

    if args.freeze_backbone:
        for name, param in model.base_model.named_parameters():
            if "fusion_blocks" not in name:
                param.requires_grad_(False)

    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    skip_keywords = model.no_weight_decay_keywords() if hasattr(model, "no_weight_decay_keywords") else set()

    gate_decay, gate_no_decay = [], []
    fusion_decay, fusion_no_decay = [], []
    base_decay, base_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip or check_keywords_in_name(name, skip_keywords)
        if name.startswith("semantic_gate"):
            (gate_no_decay if no_decay else gate_decay).append(param)
        elif "fusion_blocks" in name:
            (fusion_no_decay if no_decay else fusion_decay).append(param)
        else:
            (base_no_decay if no_decay else base_decay).append(param)

    parameters = [
        {"params": gate_decay,      "lr": config.TRAIN.BASE_LR * args.gate_lr_factor, "weight_decay": 1e-4},
        {"params": gate_no_decay,   "lr": config.TRAIN.BASE_LR * args.gate_lr_factor, "weight_decay": 0.0},
        {"params": fusion_decay,    "lr": config.TRAIN.BASE_LR * config.lr_factor2,   "weight_decay": 1e-4},
        {"params": fusion_no_decay, "lr": config.TRAIN.BASE_LR * config.lr_factor2,   "weight_decay": 0.0},
        {"params": base_decay,      "lr": config.TRAIN.BASE_LR * config.lr_factor1,   "weight_decay": 1e-4},
        {"params": base_no_decay,   "lr": config.TRAIN.BASE_LR * config.lr_factor1,   "weight_decay": 0.0},
    ]
    parameters = [g for g in parameters if len(g["params"]) > 0]

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    if opt_lower == "sgd":
        return optim.SGD(parameters, lr=config.TRAIN.BASE_LR,
                         momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                         nesterov=True, weight_decay=config.TRAIN.WEIGHT_DECAY)
    if opt_lower == "adamw":
        return optim.AdamW(parameters, lr=config.TRAIN.BASE_LR,
                           eps=config.TRAIN.OPTIMIZER.EPS,
                           betas=config.TRAIN.OPTIMIZER.BETAS,
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    raise NotImplementedError(f"Unsupported optimizer: {config.TRAIN.OPTIMIZER.NAME}")


def build_clip_cache_loader(config, args, teacher):
    target_domain = config.img_dataset[config.target_env]
    dataset = ImageDataset(
        config.dataset,
        config.DATA.DATA_PATH,
        target_domain,
        transform=teacher.preprocess,
        return_idx=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.clip_batch_size,
        shuffle=False,
        num_workers=args.clip_num_workers,
        pin_memory=True,
        persistent_workers=args.clip_num_workers > 0,
    )
    return dataset, loader


def steps_per_epoch(loader_len: int, accumulation_steps: int) -> int:
    return max(1, loader_len // max(1, accumulation_steps))


def validate(config, model, data_loader, clip_cache, flag=False):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    outputs_all = []
    labels_all = []
    with torch.no_grad():
        for data in data_loader:
            images = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            indices = data[2].long()
            clip_probs = clip_cache[indices].to(device, non_blocking=True)
            with autocast(enabled=config.AMP_ENABLE):
                logits = model(images, clip_probs=clip_probs)
            if flag:
                outputs_all.append(logits.detach().cpu())
                labels_all.append(labels.detach().cpu())
            else:
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()

    model.train()
    if not flag:
        return correct / max(total, 1)

    from sklearn.metrics import confusion_matrix

    all_output = torch.cat(outputs_all, dim=0)
    all_label = torch.cat(labels_all, dim=0)
    pred = all_output.argmax(dim=1)
    matrix = confusion_matrix(all_label.numpy(), pred.numpy())
    acc = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return acc.mean(), " ".join([str(np.round(i, 4)) for i in acc])


def save_checkpoint(config, args, model, optimizer, lr_scheduler, loss_scaler, epoch):
    os.makedirs(config.OUTPUT, exist_ok=True)
    torch.save(model.base_model.patch_embed.state_dict(), os.path.join(config.OUTPUT, "target_F_patch_embed.pt"))
    f_state = model.base_model.patch_embed.state_dict()
    f_state.update(model.base_model.layers.state_dict())
    torch.save(f_state, os.path.join(config.OUTPUT, "target_F.pt"))
    if args.neck != "" and not (args.neck == "bn_relu" and config.MODEL.NECK_DEEP == 0):
        torch.save(model.base_model.fusion_blocks.state_dict(), os.path.join(config.OUTPUT, "target_B.pt"))
    torch.save(model.base_model.classifier.state_dict(), os.path.join(config.OUTPUT, "target_C.pt"))
    torch.save(model.semantic_gate.state_dict(), os.path.join(config.OUTPUT, "target_semantic_gate.pt"))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": loss_scaler.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        os.path.join(config.OUTPUT, "cfa_mamba_training_state.pt"),
    )


def train_one_epoch(config, args, model, source_anchor, criterion, train_loader,
                    optimizer, epoch, lr_scheduler, loss_scaler, clip_cache, logger):
    model.train()
    optimizer.zero_grad()
    device = next(model.parameters()).device

    clip_w = clip_weight_schedule(epoch, config.TRAIN.EPOCHS,
                                  args.clip_warmup_epochs, args.clip_min_weight)

    num_steps = len(train_loader[0])
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    loss_meter   = AverageMeter()
    pseudo_meter = AverageMeter()
    cons_meter   = AverageMeter()
    ent_meter    = AverageMeter()
    anchor_meter = AverageMeter()
    selected_meter = AverageMeter()
    norm_meter   = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    train_iterator = zip(*train_loader)
    for batch_idx in range(num_steps):
        minibatches = [data for data in next(train_iterator)]
        samples    = torch.cat([data[0].float() for data in minibatches]).to(device, non_blocking=True)
        indices    = torch.cat([data[2].long()  for data in minibatches])
        clip_probs = clip_cache[indices].to(device, non_blocking=True)

        data_time.update(time.time() - end)

        # frozen source anchor inference (always eval, no grad)
        with torch.no_grad():
            with autocast(enabled=config.AMP_ENABLE):
                source_logits = source_anchor(samples)

        with autocast(enabled=config.AMP_ENABLE):
            logits   = model(samples, clip_probs=clip_probs)
            loss_out = criterion(logits, clip_probs,
                                 clip_weight=clip_w, source_logits=source_logits)
            loss = loss_out.total / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(batch_idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (batch_idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + batch_idx) // config.TRAIN.ACCUMULATION_STEPS)

        if device.type == "cuda":
            torch.cuda.synchronize()
        loss_scale_value = loss_scaler.state_dict()["scale"]
        batch_size = samples.size(0)
        loss_meter.update(loss_out.total.item(), batch_size)
        pseudo_meter.update(loss_out.pseudo.item(), batch_size)
        cons_meter.update(loss_out.consistency.item(), batch_size)
        ent_meter.update(loss_out.entropy.item(), batch_size)
        anchor_meter.update(loss_out.anchor.item(), batch_size)
        selected_meter.update(loss_out.selected_ratio, batch_size)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
            etas = batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{batch_idx}/{num_steps}] "
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f} "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f}) "
                f"data {data_time.val:.4f} ({data_time.avg:.4f}) "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"pseudo {pseudo_meter.avg:.4f} cons {cons_meter.avg:.4f} "
                f"ent {ent_meter.avg:.4f} anchor {anchor_meter.avg:.4f} "
                f"selected {selected_meter.avg:.3f} clip_w {clip_w:.3f} "
                f"grad_norm {norm_meter.avg:.4f} loss_scale {scaler_meter.avg:.1f} "
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(config, args):
    train_loaders, val_loaders = get_tgt_img_dataloader(config)
    logger.info(f"Creating base SfMamba model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA is not available. This script is intended for GPU training and may be very slow.")
    base_model = build_model(config).to(device)

    loss_scaler = NativeScalerWithGradNormCount()
    if args.resume and os.path.exists(os.path.join(config.OUTPUT, "cfa_mamba_training_state.pt")):
        model = ClipGuidedSfMamba(
            base_model,
            num_classes=config.MODEL.NUM_CLASSES,
            gate_hidden_dim=args.gate_hidden_dim or None,
            gate_dropout=args.gate_dropout,
            adapter_type=args.adapter_type,
        ).to(device)
        optimizer = build_cg_optimizer(config, args, model, logger)
        lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch(len(train_loaders[0]), config.TRAIN.ACCUMULATION_STEPS))
        load_cg_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger)
        # re-build source anchor from scratch when resuming
        source_anchor = build_model(config).to(device)
        load_source_model(config, args, source_anchor, logger)
    else:
        load_source_model(config, args, base_model, logger)
        # frozen source anchor: copied before any target adaptation begins
        source_anchor = copy.deepcopy(base_model).to(device)
        model = ClipGuidedSfMamba(
            base_model,
            num_classes=config.MODEL.NUM_CLASSES,
            gate_hidden_dim=args.gate_hidden_dim or None,
            gate_dropout=args.gate_dropout,
            adapter_type=args.adapter_type,
        ).to(device)
        optimizer = build_cg_optimizer(config, args, model, logger)
        lr_scheduler = build_scheduler(config, optimizer, steps_per_epoch(len(train_loaders[0]), config.TRAIN.ACCUMULATION_STEPS))

    for p in source_anchor.parameters():
        p.requires_grad_(False)
    source_anchor.eval()  # BN uses running stats, not batch stats
    logger.info("Frozen source anchor ready.")

    class_names = infer_class_names(config, args.class_names_file)
    logger.info(f"Using {len(class_names)} CLIP class names")
    teacher = ClipSemanticTeacher(
        class_names=class_names,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        backend=args.clip_backend,
        temperature=args.clip_temperature,
        device=device,
    )
    clip_dataset, clip_loader = build_clip_cache_loader(config, args, teacher)
    cache_path = args.clip_cache
    if not cache_path:
        cache_name = f"{config.dataset}_{config.domains[config.target_env]}_{args.clip_model.replace('/', '-')}.pt"
        cache_path = os.path.join(config.OUTPUT, "clip_cache", cache_name)
    clip_cache = teacher.build_cache(
        clip_loader,
        dataset_size=len(clip_dataset.x),
        cache_path=cache_path,
        force_recompute=args.force_clip_cache,
        amp=config.AMP_ENABLE,
        logger=logger,
    )
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    criterion = CGMambaLoss(
        lambda_pseudo=args.lambda_pseudo,
        lambda_consistency=args.lambda_consistency,
        lambda_entropy=args.lambda_entropy,
        lambda_diversity=args.lambda_diversity,
        lambda_anchor=args.lambda_anchor,
        confidence_threshold=args.clip_conf_threshold,
        use_clip_hard_labels=not args.soft_pseudo,
    )

    if config.THROUGHPUT_MODE:
        logger.info("Throughput mode is not implemented for CFA-Mamba because validation requires CLIP cache indices.")
        return

    logger.info("Start CFA-Mamba target adaptation")
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, args, model, source_anchor, criterion, train_loaders,
                        optimizer, epoch, lr_scheduler, loss_scaler, clip_cache, logger)
        target_acc = validate(config, model, val_loaders[0], clip_cache, flag=config.dataset == "VISDA-C")
        if config.dataset == "VISDA-C":
            acc_value, acc_list = target_acc
            logger.info(f"Accuracy of the network on TARGET: {acc_value * 100:.1f}%\nacc_list: {acc_list}")
        else:
            acc_value = target_acc
            logger.info(f"Accuracy of the network on TARGET: {acc_value * 100:.1f}%")

        if acc_value >= best_acc:
            best_acc = acc_value
            if args.issave or args.resume:
                save_checkpoint(config, args, model, optimizer, lr_scheduler, loss_scaler, epoch)
                logger.info(f"Saved best CFA-Mamba checkpoint at epoch {epoch}, acc={best_acc * 100:.1f}%")

        if epoch == config.TRAIN.EPOCHS - 1:
            endpoint_dir = os.path.join(config.OUTPUT, "endpoint_checkpoint")
            original_output = config.OUTPUT
            config.defrost()
            config.OUTPUT = endpoint_dir
            config.freeze()
            try:
                save_checkpoint(
                    config, args, model, optimizer,
                    lr_scheduler, loss_scaler, epoch
                )
                logger.info(
                    f"Saved fixed endpoint checkpoint at epoch {epoch}: "
                    f"{endpoint_dir}"
                )
            finally:
                config.defrost()
                config.OUTPUT = original_output
                config.freeze()

    total_time = time.time() - start_time
    logger.info(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    args, config = parse_option()
    seed = config.SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    config.defrost()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.TRAIN.BASE_LR   = config.TRAIN.BASE_LR   * config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR * config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR    = config.TRAIN.MIN_LR    * config.TRAIN.ACCUMULATION_STEPS
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"CFA-Mamba-{config.MODEL.NAME}")
    with open(os.path.join(config.OUTPUT, "cfa_mamba_config.yaml"), "w", encoding="utf-8") as f:
        f.write(config.dump())
    with open(os.path.join(config.OUTPUT, "cfa_mamba_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info(config.dump())
    logger.info(json.dumps(vars(args), ensure_ascii=False))
    main(config, args)
