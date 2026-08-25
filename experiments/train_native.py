"""DistributedDataParallel trainer for the structured VQ codec.

Trains the 499D :class:`~models.umr.structured_vq.StructuredVQTokenizer` from a
single YAML (``codec`` / ``loss`` / ``data`` / ``optim`` / ``train`` /
``swanlab`` blocks) with a warmup/cosine LR schedule; checkpoints
(``{"net", "codec_config", "norm", ...}``) reload via ``tools.eval``. The EMA
codebook is a buffer, so DDP runs with ``broadcast_buffers=True``.

Usage::

    torchrun --nproc_per_node=4 -m experiments.train_native \
        --config experiments/configs/split_fp_w64_s4.yaml \
        --output-dir runs/semoco_split_fp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_cached_dataset import (
    CachedUMRFeatureDataset,
    cache_is_ready,
    collate_cached_umr,
)
from data.umr_varlen_dataset import (
    DistributedLengthBucketSampler,
    VarlenUMRDataset,
    collate_varlen_crop,
    varlen_cache_is_ready,
)
from data.umr_schema import DIM_FEATURES
from experiments.normalization_layer import FeatureNormalizationLayer
from losses.umr_loss import UMRLoss, UMRLossWeights, tmr_distill_loss
from models.umr.structured_vq import (
    BackboneConfig,
    CodecConfig,
    GroupConfig,
    QuantizerConfig,
    SemanticHead,
    build_structured_vq,
)

def _ddp_state(device: str) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        if distributed:
            device_t = torch.device("cuda", local_rank)
        else:
            device_t = torch.device(device)
            if device_t.index is None:
                device_t = torch.device("cuda", 0)
        torch.cuda.set_device(device_t)
    else:
        device_t = torch.device("cpu")
    rank = dist.get_rank() if distributed else 0
    return distributed, rank, world_size, device_t

def _is_rank0(rank: int) -> bool:
    return rank == 0

def _build_group(block: dict[str, Any]) -> GroupConfig:
    return GroupConfig(
        name=str(block["name"]),
        start=int(block["start"]),
        stop=int(block["stop"]),
        backbone=BackboneConfig(**(block.get("backbone") or {})),
        quantizer=QuantizerConfig(**(block.get("quantizer") or {})),
    )

def _build_codec_cfg(block: dict[str, Any]) -> CodecConfig:
    backbone = BackboneConfig(**(block.get("backbone") or {}))
    quantizer = QuantizerConfig(**(block.get("quantizer") or {}))
    groups_blk = block.get("groups")
    groups = [_build_group(g) for g in groups_blk] if groups_blk else None
    return CodecConfig(
        backbone=backbone,
        quantizer=quantizer,
        input_dim=int(block.get("input_dim", DIM_FEATURES)),
        groups=groups,
    )

def _build_loss(block: dict[str, Any]) -> UMRLoss:
    weights = UMRLossWeights(**(block.get("weights") or {}))
    return UMRLoss(
        weights=weights,
        contact_pos_weight=block.get("contact_pos_weight"),
        loss_mode=block.get("mode", "smooth_recon"),
        root_header_alpha=float(block.get("root_header_alpha", 0.0)),
        loss_vel=float(block.get("loss_vel", 0.0)),
        fk_weight=float(block.get("fk_weight", 1.0)),
        fk_vel_weight=float(block.get("fk_vel_weight", 0.0)),
        fk_acc_weight=float(block.get("fk_acc_weight", 0.0)),
        fk_footskate_weight=float(block.get("fk_footskate_weight", 0.0)),
        fk_template_path=block.get("fk_template_path"),
    )

def _init_swanlab(*, project: str, workspace: str | None, name: str, config: dict, enabled: bool) -> Any:
    class _Null:
        def log(self, *a: Any, **k: Any) -> None:
            return None

        def finish(self) -> None:
            return None

    if not enabled:
        return _Null()
    try:
        import swanlab  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[swanlab] disabled (import failed: {exc})")
        return _Null()
    try:
        kwargs: dict[str, Any] = {"project": project, "experiment_name": name, "config": config}
        if workspace:
            kwargs["workspace"] = workspace
        swanlab.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[swanlab] init failed ({exc}); no-op")
        return _Null()

    class _Run:
        def log(self, metrics: dict, *, step: int | None = None) -> None:
            try:
                swanlab.log(metrics, step=step) if step is not None else swanlab.log(metrics)
            except Exception as exc:  # noqa: BLE001
                print(f"[swanlab] log failed: {exc}")

        def finish(self) -> None:
            try:
                swanlab.finish()
            except Exception:  # noqa: BLE001
                pass

    return _Run()

@torch.no_grad()
def _eval_recon(
    model: torch.nn.Module,
    norm: FeatureNormalizationLayer,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    window: int | None = None,
) -> tuple[float, float, int]:
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    count = 0
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True).transpose(1, 2)
        if window is not None and feats.shape[-1] > window:
            feats = feats[..., :window]
        feats = feats.contiguous()
        feats_norm = norm(feats)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            rec = model(feats_norm).features_rec
        rec = rec.float()
        gt = feats_norm.float()
        bs = feats.shape[0]
        total_se += float(torch.mean((rec - gt) ** 2)) * bs
        total_ae += float(torch.mean(torch.abs(rec - gt))) * bs
        count += bs
    model.train()
    return total_se, total_ae, count

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing model/latest.pt and train from scratch "
        "(default: auto-resume model + optimizer + epoch from latest.pt)",
    )
    args = parser.parse_args(argv)

    payload = yaml.safe_load(args.config.read_text())
    codec_blk = payload.get("codec") or {}
    loss_blk = payload.get("loss") or {}
    data_blk = payload.get("data") or {}
    optim_blk = payload.get("optim") or {}
    train_blk = payload.get("train") or {}
    swan_blk = payload.get("swanlab") or {}

    distributed, rank, world_size, device_t = _ddp_state(args.device)
    seed = int(train_blk.get("seed", 3407))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    data_root = Path(data_blk.get("data_root")) if data_blk.get("data_root") else default_data_root()
    precision = str(train_blk.get("precision", "bf16")).lower()
    if precision in {"bf16", "bf16-mixed"}:
        amp_dtype: torch.dtype | None = torch.bfloat16
    elif precision in {"fp16", "16"}:
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    norm_stats_uri = train_blk.get("normalization_stats_json")
    if norm_stats_uri is None:
        raise ValueError("train.normalization_stats_json is required")
    norm_stats_path = resolve_local_uri(norm_stats_uri, data_root)
    norm = FeatureNormalizationLayer.from_json(norm_stats_path).to(device_t)

    window = int(data_blk.get("window", 64))
    batch_size = int(data_blk.get("batch_size", 256))
    num_workers = int(data_blk.get("num_workers", 4))

    train_cache = resolve_local_uri(data_blk["training_cache_path"], data_root)
    # Optional row-aligned TMR teacher-embedding cache; adds ``tmr_emb`` to each batch.
    tmr_emb_path = data_blk.get("tmr_emb_cache_path")

    # ``data.varlen``: group similar-length clips into fixed-shape batches
    # (crop-to-batch-min); otherwise each row is a single fixed ``window``.
    is_varlen = bool(data_blk.get("varlen", False))
    megabatch_mult = int(data_blk.get("megabatch_mult", 50))

    if is_varlen:
        if not varlen_cache_is_ready(train_cache, data_root):
            raise FileNotFoundError(f"varlen training cache not ready: {train_cache}")
        train_ds = VarlenUMRDataset(train_cache, data_root=data_root, tmr_emb_path=tmr_emb_path)
        token_stride = int(train_ds.token_stride)
        train_sampler = DistributedLengthBucketSampler(
            train_ds.lengths, batch_size,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=True, seed=seed, megabatch_mult=megabatch_mult,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=lambda b: collate_varlen_crop(b, token_stride=token_stride),
        )
        val_loader = None
        val_cache_uri = data_blk.get("val_cache_path")
        if val_cache_uri:
            val_cache = resolve_local_uri(val_cache_uri, data_root)
            if not varlen_cache_is_ready(val_cache, data_root):
                raise FileNotFoundError(f"varlen val cache not ready: {val_cache}")
            val_ds = VarlenUMRDataset(val_cache, data_root=data_root)
            val_sampler = DistributedLengthBucketSampler(
                val_ds.lengths, batch_size,
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                shuffle=False, seed=seed, megabatch_mult=megabatch_mult,
            )
            val_loader = DataLoader(
                val_ds,
                batch_sampler=val_sampler,
                num_workers=max(0, num_workers // 2),
                pin_memory=True,
                persistent_workers=num_workers > 0,
                collate_fn=lambda b: collate_varlen_crop(b, token_stride=token_stride),
            )
    else:
        if not cache_is_ready(train_cache):
            raise FileNotFoundError(f"training cache not ready: {train_cache}")
        train_ds = CachedUMRFeatureDataset(
            train_cache, data_root=data_root, in_ram=False, tmr_emb_path=tmr_emb_path
        )
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_cached_umr,
        )
        val_loader = None
        val_cache_uri = data_blk.get("val_cache_path")
        if val_cache_uri:
            val_cache = resolve_local_uri(val_cache_uri, data_root)
            if not cache_is_ready(val_cache):
                raise FileNotFoundError(f"val cache not ready: {val_cache}")
            val_ds = CachedUMRFeatureDataset(val_cache, data_root=data_root, in_ram=False)
            val_sampler = (
                DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                if distributed
                else None
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                drop_last=False,
                num_workers=max(0, num_workers // 2),
                pin_memory=True,
                persistent_workers=num_workers > 0,
                collate_fn=collate_cached_umr,
            )

    codec_cfg = _build_codec_cfg(codec_blk)
    model = build_structured_vq(codec_cfg).to(device_t)
    if window % int(model.temporal_stride) != 0:
        raise ValueError(
            f"window={window} not divisible by temporal_stride={model.temporal_stride}"
        )
    loss_fn = _build_loss(loss_blk).to(device_t)

    # Gradient clipping (default off) guards the lr-warmup ramp where the extra
    # TMR term can spike before the codebook settles.
    grad_clip = loss_blk.get("grad_clip_norm", 0.0)
    grad_clip = float(grad_clip) if grad_clip else 0.0

    # TMR semantic distillation: a SemanticHead projects the semantic code onto
    # the TMR embedding space. The head is attached BEFORE the DDP wrap so its
    # params join model.parameters() (optimizer + DDP + checkpoint).
    tmr_blk = loss_blk.get("tmr_distill") or {}
    tmr_enabled = bool(tmr_blk.get("enabled", False))
    tmr_dsem = int(tmr_blk.get("dsem", 256))
    tmr_prefix_layers = int(tmr_blk.get("prefix_layers", 2))      # q0+q1 default
    tmr_per_token = bool(tmr_blk.get("per_token", False))        # token-rate vs window-pool
    tmr_weight = float(tmr_blk.get("weight", 0.15))              # λ_tmr target
    tmr_warmup_start_weight = float(tmr_blk.get("warmup_start_weight", 0.05))
    tmr_warmup_steps = int(tmr_blk.get("warmup_steps", 0))
    tmr_split = codec_cfg.quantizer.kind == "split_rvq"
    if tmr_enabled:
        # split_rvq has a dedicated semantic VQ (depth 0) that is never dropped,
        # so the SemanticHead distills the codec's ``z_sem`` branch directly.
        if not tmr_split:
            # The head reads sum(q0..q_{k-1}); that prefix must be dropout-clean.
            # A layer is dropped iff q_idx > start_drop >= cutoff, so layers
            # 0..cutoff are never dropped: prefix_layers=k needs cutoff >= k-1.
            cutoff = int(getattr(codec_cfg.quantizer, "quantize_dropout_cutoff_index", 0))
            if (
                tmr_prefix_layers > 1
                and codec_cfg.quantizer.quantize_dropout
                and cutoff < tmr_prefix_layers - 1
            ):
                raise ValueError(
                    f"tmr_distill.prefix_layers={tmr_prefix_layers} with quantize_dropout=true "
                    f"requires codec.quantizer.quantize_dropout_cutoff_index >= {tmr_prefix_layers - 1} "
                    f"so q0..q_{{{tmr_prefix_layers - 1}}} are never dropped (clean prefix); "
                    f"got cutoff_index={cutoff}. Either raise the cutoff or set prefix_layers=1."
                )
        if tmr_emb_path is None:
            raise ValueError(
                "tmr_distill.enabled=true requires data.tmr_emb_cache_path "
                "(the row-aligned TMR teacher-embedding cache)."
            )
        # The SemanticHead must receive gradient every step: with DDP
        # find_unused_parameters=False a zero start weight trips the
        # "parameter was not used / marked ready twice" assertion.
        if tmr_warmup_start_weight <= 0.0:
            raise ValueError(
                "tmr_distill.warmup_start_weight must be > 0 so the SemanticHead always "
                "gets gradient (DDP find_unused_parameters=False). Use e.g. 0.05, not 0."
            )
        model.semantic_head = SemanticHead(
            latent_dim=codec_cfg.backbone.latent_dim, dsem=tmr_dsem, per_token=tmr_per_token
        ).to(device_t)

    raw_model = model
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device_t.index] if device_t.type == "cuda" else None,
            broadcast_buffers=True,            # rank-0 EMA codebook each forward
            find_unused_parameters=False,
        )
        raw_model = model.module
    model.train()

    lr = float(optim_blk.get("lr", 2e-4))
    weight_decay = float(optim_blk.get("weight_decay", 0.0))
    betas = tuple(optim_blk.get("betas", (0.9, 0.99)))
    warmup_steps = int(optim_blk.get("warmup_steps", 2000))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_blk.get("max_epochs", 250))
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * max_epochs)

    def _lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _tmr_scale(step: int) -> float:
        """Ramp λ_tmr linearly from warmup_start_weight to tmr_weight.

        Stays >0 so the SemanticHead always gets gradient (DDP
        find_unused_parameters=False).
        """
        if tmr_warmup_steps <= 0:
            return tmr_weight
        frac = min(1.0, max(0.0, step / float(tmr_warmup_steps)))
        return tmr_warmup_start_weight + frac * (tmr_weight - tmr_warmup_start_weight)

    output_dir = args.output_dir
    model_dir = output_dir / "model"
    if _is_rank0(rank):
        model_dir.mkdir(parents=True, exist_ok=True)

    mean_np = norm.mean.detach().cpu().view(-1).numpy().astype(np.float32)
    std_np = norm.std.detach().cpu().view(-1).numpy().astype(np.float32)
    codec_cfg_dict = asdict(codec_cfg)

    opt_summary = {
        "codec": codec_cfg_dict,
        "loss": loss_blk,
        "optim": {"lr": lr, "weight_decay": weight_decay, "betas": list(betas), "warmup_steps": warmup_steps},
        "data": {"window": window, "batch_size": batch_size, "num_workers": num_workers,
                 "training_cache_path": str(train_cache), "val_cache_path": str(val_cache_uri) if val_cache_uri else None},
        "train": {"max_epochs": max_epochs, "precision": precision, "seed": seed,
                  "normalization_stats_json": str(norm_stats_path),
                  "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                  "world_size": world_size},
    }
    if _is_rank0(rank):
        (output_dir / "opt.json").write_text(json.dumps(opt_summary, indent=2))
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"[native] params={n_params/1e6:.2f}M steps/epoch={steps_per_epoch} "
              f"total_steps={total_steps} bs={batch_size} world={world_size} prec={precision}")

    run_name = swan_blk.get("experiment_name") or output_dir.name
    swan = _init_swanlab(
        project=swan_blk.get("project", "SeMoCo"),
        workspace=swan_blk.get("workspace"),
        name=run_name,
        config=opt_summary,
        enabled=bool(swan_blk.get("enabled", False)) and _is_rank0(rank),
    )

    log_every = int(train_blk.get("log_every", 50))
    ckpt_every_epochs = int(train_blk.get("ckpt_every_epochs", max(1, max_epochs // 10)))

    def _save_ckpt(name: str, *, epoch: int, it: int, val_mse: float, val_mae: float) -> None:
        ckpt = {
            "net": raw_model.state_dict(),
            "codec_config": codec_cfg_dict,
            "norm": {"mean": mean_np, "std": std_np},
            "norm_stats_json": str(norm_stats_path),
            "loss": loss_blk,
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "iter": it,
            "val_recon_mse": val_mse,
            "val_recon_mae": val_mae,
            "config": opt_summary,
        }
        torch.save(ckpt, model_dir / name)

    best_val = float("inf")
    iter_count = 0
    start_epoch = 0
    # Auto-resume: all ranks load the SAME latest.pt so weights/optimizer stay
    # identical; schedules keep counting from the restored iter.
    resume_path = model_dir / "latest.pt"
    if not args.no_resume and resume_path.is_file():
        # weights_only=False: the checkpoint carries optimizer state and python
        # scalars, not just tensors (torch>=2.6 stricter default).
        ck = torch.load(resume_path, map_location=device_t, weights_only=False)
        raw_model.load_state_dict(ck["net"])
        if ck.get("opt") is not None:
            optimizer.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        iter_count = int(ck.get("iter", 0))
        best_pt = model_dir / "best.pt"
        if best_pt.is_file():
            try:
                bv = torch.load(best_pt, map_location="cpu", weights_only=False).get("val_recon_mse", float("inf"))
                best_val = float(bv) if bv == bv else float("inf")
            except Exception:
                pass
        if _is_rank0(rank):
            print(
                f"[native] RESUME {resume_path}: start_epoch={start_epoch} "
                f"iter={iter_count} best_val={best_val:.6f}"
            )
    elif args.no_resume and resume_path.is_file() and _is_rank0(rank):
        print(f"[native] --no-resume: ignoring existing {resume_path}, training from scratch")
    start = time.time()
    last_log_t = start

    try:
        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for batch in train_loader:
                iter_count += 1
                feats = batch["features"].to(device_t, non_blocking=True).transpose(1, 2)
                # Crop fixed-length cache windows to `window` (must stay
                # divisible by the model's temporal_stride).
                if feats.shape[-1] > window:
                    feats = feats[..., :window]
                feats = feats.contiguous()
                feats_norm = norm(feats)

                for pg in optimizer.param_groups:
                    pg["lr"] = lr * _lr_scale(iter_count)

                with torch.autocast(device_type=device_t.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(feats_norm)
                    loss_out = loss_fn(out.features_rec, feats_norm, vq_loss=out.vq_loss, norm=norm)
                    total_loss = loss_out.total
                    l_tmr_val = float("nan")
                    # split_rvq distills the dedicated semantic branch (out.z_sem);
                    # single-chain RVQ distills the early prefix sum(q0..q_{k-1}).
                    if tmr_split:
                        sem_src = out.z_sem
                    elif out.layer_z_q is not None:
                        sem_src = out.layer_z_q[:, :, :tmr_prefix_layers, :].sum(dim=2)
                    else:
                        sem_src = None
                    if tmr_enabled and sem_src is not None and "tmr_emb" in batch:
                        h_sem = raw_model.semantic_head(sem_src)
                        e_tmr = batch["tmr_emb"].to(device_t, non_blocking=True)
                        l_tmr = tmr_distill_loss(h_sem, e_tmr)
                        total_loss = total_loss + _tmr_scale(iter_count) * l_tmr
                        l_tmr_val = float(l_tmr.detach())
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if _is_rank0(rank) and iter_count % log_every == 0:
                    now = time.time()
                    s_per_step = (now - last_log_t) / log_every
                    last_log_t = now

                    def _f(x: Any) -> float:
                        return float(x.detach()) if torch.is_tensor(x) else float(x)

                    metrics = out.metrics
                    l_total = _f(total_loss)
                    l_rec = _f(loss_out.recon_smooth) if loss_out.recon_smooth is not None else float("nan")
                    l_root = _f(loss_out.root_header) if loss_out.root_header is not None else float("nan")
                    l_commit = _f(loss_out.vq)
                    ppl = _f(metrics.get("perplexity", float("nan")))
                    cur_lr = optimizer.param_groups[0]["lr"]
                    swan.log(
                        {
                            "train/loss": l_total,
                            "train/loss_rec": l_rec,
                            "train/loss_vel": _f(loss_out.joints76_rot6d),
                            "train/loss_root": l_root,
                            "train/loss_commit": l_commit,
                            "train/loss_tmr": l_tmr_val,
                            "train/lambda_tmr": _tmr_scale(iter_count) if tmr_enabled else float("nan"),
                            "train/perplexity": ppl,
                            "train/lr": cur_lr,
                            "train/s_per_step": s_per_step,
                            "train/iter": iter_count,
                        },
                        step=iter_count,
                    )
                    _tmr_str = "" if l_tmr_val != l_tmr_val else f"tmr={l_tmr_val:.4f} "
                    print(f"[ep {epoch+1}/{max_epochs}] it={iter_count} "
                          f"loss={l_total:.4f} rec={l_rec:.4f} commit={l_commit:.4f} {_tmr_str}ppl={ppl:.1f} "
                          f"lr={cur_lr:.2e} {s_per_step*1000:.0f}ms/step")

            val_mse = val_mae = float("nan")
            if val_loader is not None:
                se, ae, cnt = _eval_recon(raw_model, norm, val_loader, device_t, amp_dtype, window)
                if distributed:
                    t = torch.tensor([se, ae, float(cnt)], device=device_t, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    se, ae, cnt = float(t[0]), float(t[1]), int(t[2])
                val_mse = se / max(1, cnt)
                val_mae = ae / max(1, cnt)

            if _is_rank0(rank):
                elapsed = time.time() - start
                _save_ckpt("latest.pt", epoch=epoch, it=iter_count, val_mse=val_mse, val_mae=val_mae)
                if val_loader is not None and val_mse < best_val:
                    best_val = val_mse
                    _save_ckpt("best.pt", epoch=epoch, it=iter_count, val_mse=val_mse, val_mae=val_mae)
                if (epoch + 1) % ckpt_every_epochs == 0 or (epoch + 1) == max_epochs:
                    _save_ckpt(f"ep{epoch+1:04d}.pt", epoch=epoch, it=iter_count, val_mse=val_mse, val_mae=val_mae)
                swan.log(
                    {"val/recon_mse": val_mse, "val/recon_mae": val_mae,
                     "val/best_recon_mse": best_val, "val/epoch": epoch, "val/elapsed_sec": elapsed},
                    step=iter_count,
                )
                print(f"[ep {epoch+1}/{max_epochs}] val_mse={val_mse:.6f} val_mae={val_mae:.6f} "
                      f"best={best_val:.6f} elapsed={elapsed:.1f}s")
    finally:
        swan.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
