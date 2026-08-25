"""Semantic-quality eval for TMR-distilled tokenizers, on held-out data.

Inputs: an ``experiments.train_native`` checkpoint, a held-out packed
``umr499_*.npy`` window cache, and its row-aligned frozen-TMR teacher cache
``tmr_emb_*.npy`` (2D ``[N, 256]`` only). Reports two layers:

Layer 1 -- teacher alignment (needs a SemanticHead): mean/median cos(h_sem, e_tmr).
Layer 2 -- teacher-neighborhood preservation: R@k that each window's teacher
top-1 cosine neighbor lands in the student representation's top-k; reported for
h_sem (if present) and for time-pooled z_q (model-agnostic baseline).

Usage::

    python -m tools.eval_semantic --checkpoint runs/<run>/model/best.pt \
        --umr-cache local://cache/umr499_val.npy --tmr-cache local://cache/tmr_emb_val.npy \
        --out runs/<run>/semantic_val.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from data.local_uri import default_data_root, resolve_local_uri

def _cfg_kwargs(cls: type, raw: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}

def _build_model(ckpt: dict, device: torch.device):
    from models.umr.structured_vq import (
        BackboneConfig,
        CodecConfig,
        GroupConfig,
        QuantizerConfig,
        SemanticHead,
        build_structured_vq,
    )

    cc = ckpt["codec_config"]
    groups = None
    if cc.get("groups"):
        groups = [
            GroupConfig(
                name=g["name"], start=int(g["start"]), stop=int(g["stop"]),
                backbone=BackboneConfig(**_cfg_kwargs(BackboneConfig, g["backbone"])),
                quantizer=QuantizerConfig(**_cfg_kwargs(QuantizerConfig, g["quantizer"])),
            )
            for g in cc["groups"]
        ]
    cfg = CodecConfig(
        backbone=BackboneConfig(**_cfg_kwargs(BackboneConfig, cc["backbone"])),
        quantizer=QuantizerConfig(**_cfg_kwargs(QuantizerConfig, cc["quantizer"])),
        input_dim=int(cc.get("input_dim", 499)),
        groups=groups,
    )
    model = build_structured_vq(cfg)

    net = ckpt["net"]
    # Attach a SemanticHead if the checkpoint carries one (TMR runs do).
    has_head = any(k.startswith("semantic_head.") for k in net)
    if has_head:
        w = net["semantic_head.proj.2.weight"]          # [dsem, hidden]
        dsem = int(w.shape[0])
        model.semantic_head = SemanticHead(latent_dim=cfg.backbone.latent_dim, dsem=dsem)

    load_result = model.load_state_dict(net, strict=False)
    leftover = [k for k in load_result.unexpected_keys if not k.startswith("semantic_head.")]
    if load_result.missing_keys or leftover:
        # semantic_head missing is fine for a baseline; codec must match exactly.
        missing = [k for k in load_result.missing_keys if not k.startswith("semantic_head.")]
        if missing or leftover:
            raise RuntimeError(f"param mismatch: missing={missing} unexpected={leftover}")
    model.eval().to(device)

    # Route the dedicated semantic branch / per-layer codes out at eval time.
    if hasattr(model, "quantizer") and hasattr(model.quantizer, "return_layer_codes"):
        model.quantizer.return_layer_codes = True

    kind = cfg.quantizer.kind
    return model, cfg, kind, has_head

@torch.no_grad()
def _encode_all(model, kind, has_head, feats_mm, mean_t, std_t, *, prefix_layers, batch_size, device, max_windows):
    n = feats_mm.shape[0] if max_windows is None else min(int(max_windows), feats_mm.shape[0])
    h_sems: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        feats = torch.from_numpy(np.ascontiguousarray(feats_mm[lo:hi])).float().to(device)  # [B, W, 499]
        x = feats.transpose(1, 2)                                # [B, 499, W]
        x = (x - mean_t) / std_t
        out = model(x)
        # pooled z_q (time mean) -- model-agnostic retrieval rep, works for baseline.
        if out.z_q is not None:
            pooled.append(out.z_q.mean(dim=2).cpu().numpy())     # [B, latent]
        if has_head:
            if kind == "split_rvq":
                sem_src = out.z_sem
            elif out.layer_z_q is not None:
                sem_src = out.layer_z_q[:, :, :prefix_layers, :].sum(dim=2)
            else:
                sem_src = None
            if sem_src is not None:
                h_sems.append(model.semantic_head(sem_src).cpu().numpy())  # [B, dsem]
    h = np.concatenate(h_sems, axis=0) if h_sems else None
    p = np.concatenate(pooled, axis=0) if pooled else None
    return h, p, n

def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def _teacher_nn_recall(rep: np.ndarray, teacher: np.ndarray, ks=(1, 5, 10), device="cpu") -> dict:
    """R@k that the teacher's top-1 neighbor is in the student rep's top-k."""
    R = torch.from_numpy(_l2(rep)).float().to(device)
    T = torch.from_numpy(_l2(teacher)).float().to(device)
    n = R.shape[0]
    St = T @ T.T
    St.fill_diagonal_(-2.0)
    teacher_nn = St.argmax(dim=1)
    Sr = R @ R.T
    Sr.fill_diagonal_(-2.0)
    tgt_score = Sr[torch.arange(n, device=Sr.device), teacher_nn]
    rank = (Sr > tgt_score.unsqueeze(1)).sum(dim=1)
    out = {}
    for k in ks:
        out[f"R@{k}"] = float((rank < k).float().mean())
    out["median_rank"] = float(rank.float().median())
    return out

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--umr-cache", required=True, help="packed umr499_*.npy (val/test)")
    p.add_argument("--tmr-cache", required=True, help="row-aligned tmr_emb_*.npy teacher")
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--prefix-layers", type=int, default=1, help="rvq sem source = sum(q0..q_{k-1}); ignored for split_rvq")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--retrieval-pool", type=int, default=4096, help="random subset size for R@k (0 = use all)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    umr_path = resolve_local_uri(args.umr_cache, data_root)
    tmr_path = resolve_local_uri(args.tmr_cache, data_root)
    out_path = resolve_local_uri(args.out, data_root)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, cfg, kind, has_head = _build_model(ckpt, device)
    mean_t = torch.from_numpy(np.asarray(ckpt["norm"]["mean"], dtype=np.float32)).view(1, -1, 1).to(device)
    std_t = torch.from_numpy(np.asarray(ckpt["norm"]["std"], dtype=np.float32)).view(1, -1, 1).to(device)

    feats_mm = np.load(umr_path, mmap_mode="r")
    teacher = np.asarray(np.load(tmr_path, mmap_mode="r"))
    if teacher.ndim != 2:
        raise ValueError(
            f"tmr-cache must be window-level 2D [N, dsem]; got shape {teacher.shape}. "
            f"Per-token (3D) teacher caches are not supported by eval_semantic."
        )

    h, pooled, n = _encode_all(
        model, kind, has_head, feats_mm, mean_t, std_t,
        prefix_layers=args.prefix_layers, batch_size=args.batch_size,
        device=device, max_windows=args.max_windows,
    )
    teacher = teacher[:n]
    keep = np.abs(teacher).sum(axis=1) > 0
    teacher = teacher[keep]
    if h is not None:
        h = h[keep]
    if pooled is not None:
        pooled = pooled[keep]
    n_valid = int(keep.sum())

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "umr_cache": str(umr_path),
        "tmr_cache": str(tmr_path),
        "quantizer_kind": kind,
        "has_semantic_head": bool(has_head),
        "n_windows": int(n),
        "n_valid": n_valid,
    }

    if h is not None:
        cos = (_l2(h) * _l2(teacher)).sum(axis=1)
        report["layer1_alignment"] = {
            "cos_mean": float(mean(cos.tolist())),
            "cos_median": float(median(cos.tolist())),
            "cos_p10": float(np.percentile(cos, 10)),
            "cos_p90": float(np.percentile(cos, 90)),
        }

    rng = np.random.default_rng(args.seed)
    pool = n_valid if args.retrieval_pool in (0, None) else min(args.retrieval_pool, n_valid)
    idx = rng.choice(n_valid, size=pool, replace=False)
    rdev = "cuda" if device.type == "cuda" else "cpu"
    report["layer2_retrieval"] = {"pool": int(pool)}
    if h is not None:
        report["layer2_retrieval"]["h_sem"] = _teacher_nn_recall(h[idx], teacher[idx], device=rdev)
    if pooled is not None:
        report["layer2_retrieval"]["pooled_zq"] = _teacher_nn_recall(pooled[idx], teacher[idx], device=rdev)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[eval-semantic] wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
