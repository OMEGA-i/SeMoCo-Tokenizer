"""Reconstruction eval on the SMPL-22 common joint set.

Reconstructs each ``umr499.npz`` clip with a self-contained native codec
checkpoint, FKs the reconstruction back to 77 joints, and scores it against GT
on the SMPL-22 common joint set via ``eval.recon_common`` (MPJPE / PA-MPJPE /
ACCEL / Jerk / APE, per-frame pelvis-aligned, in mm). Also reports full
77-joint MPJPE (``mpjpe77``), codebook health, and token rate. Requires the
``[soma]`` extras and the SMPL-X neutral model (see the README assets table).

Example::

    python -m tools.eval_recon_smpl22 --checkpoint runs/<run>/model/best.pt \
        --manifest local://manifests/test.txt --recordings-root local://raw/recordings \
        --device cuda --out runs/<run>/recon_smpl22.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_dataset import UMRDataset
from data.umr_schema import UMR499

log = logging.getLogger("eval_recon_smpl22")


def _codebook_report(cb_hist: dict[str, np.ndarray], cb_size: int) -> dict[str, Any]:
    """Per-stream + overall codebook usage / perplexity / dead-code ratio from
    summed bincounts ``cb_hist[stream]`` (length ``cb_size``)."""
    def _one(h: np.ndarray) -> dict[str, float]:
        total = int(h.sum())
        used = int((h > 0).sum())
        usage = used / max(cb_size, 1)
        if total > 0:
            p = h[h > 0].astype(np.float64) / total
            perplexity = float(np.exp(-(p * np.log(p)).sum()))
        else:
            perplexity = 0.0
        return {"usage": usage, "perplexity": perplexity,
                "dead_code_ratio": 1.0 - usage, "used_codes": used, "n_codes": total}

    out: dict[str, Any] = {"codebook_size": cb_size,
                           "per_stream": {s: _one(h) for s, h in cb_hist.items()}}
    total_h = np.zeros(cb_size, dtype=np.int64)
    for h in cb_hist.values():
        total_h += h
    out["overall"] = _one(total_h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                        help="Self-contained native checkpoint from experiments.train_native "
                             "(codec_config + net + norm). Required for vae/vq_vae.")
    ap.add_argument("--manifest", action="append", required=True,
                    help="One or more manifest files (path or local:// URI).")
    ap.add_argument("--recordings-root", default=None, help="Accepts local:// URIs.")
    ap.add_argument("--data-root", default=None,
                    help="Resolve local:// URIs against this root (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="max clips")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--out", required=True, help="Output JSON path (accepts local:// URIs).")
    ap.add_argument("--codec-mode", default="vq_vae", choices=["vq_vae", "vae", "identity"])
    ap.add_argument("--gt-mode", default="refk", choices=["stored", "refk"],
                    help="GT joints source: 'refk' (default) = materialize+FK the GT features "
                         "through the SAME path as the reconstruction, so a perfect reconstruction "
                         "scores exactly 0; 'stored' = the joints77_pos stored in each umr499.npz, "
                         "which also charges the materialize+FK error of the pipeline itself.")
    args = ap.parse_args()
    if args.codec_mode != "identity" and args.checkpoint is None:
        ap.error(f"--codec-mode {args.codec_mode} requires --checkpoint")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from data.soma77_fk import soma77_joints_world_xyz_from_matrices
    from data.umr_to_soma77 import materialize_features_matrices
    from eval.recon_common import recon_metrics, soma77_to_smpl22
    from tools.eval import _load_codec_adapter

    codec = _load_codec_adapter(args)

    collect_cb = bool(getattr(codec, "is_vq_vae", False))
    cb_size = int(getattr(codec, "codebook_size", 0)) or 0
    cb_hist: dict[str, np.ndarray] = {}   # stream -> bincount over cb_size (summed across clips)
    rate_total_codes = 0                  # every emitted code (all streams, layers, timesteps)
    rate_total_seconds = 0.0              # summed clip duration (input frames / fps)

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    ds = UMRDataset(
        args.manifest,
        recordings_root=args.recordings_root,
        data_root=data_root,
        window=max(1, int(args.min_frames)),
    )
    paths = ds.npz_paths[: args.limit] if args.limit is not None else ds.npz_paths
    log.info("evaluating %d clips", len(paths))

    per_metric: dict[str, list[float]] = defaultdict(list)
    clip_ids: list[str] = []
    n_ok = 0
    n_skip = 0

    for path in paths:
        try:
            umr = UMR499.from_npz(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", path, exc)
            n_skip += 1
            continue
        if umr.num_records < args.min_frames:
            n_skip += 1
            continue

        features_rec, _ = codec.reconstruct(umr.features)
        T_rec = features_rec.shape[0]

        if collect_cb and cb_size > 0:
            try:
                for stream, idx in codec.token_indices(umr.features).items():
                    flat = np.asarray(idx).reshape(-1)
                    flat = flat[(flat >= 0) & (flat < cb_size)]
                    h = cb_hist.get(stream)
                    if h is None:
                        h = np.zeros(cb_size, dtype=np.int64)
                        cb_hist[stream] = h
                    h += np.bincount(flat, minlength=cb_size)
                    rate_total_codes += int(np.asarray(idx).size)
                rate_total_seconds += float(umr.num_records) / max(float(umr.fps), 1e-6)
            except Exception as exc:  # noqa: BLE001
                log.warning("codebook stats skip %s: %s", path, exc)

        rec_mats = materialize_features_matrices(features_rec, umr.canonical_anchor)
        joints77_rec = soma77_joints_world_xyz_from_matrices(
            rec_mats.rotmat77, rec_mats.transl, umr.identity_coeffs, device=args.device
        )
        joints77_rec = np.asarray(joints77_rec, dtype=np.float32)

        if args.gt_mode == "refk":
            # symmetric caliper: GT joints from the SAME materialize+FK path as the
            # reconstruction, so a perfect tokenizer scores exactly 0 (no floor).
            gt_mats = materialize_features_matrices(umr.features[:T_rec], umr.canonical_anchor)
            joints77_gt = np.asarray(
                soma77_joints_world_xyz_from_matrices(
                    gt_mats.rotmat77, gt_mats.transl, umr.identity_coeffs, device=args.device
                ),
                dtype=np.float32,
            )
        else:
            joints77_gt = umr.joints77_pos[: T_rec + 1]

        smpl22_gt = soma77_to_smpl22(joints77_gt)
        smpl22_rec = soma77_to_smpl22(joints77_rec)

        m = recon_metrics(smpl22_rec, smpl22_gt)
        for k, v in m.to_dict().items():
            if k == "n_frames":
                continue
            if np.isfinite(v):
                per_metric[k].append(v)
        # full 77-joint pelvis-aligned MPJPE (our-representation headline)
        T77 = min(joints77_rec.shape[0], joints77_gt.shape[0])
        jr = joints77_rec[:T77] - joints77_rec[:T77, :1]
        jg = joints77_gt[:T77] - joints77_gt[:T77, :1]
        mpjpe77 = float(np.linalg.norm(jr - jg, axis=-1).mean() * 1000.0)
        if np.isfinite(mpjpe77):
            per_metric["mpjpe77"].append(mpjpe77)
        clip_ids.append(Path(path).parent.name)
        n_ok += 1
        if n_ok % 200 == 0:
            log.info("processed %d clips (mpjpe running mean %.3f)", n_ok, float(np.mean(per_metric["mpjpe"])))

    report = {
        "checkpoint": args.checkpoint,
        "n_clips": n_ok,
        "n_skip": n_skip,
        "metrics": {
            k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
            for k, v in per_metric.items()
        },
    }

    if collect_cb and cb_size > 0 and cb_hist:
        report["codebook"] = _codebook_report(cb_hist, cb_size)
        bits_per_code = float(np.log2(cb_size))
        secs = rate_total_seconds if rate_total_seconds > 0 else float("nan")
        report["token_rate"] = {
            "codebook_size": cb_size,
            "bits_per_code": bits_per_code,
            "total_codes": int(rate_total_codes),
            "total_seconds": float(rate_total_seconds),
            "tokens_per_sec": float(rate_total_codes / secs),
            "bits_per_sec": float(rate_total_codes * bits_per_code / secs),
        }

    out_path = resolve_local_uri(args.out, data_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    raw: dict[str, np.ndarray] = {
        "clip_ids": np.asarray(clip_ids),
        **{k: np.asarray(v, dtype=np.float32) for k, v in per_metric.items()},
    }
    if collect_cb and cb_size > 0 and cb_hist:
        for stream, h in cb_hist.items():
            raw[f"cb_hist_{stream}"] = h.astype(np.int64)
        raw["cb_size"] = np.asarray([cb_size], dtype=np.int64)
        raw["rate_total_codes"] = np.asarray([rate_total_codes], dtype=np.int64)
        raw["rate_total_seconds"] = np.asarray([rate_total_seconds], dtype=np.float64)
    np.savez(str(out_path.with_suffix(".raw.npz")), **raw)
    log.info("wrote %s (+ .raw.npz)", out_path)
    print(f"\n=== recon SMPL-22 (n={n_ok}) ===")
    for k in ["mpjpe", "mpjpe77", "pampjpe", "accel_err", "jerk_err", "ape_root", "ape_pose", "ape_joints"]:
        if k in report["metrics"]:
            s = report["metrics"][k]
            print(f"  {k:12s} mean={s['mean']:8.3f}  median={s['median']:8.3f}")
    if "codebook" in report:
        cb = report["codebook"]["overall"]
        tr = report["token_rate"]
        print(f"  codebook     usage={cb['usage']*100:6.2f}%  ppl={cb['perplexity']:8.1f}  "
              f"dead={cb['dead_code_ratio']*100:5.2f}%  (size={report['codebook']['codebook_size']})")
        print(f"  token_rate   {tr['tokens_per_sec']:.1f} tok/s  {tr['bits_per_sec']:.1f} bit/s  "
              f"({tr['bits_per_code']:.2f} bit/code)")


if __name__ == "__main__":
    sys.exit(main())
