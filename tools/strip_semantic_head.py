"""Canonicalize a tokenizer checkpoint: move the training-only SemanticHead out
of ``ckpt["net"]`` so ``net`` is a PURE codec state_dict.

``train_native`` checkpoints can leak ``semantic_head.*`` into ``net``, breaking
consumers that strict-load the bare codec. Rewrites the checkpoint with ``net``
= codec-only and ``semantic_head`` under its own key; idempotent.

Usage:
    python -m tools.strip_semantic_head runs/<run>/model/best.pt   # in place
    python -m tools.strip_semantic_head <dir> --glob 'best.pt'     # walk a dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

_PREFIX = "semantic_head."

def strip_one(path: Path, *, out: Path | None, dry_run: bool) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ck.get("net")
    if not isinstance(net, dict):
        return f"SKIP {path} (no dict 'net')"
    head_keys = [k for k in net if k.startswith(_PREFIX)]
    if not head_keys:
        return f"OK   {path} (already codec-only, {len(net)} tensors)"
    codec_sd = {k: v for k, v in net.items() if not k.startswith(_PREFIX)}
    head_sd = {k[len(_PREFIX):]: v for k, v in net.items() if k.startswith(_PREFIX)}
    # Preserve any pre-existing separate head; the copy inside net wins if both exist.
    if ck.get("semantic_head"):
        head_sd = {**ck["semantic_head"], **head_sd}
    ck["net"] = codec_sd
    ck["semantic_head"] = head_sd
    dst = out or path
    if dry_run:
        return f"DRY  {path} -> would strip {len(head_keys)} head tensors -> net={len(codec_sd)} (dst={dst})"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(dst)
    return f"STRIP {path} -> {dst}  (net={len(codec_sd)} codec, semantic_head={len(head_sd)})"

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="checkpoint file(s) or directory(ies)")
    ap.add_argument("--glob", default="best.pt", help="when a path is a dir, glob to match (default best.pt)")
    ap.add_argument("--suffix", default=None, help="write to <stem><suffix>.pt instead of in place")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob(args.glob)))
        elif p.is_file():
            files.append(p)
        else:
            print(f"SKIP {p} (not found)")
    if not files:
        print("no checkpoints matched")
        return 1
    for f in files:
        out = None
        if args.suffix:
            out = f.with_name(f.stem + args.suffix + f.suffix)
        print(strip_one(f, out=out, dry_run=args.dry_run))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
