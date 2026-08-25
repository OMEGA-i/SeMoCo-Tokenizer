"""Materialize per-recording ``umr499.npz`` from the derived UMR parquet shards.

The eval CLIs (``tools.eval``, ``tools.eval_recon_smpl22``) read a manifest of
recordings resolved as ``<recordings-root>/<rec_id>/umr499.npz``, while the
release ships the same clips as parquet shards. This writes the npz view back
out, optionally restricted to one provenance dataset (e.g. the HumanML3D
portion of the test split).

Example::

    python -m tools.materialize_umr_npz \
        --parquet-dir <release>/derived_umr_<hash> --split test \
        --dataset HumanML3D --out-root data/recordings \
        --manifest-out data/test.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data.umr_schema import (
    DIM_FEATURES,
    DIM_ROOT_ROT6D,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    UMR499,
)

COLUMNS = [
    "rec_id",
    "num_records",
    "fps",
    "features",
    "init_root_pos",
    "init_root_rot6d",
    "init_joints76_rot6d",
    "identity_coeffs",
    "joint_orient",
    "joints77_pos",
    "provenance_json",
    "text",
]


def _row_to_umr(row: dict) -> UMR499:
    n = int(row["num_records"])
    features = np.asarray(row["features"], dtype=np.float32).reshape(n, DIM_FEATURES)
    joints77 = np.asarray(row["joints77_pos"], dtype=np.float32).reshape(
        n + 1, UMR_NUM_JOINTS, 3
    )
    identity = np.asarray(row["identity_coeffs"], dtype=np.float32).reshape(1, -1)
    anchor = CanonicalAnchor(
        init_root_pos=np.asarray(row["init_root_pos"], dtype=np.float32).reshape(3),
        init_root_rot6d=np.asarray(row["init_root_rot6d"], dtype=np.float32).reshape(
            DIM_ROOT_ROT6D
        ),
        init_joints76_rot6d=np.asarray(
            row["init_joints76_rot6d"], dtype=np.float32
        ).reshape(UMR_NUM_JOINTS76, 6),
    )
    return UMR499(
        canonical_anchor=anchor,
        features=features,
        joints77_pos=joints77,
        identity_coeffs=identity,
        joint_orient=np.asarray(row["joint_orient"], dtype=np.float32).reshape(78, 3, 3),
        fps=float(row["fps"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True, help="<release>/derived_umr_<hash>")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="keep only rows whose provenance dataset matches (repeatable)",
    )
    ap.add_argument(
        "--rec-ids",
        default=None,
        help="file of rec_ids to keep (one per line), applied on top of --dataset",
    )
    ap.add_argument("--out-root", required=True, help="recordings root to write into")
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--captions-out", default=None, help="optional rec_id -> caption json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-records", type=int, default=0)
    ap.add_argument("--uncompressed", action="store_true", help="faster, larger npz")
    args = ap.parse_args()

    shards = sorted(Path(args.parquet_dir, args.split).glob("part-*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.parquet_dir}/{args.split}")
    keep = set(args.dataset) if args.dataset else None
    keep_recs: set[str] | None = None
    if args.rec_ids:
        keep_recs = {
            ln.strip()
            for ln in Path(args.rec_ids).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rec_ids: list[str] = []
    captions: dict[str, str] = {}
    n_written = 0
    n_skipped = 0

    for shard in shards:
        table = pq.read_table(shard, columns=COLUMNS)
        prov = table.column("provenance_json").to_pylist()
        datasets = [
            (json.loads(p).get("dataset") if p else None) for p in prov
        ]
        shard_recs = table.column("rec_id").to_pylist()
        wanted = [
            i
            for i, ds in enumerate(datasets)
            if (keep is None or ds in keep)
            and (keep_recs is None or shard_recs[i] in keep_recs)
        ]
        if not wanted:
            continue
        for i in wanted:
            if args.limit is not None and n_written >= args.limit:
                break
            row = table.slice(i, 1).to_pylist()[0]
            if int(row["num_records"]) < args.min_records:
                n_skipped += 1
                continue
            try:
                umr = _row_to_umr(row)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {row['rec_id']}: {exc}", file=sys.stderr)
                n_skipped += 1
                continue
            umr.to_npz(
                out_root / row["rec_id"] / "umr499.npz",
                compressed=not args.uncompressed,
            )
            rec_ids.append(row["rec_id"])
            if row.get("text"):
                captions[row["rec_id"]] = row["text"]
            n_written += 1
        print(f"{shard.name}: {n_written} written", flush=True)
        if args.limit is not None and n_written >= args.limit:
            break

    if args.manifest_out:
        mp = Path(args.manifest_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text("\n".join(rec_ids) + "\n")
        print(f"manifest: {mp} ({len(rec_ids)} entries)")
    if args.captions_out:
        cp = Path(args.captions_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(captions, indent=1))
        print(f"captions: {cp} ({len(captions)} entries)")
    print(f"wrote {n_written} clips, skipped {n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
