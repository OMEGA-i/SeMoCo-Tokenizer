# SeMoCo-Tokenizer

[Model](https://huggingface.co/poisonousID/SeMoCo) | [Generator](https://github.com/OMEGA-i/SeMoCo-Generator)

SeMoCo introduces a semantic-first motion codec that organizes discrete
motion tokens by semantic roles, disentangling high-level motion states from
fine-grained kinematic details to improve autoregressive text-to-motion
generation.

## Quick Start

### 1. Environment

Requires Python 3.12, PyTorch (CUDA), and [git-lfs](https://git-lfs.com/).

The SOMA-X and TMR-SOMA-RP-v1 submodules keep their assets in LFS; without
git-lfs the submodule checkout fails.

```bash
git lfs install
git clone --recursive https://github.com/OMEGA-i/SeMoCo-Tokenizer.git
cd SeMoCo-Tokenizer
git submodule update --init --recursive   # if you cloned without --recursive
uv sync                                   # or: pip install -e .
```

Optional extras: `soma` for the SOMA-X forward-kinematics runtime, `teacher`
for loading the frozen retrieval teacher, `dev` for tests. Example:
`uv sync --extra soma`.

### 2. External assets

Third-party code and models are git submodules under `third_party/`; a
`--recursive` clone fetches them. Only the registration-gated SMPL-X model is
manual:

| Asset | Purpose | Where |
|---|---|---|
| SOMA-X | differentiable forward kinematics | `third_party/SOMA-X` submodule (LFS); its bundled `SOMA_neutral.npz` covers the dataset build and the FK template |
| kimodo | `SOMASkeleton77` skeleton adapter for the teacher | `third_party/kimodo` submodule; `SKIP_MOTION_CORRECTION_IN_SETUP=1 uv pip install -e third_party/kimodo` (skips its cmake-built C++ extension, unused here) |
| TMR teacher | semantic distillation target | `third_party/TMR-SOMA-RP-v1` submodule (LFS) |
| SMPL-X neutral | per-clip body shape for the L2/L4 eval gates | register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and place `SMPLX_NEUTRAL.npz` under `third_party/SOMA-X/assets/SMPLX/` |

Training and the L1/L3/L5 eval gates run without SMPL-X.

### 3. Pretrained weights

Download the pretrained tokenizer from
[Hugging Face](https://huggingface.co/poisonousID/SeMoCo):

```bash
hf download poisonousID/SeMoCo --include 'tokenizer/*' --local-dir checkpoints/
```

### 4. Run evaluation

The eval CLIs read per-recording `umr499.npz`, so materialize them from the
released parquet shards first:

```bash
python -m tools.materialize_umr_npz \
    --parquet-dir <release>/derived_umr_<hash> --split test \
    --out-root data/recordings --manifest-out data/test.txt

python -m tools.eval_recon_smpl22 \
    --checkpoint checkpoints/tokenizer/split_branch_sem.pt \
    --manifest data/test.txt --recordings-root data/recordings \
    --out recon_smpl22.json
```

Pass `--dataset HumanML3D` to `materialize_umr_npz` to keep only that portion
of the split.

See [Training](#training) and [Evaluation](#evaluation) for the full pipelines.

## Training

Training data is motion standardized to the SOMA skeleton in the 499-D UMR
representation; any source works once converted. Artifacts are addressed with
`local://…` URIs resolved against a data root:

```bash
export MOTIONVERSE_DATA_ROOT=/path/to/your/data
```

Materialize the training clips from the released parquet shards:

```bash
python -m tools.materialize_umr_npz \
    --parquet-dir <release>/derived_umr_<hash> --split train \
    --out-root data/recordings --manifest-out data/train.txt
```

To standardize your own motion instead, `tools.build_umr_dataset` converts
`soma77.npz` recordings into the same layout (needs the `soma` extra).

Build the fixed-window cache and normalization statistics:

```bash
python -m tools.build_training_cache --manifest data/train.txt \
    --recordings-root data/recordings --window 64 \
    --out local://cache/umr_fixed_w64_train.npy
python -m tools.compute_normalization --manifest data/train.txt \
    --recordings-root data/recordings \
    --out local://cache/norm_stats.json
```

Build the teacher embedding cache (needs the `teacher` extra, kimodo, and the
TMR checkpoint):

```bash
python -m tools.build_tmr_embedding_cache \
    --umr-cache local://cache/umr_fixed_w64_train.npy \
    --out local://cache/umr_fixed_w64_train.tmr_perwin_poly.npy
```

Train the codec:

```bash
torchrun --standalone --nproc_per_node=8 -m experiments.train_native \
    --config experiments/configs/split_fp_w64_s4.yaml \
    --output-dir runs/semoco_split_fp
```

Manifest files are plain text, one recording per line: either an absolute path,
or a recording id resolved under `--recordings-root` — as `soma77.npz` for the
dataset build, and as `umr499.npz` for evaluation.

The FK-geometry loss needs `data/assets/soma77_template.npz`, which ships with
the repo; regenerate it with `python -m tools.build_soma77_template` if the
skeleton changes.

## Evaluation

Reconstruction on the SOMA skeleton:

```bash
python -m tools.eval --codec-mode vq_vae \
    --checkpoint runs/semoco_split_fp/model/best.pt \
    --manifest data/test.txt --recordings-root data/recordings \
    --out runs/semoco_split_fp/recon.json
```

Semantic alignment and retrieval:

```bash
python -m tools.eval_semantic \
    --checkpoint runs/semoco_split_fp/model/best.pt \
    --umr-cache local://cache/umr_fixed_w64_val.npy \
    --tmr-cache local://cache/umr_fixed_w64_val.tmr_perwin_poly.npy \
    --out runs/semoco_split_fp/semantic.json
```

## Acknowledgements

This project uses [SOMA-X](https://github.com/huster-wgm/SOMA-X) for forward
kinematics, and NVIDIA's [TMR-SOMA-RP-v1](https://huggingface.co/nvidia/TMR-SOMA-RP-v1)
retrieval teacher via the [kimodo](https://github.com/nv-tlabs/kimodo) skeleton
adapter. These dependencies remain under their own licenses.

## License

Apache-2.0, see [`LICENSE`](LICENSE). External dependencies retain their own
licenses.
