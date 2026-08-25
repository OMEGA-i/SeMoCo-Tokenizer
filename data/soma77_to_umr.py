"""Convert :class:`~data.soma77_schema.Soma77Canonical` → :class:`UMR499`.

Implements the ``delta_root_multiscale_sparsevel`` forward pipeline: resample
to target fps (positions polyphase, poses SLERP), rigidly canonicalize the FK
positions (floor-Y, frame-0 root xz, frame-0 facing → +Z), lift the root
axis-angle into the canonical frame, then compute the 499D records and the
frame-0 anchor. Foot contacts are derived from canonical-position velocity,
not resampled from source. Pure numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation, Slerp

from data.soma77_schema import (
    NUM_JOINTS as SOMA_NUM_JOINTS,
    Soma77Canonical,
)
from data.umr_schema import (
    DIM_FEATURES,
    DIM_ROOT_TRAJ,
    FEATURE_VARIANT,
    SPARSE_VEL_JOINTS,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    FeatureFields,
    UMR499,
    pack_features,
    root_multiscale_windows_for_fps,
)

# ---------------------------------------------------------------------------
# Face joint indices (SMPL-X body order, shared by SOMA77 chain)
# ---------------------------------------------------------------------------

# (right_hip, left_hip, right_shoulder, left_shoulder) used to estimate the
# frame-0 facing direction; SOMA77 FK indices, not SMPL-X body ordering.
FACE_JOINT_INDICES: tuple[int, int, int, int] = (72, 67, 40, 12)

# Foot contacts derived from canonical-position velocity so both pipelines
# emit identical labels; column order matches umr_schema.FOOT_CONTACT_SOMA77_INDICES.
FOOT_CONTACT_JOINTS: tuple[int, int, int, int] = (69, 70, 74, 75)
FEET_VEL_THRESHOLD: float = 0.002

# ---------------------------------------------------------------------------
# Small rotation helpers (numpy)
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle ``[..., 3]`` → rotation matrix ``[..., 3, 3]``."""
    if aa.shape[-1] != 3:
        raise ValueError(f"axis_angle last dim must be 3; got {aa.shape}")
    lead = aa.shape[:-1]
    flat = aa.reshape(-1, 3).astype(np.float32, copy=False)
    R = Rotation.from_rotvec(flat).as_matrix().astype(np.float32)
    return R.reshape(*lead, 3, 3)

def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Zhou 6D = first two columns of R, flattened to ``[..., 6]``."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"rotmat last two dims must be (3, 3); got {R.shape}")
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1).astype(np.float32)

def slerp_axis_angle(
    axis_angle: np.ndarray, src_fps: float, target_fps: float, target_frames: int
) -> np.ndarray:
    """Per-joint SLERP from ``src_fps`` to ``target_fps``; returns ``[target_frames, J, 3]``.

    Times are aligned at the start; the last target time is clamped at
    ``(T_src - 1) / src_fps`` to avoid extrapolation.
    """
    T_src, J, _ = axis_angle.shape
    if target_frames <= 0 or T_src <= 0:
        return np.zeros((max(target_frames, 0), J, 3), dtype=np.float32)
    if abs(src_fps - target_fps) < 1e-6 and target_frames == T_src:
        return axis_angle.astype(np.float32, copy=False)

    times_src = np.arange(T_src, dtype=np.float64) / float(src_fps)
    times_tgt = np.minimum(
        np.arange(target_frames, dtype=np.float64) / float(target_fps),
        times_src[-1],
    )
    out = np.empty((target_frames, J, 3), dtype=np.float32)
    for j in range(J):
        slerp = Slerp(times_src, Rotation.from_rotvec(axis_angle[:, j, :]))
        out[:, j] = slerp(times_tgt).as_rotvec().astype(np.float32)
    return out

def resample_positions(positions: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Polyphase resample of ``[T, J, 3]`` positions.

    ``padtype="line"`` extends channels with their boundary slope; the default
    zero-padding would pull boundary frames toward the world origin (visible
    "snap back" at clip ends).
    """
    if src_fps <= 0:
        raise ValueError(f"src_fps must be > 0, got {src_fps}")
    if abs(src_fps - target_fps) < 1e-6:
        return positions.astype(np.float32, copy=False)
    src_i = max(1, int(round(src_fps)))
    tgt_i = max(1, int(round(target_fps)))
    T, J, C = positions.shape
    flat = positions.reshape(T, J * C)
    rs = resample_poly(flat, tgt_i, src_i, axis=0, padtype="line")
    return rs.reshape(rs.shape[0], J, C).astype(np.float32)

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize_positions(
    joints77: np.ndarray,
    *,
    face_indices: tuple[int, int, int, int] = FACE_JOINT_INDICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid canonicalization of ``joints77 [T, 77, 3]``.

    Floor-Y align (``min(y) == 0``), zero the frame-0 root xz, then rotate
    around y so the frame-0 facing vector points to +Z. Returns
    ``(joints77_can, canon_quat_wxyz)`` where ``canon_quat_wxyz`` is the y-axis
    rotation, reused by :func:`apply_canon_to_root_axis_angle`.
    """
    if joints77.ndim != 3 or joints77.shape[1] != UMR_NUM_JOINTS or joints77.shape[2] != 3:
        raise ValueError(f"joints77 shape {joints77.shape} must be (T, {UMR_NUM_JOINTS}, 3)")

    pos = joints77.astype(np.float32, copy=True)

    # 1. floor-Y align.
    floor = float(pos[..., 1].min())
    pos[..., 1] -= floor

    # 2. zero frame-0 root xz.
    root_xz_init = pos[0, 0] * np.array([1.0, 0.0, 1.0], dtype=np.float32)
    pos = pos - root_xz_init

    # 3. align frame-0 facing to +Z.
    r_hip, l_hip, sdr_r, sdr_l = face_indices
    across = (pos[0, r_hip] - pos[0, l_hip]) + (pos[0, sdr_r] - pos[0, sdr_l])
    across = across / (np.linalg.norm(across) + 1e-12)
    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), across, axis=-1)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    canon_rot = Rotation.align_vectors(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),  # target = +Z
        forward[None, :],                                # source = body forward
    )[0]
    canon_quat_xyzw = canon_rot.as_quat().astype(np.float32)
    # Pack as wxyz so callers can reuse it like ``np.array([w, x, y, z])``.
    canon_quat_wxyz = np.array(
        [canon_quat_xyzw[3], canon_quat_xyzw[0], canon_quat_xyzw[1], canon_quat_xyzw[2]],
        dtype=np.float32,
    )
    canon_mat = canon_rot.as_matrix().astype(np.float32)
    pos = np.einsum("ij,tkj->tki", canon_mat, pos).astype(np.float32, copy=False)
    return pos, canon_quat_wxyz

def apply_canon_to_root_axis_angle(
    root_axis_angle: np.ndarray,
    canon_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Lift ``poses[:, 0]`` (axis-angle, world frame) into the canonical frame; returns ``[T, 3]``."""
    canon_xyzw = np.array(
        [canon_quat_wxyz[1], canon_quat_wxyz[2], canon_quat_wxyz[3], canon_quat_wxyz[0]],
        dtype=np.float32,
    )
    canon = Rotation.from_quat(canon_xyzw)
    out = (canon * Rotation.from_rotvec(root_axis_angle.astype(np.float32, copy=False))).as_rotvec()
    return out.astype(np.float32, copy=False)

# ---------------------------------------------------------------------------
# Root-yaw local frame
# ---------------------------------------------------------------------------

def _root_yaw_inv_mats(root_mats: np.ndarray) -> np.ndarray:
    """Yaw-only matrices that rotate canonical vectors into the per-frame root-yaw frame."""
    forward = root_mats[:, :, 2].astype(np.float32)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    mats = np.zeros((root_mats.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = c
    mats[:, 0, 2] = -s
    mats[:, 1, 1] = 1.0
    mats[:, 2, 0] = s
    mats[:, 2, 2] = c
    return mats

def _build_root_traj(
    canon_positions: np.ndarray,
    root_mats: np.ndarray,
    *,
    target_fps: float,
) -> np.ndarray:
    """Compute the 9D ``root_traj`` channel per record.

    Channels::
        [0:2]  local_dxz       — world xz delta rotated into the t+1 root-yaw frame
        [2:4]  world_dxz       — canonical world xz delta
        [4]    root_y          — absolute root y at frame t+1
        [5:7]  multiscale_0p2s — (xz[t+1] - xz[max(0, t+1-round(0.2*fps))]) / window
        [7:9]  multiscale_0p4s — same with round(0.4*fps)
    """
    T = canon_positions.shape[0]
    yaw_inv = _root_yaw_inv_mats(root_mats[1:])
    world_delta = np.zeros((T - 1, 3), dtype=np.float32)
    world_delta[:, 0] = canon_positions[1:, 0, 0] - canon_positions[:-1, 0, 0]
    world_delta[:, 2] = canon_positions[1:, 0, 2] - canon_positions[:-1, 0, 2]
    local_delta = np.einsum("tij,tj->ti", yaw_inv, world_delta, optimize=True)

    root_traj = np.empty((T - 1, DIM_ROOT_TRAJ), dtype=np.float32)
    root_traj[:, 0] = local_delta[:, 0]
    root_traj[:, 1] = local_delta[:, 2]
    root_traj[:, 2] = world_delta[:, 0]
    root_traj[:, 3] = world_delta[:, 2]
    root_traj[:, 4] = canon_positions[1:, 0, 1]

    root_xz = canon_positions[:, 0, [0, 2]].astype(np.float32)
    w4, w8 = root_multiscale_windows_for_fps(target_fps)
    for out_i, t in enumerate(range(1, T)):
        prev4 = max(0, t - w4)
        prev8 = max(0, t - w8)
        root_traj[out_i, 5:7] = (root_xz[t] - root_xz[prev4]) / float(t - prev4)
        root_traj[out_i, 7:9] = (root_xz[t] - root_xz[prev8]) / float(t - prev8)
    return root_traj

def _build_sparse_vel(
    canon_positions: np.ndarray,
    root_mats: np.ndarray,
) -> np.ndarray:
    """Compute the ``[T-1, 8, 3]`` sparse-key-joint velocity in the t+1 root-yaw frame."""
    yaw_inv = _root_yaw_inv_mats(root_mats[1:])
    keys = np.asarray(SPARSE_VEL_JOINTS, dtype=np.int64)
    delta = canon_positions[1:, keys] - canon_positions[:-1, keys]
    local = np.einsum("tij,tkj->tki", yaw_inv, delta, optimize=True)
    return local.astype(np.float32, copy=False)

def derive_foot_contacts(
    canon_positions: np.ndarray,
    *,
    feet_threshold: float = FEET_VEL_THRESHOLD,
) -> np.ndarray:
    """Record-aligned ``[T-1, 4]`` foot contacts from canonical-position velocity.

    Contact for record ``i`` (the transition into frame ``i + 1``) is ``1`` when
    the squared displacement of the foot joint between frames ``i`` and
    ``i + 1`` is below ``feet_threshold``. Column order is
    ``[LeftFoot, LeftToeBase, RightFoot, RightToeBase]``.
    """
    if canon_positions.shape[0] < 2:
        raise ValueError("derive_foot_contacts requires at least two frames")
    keys = np.asarray(FOOT_CONTACT_JOINTS, dtype=np.int64)
    vel = ((canon_positions[1:, keys] - canon_positions[:-1, keys]) ** 2).sum(axis=-1)
    return (vel < feet_threshold).astype(np.float32)

def build_features_and_anchor(
    canon_positions: np.ndarray,        # [N, 77, 3]   canonical SOMA-X FK positions
    canon_axis_angle: np.ndarray,       # [N, 77, 3]   canonical-frame axis-angle (root lifted)
    *,
    target_fps: float = UMR_FPS,
    feet_threshold: float = FEET_VEL_THRESHOLD,
) -> tuple[np.ndarray, CanonicalAnchor]:
    """Pack the 499D records and the frame-0 anchor.

    Returns ``(features [N-1, 499], CanonicalAnchor)``. Foot contacts are
    derived from canonical-position velocity (see :func:`derive_foot_contacts`).
    """
    if canon_positions.shape[0] != canon_axis_angle.shape[0]:
        raise ValueError(
            f"T mismatch: canon_positions {canon_positions.shape[0]} vs "
            f"canon_axis_angle {canon_axis_angle.shape[0]}"
        )
    N = int(canon_positions.shape[0])
    if N < 2:
        raise ValueError(f"need at least 2 frames; got N={N}")

    rotmat77 = axis_angle_to_rotmat(canon_axis_angle)              # [N, 77, 3, 3]
    rot6d77 = rotmat_to_rot6d(rotmat77)                            # [N, 77, 6]
    root_mats = rotmat77[:, 0]                                     # [N, 3, 3]

    root_traj = _build_root_traj(
        canon_positions,
        root_mats,
        target_fps=target_fps,
    )                                                              # [N-1, 9]
    root_rot6d = rot6d77[1:, 0]                                    # [N-1, 6]
    joints76_rot6d = rot6d77[1:, 1:UMR_NUM_JOINTS]                 # [N-1, 76, 6]
    sparse_vel = _build_sparse_vel(canon_positions, root_mats)    # [N-1, 8, 3]
    foot_target = derive_foot_contacts(
        canon_positions, feet_threshold=feet_threshold
    )                                                              # [N-1, 4]

    features = pack_features(
        FeatureFields(
            root_traj=root_traj,
            root_rot6d=root_rot6d,
            joints76_rot6d=joints76_rot6d,
            sparse_vel=sparse_vel,
            foot_contact=foot_target,
        )
    )
    if features.shape != (N - 1, DIM_FEATURES):
        raise ValueError(f"features shape {features.shape} != ({N - 1}, {DIM_FEATURES})")

    anchor = CanonicalAnchor(
        init_root_pos=canon_positions[0, 0].astype(np.float32, copy=False),
        init_root_rot6d=rot6d77[0, 0].astype(np.float32, copy=False),
        init_joints76_rot6d=rot6d77[0, 1:UMR_NUM_JOINTS].astype(np.float32, copy=False),
    )
    return features, anchor

# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

def soma77_to_umr499(
    canonical: Soma77Canonical,
    *,
    joints77_world: np.ndarray | None = None,
    target_fps: float = UMR_FPS,
) -> UMR499:
    """Forward conversion ``Soma77Canonical → UMR499``.

    ``joints77_world`` must be pre-computed SOMA-X FK positions
    ``[T_src, 77, 3]`` at ``canonical.fps_src`` (see
    :func:`data.soma77_fk.soma77_joints_world_xyz`); source data is
    polyphase-resampled when ``canonical.fps_src != target_fps``.
    """
    if joints77_world is None:
        raise ValueError(
            "soma77_to_umr499: joints77_world is required; run "
            "`data.soma77_fk.soma77_joints_world_xyz(...)` first to obtain "
            "the SOMA-X FK joint world positions."
        )

    canonical.validate_shapes()
    if joints77_world.shape != (canonical.num_frames, UMR_NUM_JOINTS, 3):
        raise ValueError(
            f"joints77_world shape {joints77_world.shape} must be "
            f"({canonical.num_frames}, {UMR_NUM_JOINTS}, 3)"
        )
    identity_coeffs = np.asarray(canonical.identity_coeffs, dtype=np.float32)
    if identity_coeffs.ndim != 2:
        raise ValueError(f"identity_coeffs must be 2-D, got {identity_coeffs.shape}")
    if identity_coeffs.shape[0] != 1:
        # UMR499 carries FK static context. SOMA77 identity is expected to be
        # clip-static; if an upstream writer emitted [T, C], store the clip
        # mean as the static identity used for reconstruction/eval.
        identity_coeffs = identity_coeffs.mean(axis=0, keepdims=True).astype(np.float32)

    src_fps = float(canonical.fps_src)
    tgt_fps = float(target_fps)

    # Resample positions to target fps.
    joints77_rs = resample_positions(joints77_world, src_fps, tgt_fps)
    N = int(joints77_rs.shape[0])

    # Resample poses (per-joint SLERP keeps rotation arc continuous).
    poses_rs = slerp_axis_angle(canonical.poses, src_fps, tgt_fps, N)

    # Canonicalize positions; lift root axis-angle into the canonical frame.
    canon_positions, canon_quat_wxyz = canonicalize_positions(joints77_rs)
    canon_axis_angle = poses_rs.astype(np.float32, copy=True)
    canon_axis_angle[:, 0, :] = apply_canon_to_root_axis_angle(
        poses_rs[:, 0, :], canon_quat_wxyz
    )

    # Foot contacts are derived from canonical-position velocity inside
    # build_features_and_anchor; the stored canonical.foot_contacts are
    # intentionally not used.
    features, anchor = build_features_and_anchor(
        canon_positions,
        canon_axis_angle,
        target_fps=tgt_fps,
    )

    return UMR499(
        canonical_anchor=anchor,
        features=features,
        joints77_pos=canon_positions.astype(np.float32, copy=False),
        identity_coeffs=identity_coeffs.astype(np.float32, copy=False),
        joint_orient=canonical.joint_orient.astype(np.float32, copy=False),
        fps=tgt_fps,
        feature_variant=FEATURE_VARIANT,
        source_path=canonical.source_path,
    )

__all__ = [
    "FACE_JOINT_INDICES",
    "apply_canon_to_root_axis_angle",
    "axis_angle_to_rotmat",
    "build_features_and_anchor",
    "canonicalize_positions",
    "derive_foot_contacts",
    "resample_positions",
    "rotmat_to_rot6d",
    "slerp_axis_angle",
    "soma77_to_umr499",
]
