"""
Pose overlap utilities for deduplicating redundant camera views.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-6


def rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic rotation angle in degrees between two 3x3 rotation matrices."""
    R_rel = R_a.T @ R_b
    trace = float(np.trace(R_rel))
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(angle_rad))


def translation_distance(t_a: np.ndarray, t_b: np.ndarray) -> float:
    return float(np.linalg.norm(t_a - t_b))


def pose_rotation(pose: np.ndarray) -> np.ndarray:
    return pose[:3, :3]


def pose_translation(pose: np.ndarray) -> np.ndarray:
    return pose[:3, 3]


def compute_scene_scale(positions: np.ndarray) -> float:
    """
    Robust scene scale from camera positions: max pairwise distance,
    with centroid-std fallback when degenerate.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return 1.0

    n = positions.shape[0]
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            max_dist = max(max_dist, d)

    if max_dist > _EPS:
        return max_dist

    std = float(np.std(positions, axis=0).max())
    return max(std, _EPS)


def compute_path_span(positions: np.ndarray) -> float:
    """Max pairwise distance among a set of kept positions."""
    return compute_scene_scale(np.asarray(positions, dtype=np.float64))


def translation_threshold(
    scene_scale: float,
    kept_positions: Sequence[np.ndarray],
    *,
    frac_scene: float = 0.08,
    frac_path: float = 0.20,
    min_abs: float = 1e-3,
    max_frac: float = 0.25,
) -> float:
    """
    Adaptive translation threshold from scene scale and exploration so far.
    """
    scene_scale = max(float(scene_scale), _EPS)
    kept = list(kept_positions)
    if len(kept) >= 2:
        path_span = compute_path_span(np.stack(kept, axis=0))
    else:
        path_span = scene_scale
    path_span = max(path_span, _EPS)

    thresh = max(
        min_abs,
        frac_scene * scene_scale,
        frac_path * path_span,
    )
    thresh = min(thresh, max_frac * scene_scale)
    return float(thresh)


def is_overlapping(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    rot_deg_thresh: float,
    trans_thresh: float,
) -> Tuple[bool, float, float]:
    """
    True if both rotation and translation are within thresholds (AND logic).
    Returns (overlapping, rot_deg, trans_dist).
    """
    rot_deg = rotation_geodesic_deg(pose_rotation(pose_a), pose_rotation(pose_b))
    trans_dist = translation_distance(pose_translation(pose_a), pose_translation(pose_b))
    overlapping = rot_deg <= rot_deg_thresh and trans_dist <= trans_thresh
    return overlapping, rot_deg, trans_dist


def dedupe_frames_by_pose_overlap(
    selected_indices: Sequence[int],
    poses: Mapping[int, np.ndarray],
    *,
    rot_deg_thresh: float = 10.0,
    frac_scene: float = 0.08,
    frac_path: float = 0.20,
    min_abs: float = 1e-3,
    max_frac: float = 0.25,
    min_kept_views: int = 3,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Walk selected frames in temporal order; drop frames redundant with any kept frame.

    Returns:
        kept_indices: sorted list of frame indices to keep
        dropped_records: debug info for each dropped frame
    """
    indices = sorted(selected_indices)
    if not indices:
        return [], []

    selected_positions = np.stack([pose_translation(poses[i]) for i in indices], axis=0)
    scene_scale = compute_scene_scale(selected_positions)

    kept: List[int] = []
    dropped_records: List[Dict[str, Any]] = []

    for pos, idx in enumerate(indices):
        pose = poses[idx]
        if not kept:
            kept.append(idx)
            continue

        kept_positions = [pose_translation(poses[k]) for k in kept]
        trans_thresh = translation_threshold(
            scene_scale,
            kept_positions,
            frac_scene=frac_scene,
            frac_path=frac_path,
            min_abs=min_abs,
            max_frac=max_frac,
        )

        is_redundant = False
        best_match: Optional[Dict[str, Any]] = None
        for kept_idx in kept:
            overlapping, rot_deg, trans_dist = is_overlapping(
                pose, poses[kept_idx], rot_deg_thresh, trans_thresh
            )
            if overlapping:
                is_redundant = True
                if best_match is None or trans_dist < best_match["trans_dist"]:
                    best_match = {
                        "closest_kept": kept_idx,
                        "rot_deg": rot_deg,
                        "trans_dist": trans_dist,
                        "trans_thresh": trans_thresh,
                    }

        # Frames after current (if we drop current, max kept = len(kept) + remaining_after)
        remaining_after = len(indices) - pos - 1
        would_drop_below_min = len(kept) + remaining_after < min_kept_views

        if is_redundant and not would_drop_below_min:
            dropped_records.append(
                {
                    "index": idx,
                    "reason": "overlap_with_prior",
                    **(best_match or {}),
                }
            )
        else:
            kept.append(idx)

    return kept, dropped_records
