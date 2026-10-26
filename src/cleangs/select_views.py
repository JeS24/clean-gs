#!/usr/bin/env python3
"""Stage 1: select k views from a Pareto front.

Score each subset S of k candidate views on four objectives, then normalize
the scores to [0, 1]:

    A(S)  mask reliability: the minimum coverage score in S by default
    B(S)  mean pairwise camera-center baseline
    O(S)  mean pairwise dissimilarity of the in-mask SfM points
    T(S)  mean pairwise viewing-angle difference

By default, select the subset on the Pareto front with the smallest weighted
Chebyshev distance to the ideal point.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import struct
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
SAM3_SCRIPT = str(SCRIPT_DIR / "sam3" / "predict.py")
SAM3_PYTHONPATH = str(SCRIPT_DIR / "sam3")

DEFAULT_K = 3
DEFAULT_POOL = 60  # keep the top-N views by coverage score before enumeration
DEFAULT_W_COVERAGE = 0.25
DEFAULT_W_BASELINE = 0.30
DEFAULT_W_OVERLAP = 0.15
DEFAULT_W_ANGLE = 0.30
DEFAULT_MIN_COVERAGE = 0.00
DEFAULT_MAX_COVERAGE = 0.90
DEFAULT_TARGET_COVERAGE = 0.6
DEFAULT_COVERAGE_SIGMA = 0.15
DEFAULT_MAX_BORDER_TOUCH_SIDES = 10
DEFAULT_MIN_BBOX_MARGIN = 0.00
DEFAULT_MAX_BBOX_FRAC = 0.5


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_images_binary(paths: list) -> Dict:
    """Read COLMAP images.bin files into {image_id: entry}."""
    images = {}
    for path in paths:
        with open(path, "rb") as f:
            num_reg = _read_next_bytes(f, 8, "Q")[0]
            for _ in range(num_reg):
                img_id = _read_next_bytes(f, 4, "I")[0]
                qvec = np.array(_read_next_bytes(f, 32, "dddd"))  # w,x,y,z
                tvec = np.array(_read_next_bytes(f, 24, "ddd"))
                cam_id = _read_next_bytes(f, 4, "I")[0]
                name = b""
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    name += ch
                name = name.decode("utf-8")
                num_pts2d = _read_next_bytes(f, 8, "Q")[0]
                # Each keypoint is (x: double, y: double, point3D_id: int64).
                if num_pts2d:
                    xs_ys_ids = _read_next_bytes(f, 24 * num_pts2d, ("ddq" * num_pts2d))
                    xs = np.array(xs_ys_ids[0::3])
                    ys = np.array(xs_ys_ids[1::3])
                    ids = np.array(xs_ys_ids[2::3], dtype=np.int64)
                else:
                    xs = np.array([])
                    ys = np.array([])
                    ids = np.array([], dtype=np.int64)
                images[img_id] = dict(
                    name=name,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=cam_id,
                    keypoints=np.column_stack([xs, ys])
                    if xs.size and ys.size
                    else np.empty((0, 2)),
                    point3D_ids=ids,
                )
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def tvec_to_camera_centre(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return the camera center, -R^T t, from a COLMAP world-to-camera pose."""
    R = qvec_to_rotmat(qvec)
    return -(R.T @ tvec)


def run_sam3(dataset_path: str, prompt: str) -> str:
    import subprocess

    masks_dir = os.path.join(dataset_path, "masks_latest")
    if os.path.isdir(masks_dir) and os.listdir(masks_dir):
        logging.info("Masks already exist at %s — skipping SAM3.", masks_dir)
        return masks_dir
    if not os.path.isfile(SAM3_SCRIPT):
        raise FileNotFoundError(
            f"No masks at {masks_dir} and SAM3 is not bundled; pass --masks_dir."
        )
    logging.info("Running SAM3  prompt='%s'  dataset=%s", prompt, dataset_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = SAM3_PYTHONPATH + ":" + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        SAM3_SCRIPT,
        "--dataset",
        os.path.join(dataset_path, "images"),
        "--prompt",
        prompt,
    ]
    subprocess.run(cmd, check=True, env=env)
    if not os.path.isdir(masks_dir):
        raise RuntimeError(f"SAM3 did not produce masks at {masks_dir}")
    return masks_dir


def score_coverage(masks_dir: str) -> Dict[str, float]:
    """Return the fraction of foreground pixels for each mask filename."""
    scores: Dict[str, float] = {}
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mask = np.array(Image.open(os.path.join(masks_dir, fname)).convert("L"))
        scores[fname] = float(np.count_nonzero(mask > 127)) / float(mask.size)
    return scores


def _mask_fname_to_img_stem(mask_fname: str) -> str:
    return os.path.splitext(mask_fname)[0]


def bandpass_coverage_score(
    raw_coverage: float, target: float, sigma: float, score_type: str = "gaussian"
) -> float:
    """Score mask coverage in [0, 1], with the maximum at the target coverage."""
    if sigma <= 1e-9:
        return 1.0 if abs(raw_coverage - target) < 1e-9 else 0.0
    if score_type == "gaussian":
        z = (raw_coverage - target) / sigma
        return float(np.exp(-0.5 * z * z))
    elif score_type == "step":
        return 1.0 if abs(raw_coverage - target) <= sigma else 0.0
    elif score_type == "linear":
        diff = abs(raw_coverage - target)
        return float(max(0.0, 1.0 - diff / sigma))
    return 0.0


def compute_mask_geometry(mask_binary: np.ndarray) -> Dict[str, float]:
    """Measure which image borders a mask touches and its bounding-box size and margin."""
    h, w = mask_binary.shape
    ys, xs = np.where(mask_binary)
    if ys.size == 0:
        return {
            "touch_top": 0,
            "touch_bottom": 0,
            "touch_left": 0,
            "touch_right": 0,
            "num_touched_sides": 0,
            "bbox_margin_norm": 0.0,
            "bbox_w_frac": 0.0,
            "bbox_h_frac": 0.0,
        }

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    touch_top = int(mask_binary[0, :].any())
    touch_bottom = int(mask_binary[h - 1, :].any())
    touch_left = int(mask_binary[:, 0].any())
    touch_right = int(mask_binary[:, w - 1].any())
    num_touched = touch_top + touch_bottom + touch_left + touch_right

    bbox_w = float(x1 - x0 + 1)
    bbox_h = float(y1 - y0 + 1)
    bbox_w_frac = bbox_w / float(w)
    bbox_h_frac = bbox_h / float(h)

    margin_px = float(min(x0, y0, (w - 1 - x1), (h - 1 - y1)))
    bbox_margin_norm = margin_px / float(min(h, w))

    return {
        "touch_top": touch_top,
        "touch_bottom": touch_bottom,
        "touch_left": touch_left,
        "touch_right": touch_right,
        "num_touched_sides": int(num_touched),
        "bbox_margin_norm": float(bbox_margin_norm),
        "bbox_w_frac": float(bbox_w_frac),
        "bbox_h_frac": float(bbox_h_frac),
    }


def find_colmap_images_bin(scene: str) -> str:
    image_binaries = []
    for sub in ["colmap/sparse/0","colmap/sparse", "sparse/0", "colmap/sparse/1"]:
        p = os.path.join(scene, sub, "images.bin")
        if os.path.isfile(p):
            image_binaries.append(p)
    
    if(len(image_binaries) == 0):
        raise FileNotFoundError(f"images.bin not found under {scene}")
    
    return image_binaries

def build_view_data(
    colmap_images: Dict,
    masks_dir: str,
    coverage: Dict[str, float],
    min_coverage: float,
    max_coverage: float,
    target_coverage: float,
    coverage_sigma: float,
    max_border_touch_sides: int,
    min_bbox_margin: float,
    max_bbox_frac: float,
    images_dir: str,
    disable_heuristics: bool = False,
    coverage_score_type: str = "gaussian",
) -> Dict[str, dict]:
    """Build scores, camera poses and sets of SfM points inside each view's mask.

    Return records keyed by image name, excluding views that fail the coverage
    or containment tests.
    """
    by_name = {e["name"]: e for e in colmap_images.values()}

    by_name_keys = list(by_name.keys())[:4]
    logging.info("COLMAP image count: %d", len(by_name.keys()))
    logging.info("First image names: %s", by_name_keys)
    stem_to_mask: Dict[str, str] = {}
    for mf in coverage:
        stem_to_mask[_mask_fname_to_img_stem(mf)] = mf

    views: Dict[str, dict] = {}
    skipped = 0
    for img_name, entry in by_name.items():
        stem = os.path.splitext(img_name)[0]
        mask_fname = stem_to_mask.get(stem)
        if mask_fname is None:
            skipped += 1
            continue
        mask_path = os.path.join(masks_dir, mask_fname)
        mask_img = np.array(Image.open(mask_path).convert("L"))
        mask_bin = mask_img > 127

        cov = coverage[mask_fname]
        logging.info("Coverage: %f", cov)
        if cov < min_coverage:
            logging.info(
                "Skipping image %s: coverage %f < min_coverage %f",
                img_name,
                cov,
                min_coverage,
            )
            continue
        if cov > max_coverage:
            logging.info(
                "Skipping image %s: coverage %f > max_coverage %f",
                img_name,
                cov,
                max_coverage,
            )
            continue

        geom = compute_mask_geometry(mask_bin)
        if not disable_heuristics:
            if geom["num_touched_sides"] > max_border_touch_sides:
                logging.info(
                    "Mask touches %d image borders; maximum allowed: %d",
                    geom["num_touched_sides"],
                    max_border_touch_sides,
                )
                continue
            if geom["bbox_margin_norm"] < min_bbox_margin:
                logging.info("Bounding-box margin below minimum: %f", geom["bbox_margin_norm"])
                continue
            if (
                geom["bbox_w_frac"] > max_bbox_frac
                or geom["bbox_h_frac"] > max_bbox_frac
            ):
                logging.info("Bounding box exceeds the size limit (width fraction: %f)", geom["bbox_w_frac"])
                continue

        cov_score = bandpass_coverage_score(
            cov, target_coverage, coverage_sigma, coverage_score_type
        )

        qvec = entry["qvec"]
        tvec = entry["tvec"]
        centre = tvec_to_camera_centre(qvec, tvec)
        R = qvec_to_rotmat(qvec)
        viewing_dir = R.T[:, 2]

        # SfM points whose keypoints fall inside the mask
        h, w = mask_img.shape
        kps = entry["keypoints"]
        ids = entry["point3D_ids"]
        xi = np.clip(np.round(kps[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(kps[:, 1]).astype(int), 0, h - 1)
        inside = mask_img[yi, xi] > 127
        valid = inside & (ids >= 0)
        pt3d_mask = set(ids[valid].tolist())

        views[img_name] = dict(
            a=cov_score,
            a_raw=cov,
            num_touched_sides=geom["num_touched_sides"],
            bbox_margin_norm=geom["bbox_margin_norm"],
            bbox_w_frac=geom["bbox_w_frac"],
            bbox_h_frac=geom["bbox_h_frac"],
            centre=centre,
            viewing_dir=viewing_dir,
            pt3d_mask=pt3d_mask,
        )

    if skipped:
        logging.debug("Skipped %d COLMAP entries with no matching mask.", skipped)
    logging.info(
        "Built view data for %d images (coverage:[%.3f, %.3f], target=%.3f, sigma=%.3f, touch_sides<=%d, margin>=%.3f, bbox_frac<=%.3f).",
        len(views),
        min_coverage,
        max_coverage,
        target_coverage,
        coverage_sigma,
        max_border_touch_sides,
        min_bbox_margin,
        max_bbox_frac,
    )
    return views


def mask_level_pareto(views: Dict[str, dict]) -> Set[str]:
    """Return non-dominated views, favoring higher coverage scores and box margins
    and fewer contacts with the image borders.
    """
    names = list(views.keys())
    M = len(names)
    if M == 0:
        return set()
    obj = np.zeros((M, 3), dtype=float)
    for i, n in enumerate(names):
        v = views[n]
        obj[i, 0] = v.get("a", 0.0)
        obj[i, 1] = v.get("bbox_margin_norm", 0.0)
        obj[i, 2] = -float(v.get("num_touched_sides", 0))

    non_dom = []
    for i in range(M):
        others = np.delete(obj, i, axis=0)
        if not is_dominated(obj[i], others):
            non_dom.append(names[i])
    return set(non_dom)


def subset_coverage(
    names: Tuple[str, ...], views: Dict, aggregation: str = "min"
) -> float:
    """Compute A(S) as the minimum per-view score (used in the paper) or the mean."""
    vals = [views[n]["a"] for n in names]
    if not vals:
        return 0.0
    if aggregation == "mean":
        return float(np.mean(vals))
    return float(np.min(vals))


def subset_baseline(names: Tuple[str, ...], views: Dict) -> float:
    """B(S): mean pairwise distance between camera centers."""
    centres = [views[n]["centre"] for n in names]
    k = len(centres)
    if k < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += float(np.linalg.norm(centres[i] - centres[j]))
            count += 1
    return total / count if count > 0 else 0.0


def subset_overlap(
    names: Tuple[str, ...], views: Dict, overlap_type: str = "jaccard"
) -> float:
    """O(S): mean pairwise dissimilarity (1 - Jaccard by default) of in-mask SfM points."""
    sets = [views[n]["pt3d_mask"] for n in names]
    k = len(sets)
    if k < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            si, sj = sets[i], sets[j]
            inter = len(si & sj)
            if overlap_type == "min_containment":
                denom = min(len(si), len(sj))
                dissim = 1.0 if denom == 0 else 1.0 - inter / denom
            else:  # jaccard
                union = len(si | sj)
                dissim = 1.0 if union == 0 else 1.0 - inter / union
            total += dissim
            count += 1
    return total / count if count > 0 else 0.0


def subset_angle_diversity(names: Tuple[str, ...], views: Dict) -> float:
    """T(S): mean pairwise angle between viewing directions, in degrees."""
    dirs = [views[n]["viewing_dir"] for n in names]
    k = len(dirs)
    if k < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(k):
        di = dirs[i] / (np.linalg.norm(dirs[i]) + 1e-12)
        for j in range(i + 1, k):
            dj = dirs[j] / (np.linalg.norm(dirs[j]) + 1e-12)
            cos_ij = float(np.clip(np.dot(di, dj), -1.0, 1.0))
            total += math.degrees(math.acos(cos_ij))
            count += 1
    return total / count if count > 0 else 0.0


def _normalize(arr: np.ndarray, norm_type: str = "minmax") -> np.ndarray:
    if norm_type == "minmax":
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.ones_like(arr)
        return (arr - lo) / (hi - lo)
    elif norm_type == "zscore":
        # z-scores rescaled to [0, 1], like the other options
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return np.ones_like(arr)
        z = (arr - mean) / std
        lo, hi = z.min(), z.max()
        if hi == lo:
            return np.ones_like(z)
        return (z - lo) / (hi - lo)
    elif norm_type == "rank":
        from scipy.stats import rankdata

        ranked = rankdata(arr)
        return _normalize(ranked, "minmax")
    return arr


def is_dominated(v: np.ndarray, others: np.ndarray) -> bool:
    """Return whether another row dominates v when maximizing all objectives."""
    diff = others - v
    ge = (diff >= -1e-12).all(axis=1)
    gt = (diff > 1e-12).any(axis=1)
    return bool((ge & gt).any())


def pareto_front(
    subsets: List[Tuple[str, ...]],
    obj_matrix: np.ndarray,
) -> List[int]:
    """Return indices of non-dominated subsets using an exact O(N^2) comparison."""
    N = len(subsets)
    non_dom = []
    for i in range(N):
        others = np.delete(obj_matrix, i, axis=0)
        if not is_dominated(obj_matrix[i], others):
            non_dom.append(i)
    return non_dom


def scalarization_knee(
    front_indices: List[int],
    obj_matrix: np.ndarray,
    weights: np.ndarray,
    ideal_choice: str = "fixed",
    scalar_type: str = "chebyshev",
) -> int:
    if ideal_choice == "fixed":
        ideal = np.ones(obj_matrix.shape[1])
    else:
        ideal = obj_matrix[front_indices].max(axis=0)

    best_i = front_indices[0]
    best_d = np.inf

    if scalar_type == "chebyshev":
        sorted_front_indices = sorted(
            front_indices, key=lambda x: np.max(weights * np.abs(ideal - obj_matrix[x]))
        )
        for idx in front_indices:
            obj = obj_matrix[idx]
            d = float(np.max(weights * np.abs(ideal - obj)))
            if d < best_d:
                best_d = d
                best_i = idx
    elif scalar_type == "l2":
        sorted_front_indices = sorted(
            front_indices,
            key=lambda x: np.linalg.norm(weights * (ideal - obj_matrix[x])),
        )
        for idx in front_indices:
            obj = obj_matrix[idx]
            d = float(np.linalg.norm(weights * (ideal - obj)))
            if d < best_d:
                best_d = d
                best_i = idx
    elif scalar_type == "topsis":
        # Closeness C = d- / (d+ + d-) to the weighted ideal and anti-ideal points
        front_objs = obj_matrix[front_indices]
        weighted = front_objs * weights[np.newaxis, :]

        if ideal_choice == "fixed":
            ideal_vec = np.ones(obj_matrix.shape[1])
            anti_ideal_vec = np.zeros(obj_matrix.shape[1])
        else:
            ideal_vec = front_objs.max(axis=0)
            anti_ideal_vec = front_objs.min(axis=0)

        ideal_w = ideal_vec * weights
        anti_w = anti_ideal_vec * weights

        best_C = -np.inf
        sorted_front_indices = []
        closeness_scores = []
        for i, idx in enumerate(front_indices):
            row = weighted[i]
            d_plus = float(np.linalg.norm(row - ideal_w))
            d_minus = float(np.linalg.norm(row - anti_w))
            denom = d_plus + d_minus
            C = 0.0 if denom == 0.0 else d_minus / denom
            closeness_scores.append((idx, C))
            if C > best_C:
                best_C = C
                best_i = idx

        sorted_front_indices = [t[0] for t in sorted(closeness_scores, key=lambda x: -x[1])]

    return best_i, sorted_front_indices[:3]


def write_pareto_json(
    path: str,
    subsets: List[Tuple[str, ...]],
    front_indices: List[int],
    obj_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    knee_idx: int,
) -> None:
    entries = []
    for rank, idx in enumerate(front_indices):
        entries.append(
            dict(
                rank=rank,
                is_knee=(idx == knee_idx),
                views=list(subsets[idx]),
                A_coverage_norm=round(float(obj_matrix[idx, 0]), 6),
                B_baseline_norm=round(float(obj_matrix[idx, 1]), 6),
                O_overlap_norm=round(float(obj_matrix[idx, 2]), 6),
                T_angle_norm=round(float(obj_matrix[idx, 3]), 6),
                A_coverage_raw=round(float(raw_matrix[idx, 0]), 6),
                B_baseline_raw=round(float(raw_matrix[idx, 1]), 6),
                O_overlap_raw=round(float(raw_matrix[idx, 2]), 6),
                T_angle_raw=round(float(raw_matrix[idx, 3]), 6),
            )
        )
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def write_selected_views(path: str, views: List[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(views) + "\n")


def copy_masks(
    selected: List[str],
    masks_dir: str,
    output_masks_dir: str,
) -> None:
    if os.path.exists(output_masks_dir):
        shutil.rmtree(output_masks_dir)
    os.makedirs(output_masks_dir, exist_ok=True)
    for img_name in selected:
        stem = os.path.splitext(img_name)[0]
        for ext in (".png", ".jpg", ".jpeg",".PNG",".JPG",".JPEG",".tiff",".TIFF"):
            src = os.path.join(masks_dir, stem + ext)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_masks_dir, stem + ext))
                break


def write_log(
    path: str,
    views: Dict,
    subsets: List[Tuple[str, ...]],
    front_indices: List[int],
    knee_idx: int,
    obj_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    params: dict,
) -> None:
    knee_names = list(subsets[knee_idx])
    with open(path, "w") as f:
        f.write("=" * 70 + "\n Pareto View Selection Log\n" + "=" * 70 + "\n\n")
        f.write(f"Params: {params}\n\n")
        f.write(f"Total candidates: {len(views)}\n")
        f.write(f"Total subsets evaluated: {len(subsets)}\n")
        f.write(f"Pareto front size: {len(front_indices)}\n\n")
        f.write("--- Knee (selected) subset ---\n")
        for n in knee_names:
            f.write(
                f"  {n}  (coverage_score={views[n]['a']:.4f} "
                f"raw_coverage={views[n]['a_raw']:.4f} "
                f"touch_sides={views[n]['num_touched_sides']} "
                f"bbox_margin={views[n]['bbox_margin_norm']:.4f} "
                f"bbox_w={views[n]['bbox_w_frac']:.4f} bbox_h={views[n]['bbox_h_frac']:.4f} "
                f"|pts_in_mask|={len(views[n]['pt3d_mask'])})\n"
            )
        f.write("\n--- Pareto front (all non-dominated subsets) ---\n")
        f.write(
            f"  {'Rank':>4}  {'Views':<60}  "
            f"{'A(cov)':>8}  {'B(base)':>8}  {'O(ovlp)':>8}  {'T(angle)':>8}\n"
        )
        f.write("  " + "-" * 108 + "\n")
        for rank, idx in enumerate(front_indices):
            marker = " ← knee" if idx == knee_idx else ""
            view_str = ", ".join(subsets[idx])
            f.write(
                f"  {rank:>4}  {view_str:<60}  "
                f"{obj_matrix[idx, 0]:>8.4f}  {obj_matrix[idx, 1]:>8.4f}  "
                f"{obj_matrix[idx, 2]:>8.4f}  {obj_matrix[idx, 3]:>8.4f}{marker}\n"
            )
        f.write("\n--- Per-view coverage scores (all candidates) ---\n")
        for name, vdata in sorted(views.items(), key=lambda x: -x[1]["a"]):
            f.write(
                f"  {name:<50}  coverage_score={vdata['a']:.4f} "
                f"raw_coverage={vdata['a_raw']:.4f}  "
                f"touch_sides={vdata['num_touched_sides']} "
                f"bbox_margin={vdata['bbox_margin_norm']:.4f} "
                f"bbox_w={vdata['bbox_w_frac']:.4f} bbox_h={vdata['bbox_h_frac']:.4f}  "
                f"|pt3d_mask|={len(vdata['pt3d_mask'])}\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select views from a Pareto front of mask and camera scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, help="Scene root directory")
    ap.add_argument("--prompt", default="object", help="SAM3 text prompt")
    ap.add_argument("--masks_dir", default=None, help="Use existing masks from this directory and skip SAM3")
    ap.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Number of views to select per subset"
    )
    ap.add_argument(
        "--candidate_pool",
        type=int,
        default=DEFAULT_POOL,
        help="Keep the N views with the highest coverage scores before enumerating subsets. "
        "Set to 0 to use all eligible views (can be slow).",
    )
    ap.add_argument(
        "--min_coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="Discard views with mask coverage below this fraction",
    )
    ap.add_argument(
        "--max_coverage",
        type=float,
        default=DEFAULT_MAX_COVERAGE,
        help="Discard views with mask coverage above this fraction",
    )
    ap.add_argument(
        "--target_coverage",
        type=float,
        default=DEFAULT_TARGET_COVERAGE,
        help="Mask coverage fraction assigned the highest score",
    )
    ap.add_argument(
        "--coverage_sigma",
        type=float,
        default=DEFAULT_COVERAGE_SIGMA,
        help="Width of the coverage score around the target fraction",
    )
    ap.add_argument(
        "--max_border_touch_sides",
        type=int,
        default=DEFAULT_MAX_BORDER_TOUCH_SIDES,
        help="Reject views whose mask touches more than this many image sides",
    )
    ap.add_argument(
        "--min_bbox_margin",
        type=float,
        default=DEFAULT_MIN_BBOX_MARGIN,
        help="Minimum mask bounding-box margin, as a fraction of the shorter image dimension",
    )
    ap.add_argument(
        "--max_bbox_frac",
        type=float,
        default=DEFAULT_MAX_BBOX_FRAC,
        help="Maximum mask bounding-box width or height as a fraction of the image dimension",
    )
    ap.add_argument(
        "--w_coverage",
        type=float,
        default=DEFAULT_W_COVERAGE,
        help="Weight for mask reliability A(S)",
    )
    ap.add_argument(
        "--w_baseline",
        type=float,
        default=DEFAULT_W_BASELINE,
        help="Weight for camera baseline B(S)",
    )
    ap.add_argument(
        "--w_overlap",
        type=float,
        default=DEFAULT_W_OVERLAP,
        help="Weight for SfM point dissimilarity O(S)",
    )
    ap.add_argument(
        "--w_angle",
        type=float,
        default=DEFAULT_W_ANGLE,
        help="Weight for viewing-direction diversity T(S)",
    )
    ap.add_argument(
        "--output_masks_dir",
        default=None,
        help="If given, copy selected masks to <dataset>/<output_masks_dir>",
    )
    ap.add_argument("--log_file", default="pareto_view_selection.log")
    ap.add_argument("-v", "--verbose", action="store_true")

    # Ablations
    ap.add_argument(
        "--skip_pareto",
        action="store_true",
        help="Rank all subsets with the chosen metric, skipping Pareto filtering",
    )
    ap.add_argument(
        "--coverage_aggregation",
        type=str,
        default="min",
        choices=["min", "mean"],
        help="Combine per-view coverage scores into A(S) using the minimum "
        "(used in the paper) or the mean (for comparison).",
    )
    ap.add_argument(
        "--coverage_score_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "step", "linear"],
        help="Function used to score mask coverage relative to the target fraction",
    )
    ap.add_argument(
        "--normalization_type",
        type=str,
        default="minmax",
        choices=["minmax", "rank", "zscore"],
        help="Method used to normalize each objective across subsets",
    )
    ap.add_argument(
        "--scalarization_type",
        type=str,
        default="chebyshev",
        choices=["chebyshev", "l2", "topsis"],
        help="Metric used to rank subsets on the Pareto front",
    )
    ap.add_argument(
        "--ideal_point_choice",
        type=str,
        default="fixed",
        choices=["fixed", "observed"],
        help="Use an all-ones ideal point or the best observed value of each objective",
    )
    ap.add_argument(
        "--disable_heuristics",
        action="store_true",
        help="Skip border-contact and bounding-box tests; coverage limits still apply",
    )
    ap.add_argument(
        "--overlap_type",
        type=str,
        default="jaccard",
        choices=["jaccard", "min_containment"],
        help="Pairwise dissimilarity metric for O(S)",
    )
    ap.add_argument(
        "--select_rank",
        type=int,
        default=0,
        help="Choose among the top three ranked subsets by index (0 selects the best).",
    )
    ap.add_argument(
        "--merge_top_k",
        type=int,
        default=0,
        help="If positive, combine views from this many of the top three ranked subsets.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    scene = os.path.abspath(args.dataset)

    w = np.array(
        [args.w_coverage, args.w_baseline, args.w_overlap, args.w_angle], dtype=float
    )
    if w.sum() <= 0:
        raise ValueError("At least one weight must be positive.")
    w /= w.sum()
    logging.info(
        "Objective weights  coverage=%.3f  baseline=%.3f  overlap=%.3f  angle=%.3f",
        w[0],
        w[1],
        w[2],
        w[3],
    )

    # 1. Masks
    masks_dir = args.masks_dir if args.masks_dir else run_sam3(scene, args.prompt)
    logging.info("Masks dir: %s", masks_dir)

    coverage = score_coverage(masks_dir)
    logging.info("Scored %d mask images.", len(coverage))

    # 2. COLMAP
    colmap_bin = find_colmap_images_bin(scene)
    logging.info("Loading COLMAP images.bin from %s …", colmap_bin)
    colmap_images = read_images_binary(colmap_bin)
    logging.info("Loaded %d COLMAP images.", len(colmap_images))

    images_dir = os.path.join(scene, "images")
    if not os.path.isdir(images_dir):
        images_dir = scene

    # 3. Per-view data
    views = build_view_data(
        colmap_images,
        masks_dir,
        coverage,
        args.min_coverage,
        args.max_coverage,
        args.target_coverage,
        args.coverage_sigma,
        args.max_border_touch_sides,
        args.min_bbox_margin,
        args.max_bbox_frac,
        images_dir,
        args.disable_heuristics,
        args.coverage_score_type,
    )

    if len(views) < args.k:
        logging.warning(
            "Only %d views passed coverage and containment filtering (need at least %d). "
            "Adjust the thresholds or add more images.",
            len(views),
            args.k,
        )
        return

    # 4. Candidate pool: non-dominated masks, then the pool cap
    mask_pareto_keep = mask_level_pareto(views)
    if (
        mask_pareto_keep
        and args.k <= len(mask_pareto_keep) < len(views)
    ):
        logging.info(
            "Mask-level Pareto filtered %d -> %d masks.",
            len(views),
            len(mask_pareto_keep),
        )
        views = {n: views[n] for n in views.keys() if n in mask_pareto_keep}
    elif mask_pareto_keep and len(mask_pareto_keep) < args.k:
        logging.warning(
            "Mask-level Pareto would leave only %d masks for k=%d; "
            "retaining all %d eligible masks.",
            len(mask_pareto_keep),
            args.k,
            len(views),
        )
    all_names = list(views.keys())
    pool_size = args.candidate_pool if args.candidate_pool > 0 else len(all_names)
    if pool_size < len(all_names):
        all_names = sorted(all_names, key=lambda n: views[n]["a"], reverse=True)[
            :pool_size
        ]
        logging.info(
            "Candidate pool pre-filtered to top-%d by coverage score.", pool_size
        )

    n_subsets = math.comb(len(all_names), args.k)
    logging.info(
        "Evaluating C(%d,%d) = %d subsets …", len(all_names), args.k, n_subsets
    )
    if n_subsets > 500_000:
        logging.warning(
            "Very large enumeration (%d subsets). Consider reducing --candidate_pool.",
            n_subsets,
        )

    # 5. Score every subset
    subsets: List[Tuple[str, ...]] = []
    raw_A: List[float] = []
    raw_B: List[float] = []
    raw_O: List[float] = []
    raw_T: List[float] = []

    for combo in combinations(all_names, args.k):
        subsets.append(combo)
        raw_A.append(subset_coverage(combo, views, args.coverage_aggregation))
        raw_B.append(subset_baseline(combo, views))
        raw_O.append(subset_overlap(combo, views, args.overlap_type))
        raw_T.append(subset_angle_diversity(combo, views))

    raw_matrix = np.column_stack([raw_A, raw_B, raw_O, raw_T])
    obj_matrix = np.column_stack(
        [
            _normalize(raw_matrix[:, 0], args.normalization_type),
            _normalize(raw_matrix[:, 1], args.normalization_type),
            _normalize(raw_matrix[:, 2], args.normalization_type),
            _normalize(raw_matrix[:, 3], args.normalization_type),
        ]
    )

    logging.info("Objectives computed for %d subsets.", len(subsets))

    # 6. Pareto front
    if args.skip_pareto:
        logging.info("Skipping Pareto front, using all subsets ...")
        front_indices = list(range(len(subsets)))
    else:
        logging.info("Computing exact Pareto front …")
        front_indices = pareto_front(subsets, obj_matrix)
        logging.info(
            "Pareto front contains %d non-dominated subsets.", len(front_indices)
        )

    # 7. Selection
    knee_idx, knee_indices = scalarization_knee(
        front_indices,
        obj_matrix,
        w,
        ideal_choice=args.ideal_point_choice,
        scalar_type=args.scalarization_type,
    )

    if args.merge_top_k > 0:
        top_k_indices = knee_indices[: args.merge_top_k]
        knee_names = sorted(list(set(v for idx in top_k_indices for v in subsets[idx])))
        logging.info(
            "Merging top-%d subsets. Total unique views: %d",
            args.merge_top_k,
            len(knee_names),
        )
        selected_desc = f"Union of top-{args.merge_top_k}"
    else:
        if args.select_rank < len(knee_indices):
            actual_idx = knee_indices[args.select_rank]
        else:
            logging.warning(
                "Requested rank %d is outside the %d available ranked subsets. Using the best subset.",
                args.select_rank,
                len(knee_indices),
            )
            actual_idx = knee_idx
        
        knee_idx = actual_idx
        knee_names = list(subsets[knee_idx])
        selected_desc = f"Rank {args.select_rank}"

    logging.info("Selected subset (%s, knee_idx=%d): %s", selected_desc, knee_idx, ", ".join(knee_names))
    for n in knee_names:
        logging.info(
            "  %s  coverage_score=%.4f raw_coverage=%.4f",
            n,
            views[n]["a"],
            views[n]["a_raw"],
        )
    logging.info(
        "  A=%.4f  B=%.4f  O=%.4f  T=%.4f  (normalised)",
        obj_matrix[knee_idx, 0],
        obj_matrix[knee_idx, 1],
        obj_matrix[knee_idx, 2],
        obj_matrix[knee_idx, 3],
    )

    # 8. Outputs; stage 2 reads the copied masks
    sv_path = os.path.join(scene, "selected_views.txt")
    write_selected_views(sv_path, knee_names)
    logging.info("-> %s", sv_path)

    sv_pareto_path = os.path.join(scene, "selected_views_pareto.txt")
    with open(sv_pareto_path, "w") as f:
        for rank, idx in enumerate(front_indices):
            marker = "  # knee" if idx == knee_idx else ""
            f.write(", ".join(subsets[idx]) + marker + "\n")
    logging.info("-> %s  (%d subsets)", sv_pareto_path, len(front_indices))

    json_path = os.path.join(scene, "pareto_front.json")
    write_pareto_json(
        json_path, subsets, front_indices, obj_matrix, raw_matrix, knee_idx
    )
    logging.info("-> %s", json_path)

    log_path = os.path.join(scene, args.log_file)
    write_log(
        log_path,
        views,
        subsets,
        front_indices,
        knee_idx,
        obj_matrix,
        raw_matrix,
        vars(args),
    )
    logging.info("-> %s", log_path)

    if args.output_masks_dir:
        dst = os.path.join(scene, args.output_masks_dir)
        copy_masks(knee_names, masks_dir, dst)
        logging.info("-> Selected masks copied to %s", dst)

    # 9. Summary
    print("\n" + "=" * 70)
    print(f" Pareto View Selection  —  {len(front_indices)} non-dominated subsets")
    print("=" * 70)
    print(
        f"  {'Rank':>4}  {'A(cov)':>8}  {'B(base)':>8}  {'O(ovlp)':>8}  {'T(angle)':>8}  Views"
    )
    print("  " + "-" * 78)
    for rank, idx in enumerate(front_indices[:20]):  # print at most 20
        marker = " ← knee" if idx == knee_idx else ""
        vs = ", ".join(subsets[idx])
        print(
            f"  {rank:>4}  {obj_matrix[idx, 0]:>8.4f}  "
            f"{obj_matrix[idx, 1]:>8.4f}  {obj_matrix[idx, 2]:>8.4f}  "
            f"{obj_matrix[idx, 3]:>8.4f}  {vs}{marker}"
        )
    if len(front_indices) > 20:
        print(f"  … and {len(front_indices) - 20} more (see {json_path})")
    print()
    print("Selected (knee):", ", ".join(knee_names))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
