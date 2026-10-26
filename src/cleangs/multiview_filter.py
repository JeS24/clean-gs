#!/usr/bin/env python3
"""Stage 2: filter a pretrained Gaussian scene using multiple views and export it."""

import os
from sklearn.cluster import HDBSCAN

import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from datetime import datetime
from scipy.spatial import cKDTree
import tqdm
import torch
try:
    from read_colmap import read_points3D_binary
except Exception:
    read_points3D_binary = None

try:
    import psutil
except ImportError:
    psutil = None

from .camera_utils import (
    load_cameras_from_colmap,
    load_cameras_json,
)

from gsplat.rendering import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



def run_clustering_hdbscan_sub(
    xyz, keep_top_n=1, max_points=200_000, smoothing=None, seed=None
):
    """Cluster sampled Gaussian centers with HDBSCAN and keep the largest clusters.

    Set the minimum cluster size to 0.1% of the sample, clamped to [20, 2000].
    smoothing sets min_samples; by default it equals this minimum cluster size.
    seed makes the sampling reproducible. Assign unsampled centers the label
    of their nearest sampled center.
    """
    n = len(xyz)

    sample_size = min(n, max_points)

    # min_cluster_size: 0.1% of the sample, clamped
    auto_mcs = int(np.clip(sample_size * 0.001, 20, 2000))
    mss = auto_mcs if smoothing is None else max(2, int(smoothing))
    print(f"  Cloud: {n:,} pts -> sample: {sample_size:,} | min_cluster_size={auto_mcs}"
          f" | min_samples={mss} | seed={seed}")

    if n > max_points:
        rng = np.random if seed is None else np.random.default_rng(seed)
        idx_sample = rng.choice(n, max_points, replace=False)
        xyz_sample = xyz[idx_sample]
    else:
        idx_sample = np.arange(n)
        xyz_sample = xyz

    clusterer = HDBSCAN(min_cluster_size=auto_mcs, min_samples=mss, n_jobs=-1)
    sample_labels = clusterer.fit_predict(xyz_sample)

    unique_labels, counts = np.unique(sample_labels[sample_labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        print("  No HDBSCAN clusters found; keeping all points.")
        return np.ones(n, dtype=bool)

    sorted_idx = np.argsort(-counts)
    print(f"  Found {len(unique_labels)} clusters, top sizes: {counts[sorted_idx[:5]]}")
    keep_labels = set(unique_labels[sorted_idx[:keep_top_n]])

    # Each point takes the label of its nearest sampled point
    sample_keep_mask = np.isin(sample_labels, list(keep_labels))
    
    if n > max_points:
        tree = cKDTree(xyz_sample)
        _, nn_idx = tree.query(xyz, k=1, workers=-1)
        full_mask = sample_keep_mask[nn_idx]
    else:
        full_mask = sample_keep_mask

    kept = np.sum(full_mask)
    print(f"  Keeping {kept:,} / {n:,} Gaussians ({100*kept/n:.1f}%)")
    return full_mask


def render_frame(splats, viewmat, K, width, height, sh_degree, device, near=0.01):
    """Render RGB [H, W, 3], accumulated depth [H, W] and alpha [H, W]."""
    sh0 = splats["sh0"]
    if sh0.dim() == 2:
        sh0 = sh0.unsqueeze(1)
    shN = splats.get("shN")
    if shN is not None:
        if shN.dim() == 2:
            shN = shN.view(sh0.shape[0], -1, 3)
        colors_cat = torch.cat([sh0, shN], dim=1)
    else:
        colors_cat = sh0

    rendered, alphas, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors_cat,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=near,
        far_plane=1e10,
        render_mode="RGB+D",
    )
    out = rendered.squeeze(0)
    rgb = out[:, :, :3].clamp(0, 1)
    depth = out[:, :, 3]
    alpha = alphas.squeeze(0).squeeze(-1)
    return rgb, depth, alpha


def get_masked_metrics(psnr_m, ssim_m, lpips_m, render, gt, mask, device):
    """Compute masked PSNR, SSIM and LPIPS for images shaped [1, 3, H, W].

    Return three NaN values if the mask is empty.
    """
    mask = mask.to(device)

    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    diff_sq = (render - gt) ** 2
    masked_mse = (diff_sq * mask).sum() / (mask.sum() * 3 + 1e-10)
    psnr_val = (10 * torch.log10(1.0 / (masked_mse + 1e-10))).item()

    # Average the SSIM map within the mask.
    ssim_full = StructuralSimilarityIndexMeasure(
        data_range=1.0, return_full_image=True
    ).to(device)
    _, ssim_map = ssim_full(render, gt)
    ssim_map_gray = ssim_map.mean(dim=1, keepdim=True)
    ssim_val = ((ssim_map_gray * mask).sum() / (mask.sum() + 1e-10)).item()

    # Compute LPIPS on the masked crop within the target's bounding box.
    mask_2d = mask.squeeze(0).squeeze(0)
    ys, xs = torch.where(mask_2d > 0.5)
    if len(ys) == 0:
        return psnr_val, ssim_val, float("nan")

    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1
    crop_mask = mask[:, :, y1:y2, x1:x2]
    render_crop = render[:, :, y1:y2, x1:x2] * crop_mask
    gt_crop = gt[:, :, y1:y2, x1:x2] * crop_mask

    _, _, ch, cw = render_crop.shape
    if ch < 64 or cw < 64:
        pad_h = max(0, 64 - ch)
        pad_w = max(0, 64 - cw)
        render_crop = torch.nn.functional.pad(render_crop, (0, pad_w, 0, pad_h))
        gt_crop = torch.nn.functional.pad(gt_crop, (0, pad_w, 0, pad_h))

    lpips_val = lpips_m(render_crop, gt_crop).item()
    return psnr_val, ssim_val, lpips_val


def _render_subset(
    t_splats: dict,
    subset_mask: np.ndarray,
    cameras: list,
    masks_dir: str,
    sh_degree: int,
    device: torch.device,
    num_gaussians: int,
    psnr_m,
    ssim_m,
    lpips_m,
    save_dir: Path = None,
) -> dict:
    """Render the Gaussians in subset_mask and return mean masked PSNR, SSIM and LPIPS."""
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    per_view_metrics = []

    idx = np.where(subset_mask)[0]
    if len(idx) == 0:
        print("No Gaussians in this subset; skipping evaluation.")
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "lpips": float("nan"),
            "n_views": 0,
        }

    idx_t = torch.from_numpy(idx).long().to(device)

    sub_splats = {k: v[idx_t] for k, v in t_splats.items()}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    eval_cams = cameras

    for cam in tqdm.tqdm(eval_cams, desc="    Rendering", leave=False):
        if cam.image_name is None or not os.path.exists(cam.image_path):
            continue
        gt_img = cv2.imread(cam.image_path)
        if gt_img is None:
            continue
        if gt_img.shape[0] != cam.height or gt_img.shape[1] != cam.width:
            gt_img = cv2.resize(gt_img, (cam.width, cam.height))
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        mask_np = None
        if masks_dir:
            stem = Path(cam.image_name).stem
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                mp = os.path.join(masks_dir, f"{stem}{ext}")
                if os.path.exists(mp):
                    mask_np = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    break
            if mask_np is not None:
                if mask_np.shape[0] != cam.height or mask_np.shape[1] != cam.width:
                    mask_np = cv2.resize(
                        mask_np,
                        (cam.width, cam.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_np = (mask_np > 127).astype(np.float32)
                mask_np_ones = np.ones_like(mask_np)

        if mask_np is None or mask_np.sum() == 0:
            continue

        R = torch.from_numpy(cam.R).float().to(device)
        T = torch.from_numpy(cam.T).float().to(device)
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        K = torch.from_numpy(cam.K).float().to(device).unsqueeze(0)

        with torch.no_grad():
            rgb_render, _, _ = render_frame(
                sub_splats, viewmat, K, cam.width, cam.height, sh_degree, device
            )

        gt_t = torch.from_numpy(gt_rgb).float().to(device) / 255.0
        gt_t = gt_t.permute(2, 0, 1).unsqueeze(0)
        rnd_t = rgb_render.permute(2, 0, 1).unsqueeze(0)
        msk_t = (
            torch.from_numpy(mask_np)
            .float()
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        psnr_v, ssim_v, lpips_v = get_masked_metrics(
            psnr_m, ssim_m, lpips_m, rnd_t, gt_t, msk_t, device
        )
        per_view_metrics.append(
            {
                "view": Path(cam.image_name).stem if cam.image_name is not None else "",
                "psnr": psnr_v,
                "ssim": ssim_v,
                "lpips": lpips_v,
            }
        )
        if not np.isnan(psnr_v):
            psnr_vals.append(psnr_v)
        if not np.isnan(ssim_v):
            ssim_vals.append(ssim_v)
        if not np.isnan(lpips_v):
            lpips_vals.append(lpips_v)

        if save_dir is not None:
            stem = Path(cam.image_name).stem
            render_np = (rgb_render.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                str(save_dir / f"{stem}_render.png"),
                cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR),
            )
            masked_render_np = (render_np * mask_np[:, :, None]).astype(np.uint8)
            cv2.imwrite(
                str(save_dir / f"{stem}_render_masked.png"),
                cv2.cvtColor(masked_render_np, cv2.COLOR_RGB2BGR),
            )

    if not psnr_vals:
        print("No views with readable images and nonempty masks could be evaluated.")
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "lpips": float("nan"),
            "n_views": 0,
        }

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
        "n_views": len(psnr_vals),
        "per_view": per_view_metrics,
    }


def evaluate_step(
    step_label: str,
    t_splats: dict,
    current_keep_mask: np.ndarray,
    previous_keep_mask: np.ndarray,
    cameras: list,
    masks_dir: str,
    sh_degree: int,
    device: torch.device,
    num_gaussians: int,
    metrics_log: list,
    run_dir: Path = None,
):
    """Log masked metrics for the Gaussians retained after a filtering step.

    Save renders to run_dir/renders/<step_label>/kept/ if run_dir is supplied.
    """
    print(f"\n  [Metrics] Evaluating {step_label} ...")

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
        device
    )

    removed_mask = previous_keep_mask & (~current_keep_mask)
    all_mask = np.ones(num_gaussians, dtype=bool)

    modes = {
        "kept": current_keep_mask,
    }

    step_result = {
        "step": step_label,
        "n_kept": int(current_keep_mask.sum()),
        "n_removed": int(removed_mask.sum()),
    }

    for mode_name, mask in modes.items():
        n = int(mask.sum())
        if n == 0:
            print(f"    [{mode_name}] No Gaussians — skipping render")
            step_result[f"{mode_name}_psnr"] = float("nan")
            step_result[f"{mode_name}_ssim"] = float("nan")
            step_result[f"{mode_name}_lpips"] = float("nan")
            step_result[f"{mode_name}_n_views"] = 0
            continue

        save_dir = None
        if run_dir is not None and mode_name == "kept":
            save_dir = Path(run_dir) / "renders" / step_label / mode_name

        print(
            f"    [{mode_name}] Rendering {n:,} Gaussians for evaluation ..."
        )
        res = _render_subset(
            t_splats=t_splats,
            subset_mask=mask,
            cameras=cameras,
            masks_dir=masks_dir,
            sh_degree=sh_degree,
            device=device,
            num_gaussians=num_gaussians,
            psnr_m=psnr_m,
            ssim_m=ssim_m,
            lpips_m=lpips_m,
            save_dir=save_dir,
        )
        step_result[f"{mode_name}_psnr"] = res["psnr"]
        step_result[f"{mode_name}_ssim"] = res["ssim"]
        step_result[f"{mode_name}_lpips"] = res["lpips"]
        step_result[f"{mode_name}_n_views"] = res["n_views"]
        step_result[f"{mode_name}_per_view"] = res.get("per_view", [])
        print(
            f"    [{mode_name}] PSNR={res['psnr']:.3f} dB  SSIM={res['ssim']:.4f}  "
            f"LPIPS={res['lpips']:.4f}  (over {res['n_views']} views)"
        )
        if save_dir is not None:
            print(f"    [{mode_name}] Renders saved to {save_dir}")

    metrics_log.append(step_result)
    return step_result


def print_metrics_table(metrics_log: list):
    """Print a table of the per-step metrics."""
    if not metrics_log:
        return

    header = (
        f"{'Step':<14} {'N_kept':>9} {'N_removed':>10} "
        f"{'All_PSNR':>9} {'All_SSIM':>9} {'All_LPIPS':>10} "
        f"{'Kpt_PSNR':>9} {'Kpt_SSIM':>9} {'Kpt_LPIPS':>10} "
        f"{'Rem_PSNR':>9} {'Rem_SSIM':>9} {'Rem_LPIPS':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("MASKED METRICS SUMMARY  (PSNR dB ↑ | SSIM ↑ | LPIPS ↓)")
    print(sep)
    print(header)
    print(sep)

    def _fmt(v):
        return f"{v:9.3f}" if v is not None and not np.isnan(v) else f"{'N/A':>9}"

    for r in metrics_log:
        print(
            f"{r['step']:<14} {r['n_kept']:>9,} {r['n_removed']:>10,} "
            f"{_fmt(r.get('all_psnr'))} {_fmt(r.get('all_ssim'))} {_fmt(r.get('all_lpips')):>10} "
            f"{_fmt(r.get('kept_psnr'))} {_fmt(r.get('kept_ssim'))} {_fmt(r.get('kept_lpips')):>10} "
            f"{_fmt(r.get('removed_psnr'))} {_fmt(r.get('removed_ssim'))} {_fmt(r.get('removed_lpips')):>10}"
        )
    print(sep + "\n")


def save_metrics_csv(metrics_log: list, output_path: Path):
    """Write aggregate metrics and available per-view metrics to CSV files."""
    import csv

    if not metrics_log:
        return
    aggregate_keys = [k for k in metrics_log[0].keys() if not k.endswith("_per_view")]
    csv_path = output_path / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_keys)
        writer.writeheader()
        for row in metrics_log:
            writer.writerow({k: row.get(k) for k in aggregate_keys})
    print(f"  Aggregate metrics saved to {csv_path}")

    pv_rows = []
    for row in metrics_log:
        step = row["step"]
        for mode in ("all", "kept"):
            pv_key = f"{mode}_per_view"
            if pv_key not in row or not row[pv_key]:
                continue
            for pv in row[pv_key]:
                pv_rows.append(
                    {
                        "step": step,
                        "mode": mode,
                        "view": pv.get("view", ""),
                        "psnr": pv.get("psnr"),
                        "ssim": pv.get("ssim"),
                        "lpips": pv.get("lpips"),
                    }
                )
    if pv_rows:
        pv_csv_path = output_path / "metrics_per_view.csv"
        with open(pv_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "mode", "view", "psnr", "ssim", "lpips"]
            )
            writer.writeheader()
            writer.writerows(pv_rows)
        print(f"  Per-view metrics saved to {pv_csv_path}")


def clean_gs(
    ply_path,
    data_dir,
    masked_images_dir,
    output_path,
    cameras_path=None,
    data_factor=1,
    normalize=True,
    test_every=8,
    sh_degree=3,
    steps=[1, 2],
    step1_outlier_mode="multiview",
    step2_outlier_mode="neighbor",
    min_views=1,
    spatial_percentile=99,
    neighbor_percentile=95,
    cluster_top_n=1,
    cluster_smoothing=None,
    cluster_seed=None,
    opacity_threshold=None,
    opacity_mode="threshold",
    scene_name=None,
    only_view_selection=True,
    compute_metrics=True,
    masks_dir=None,  # evaluation masks; default <data_dir>/masks_latest
    surface_distance_threshold=None,
):
    print("\n" + "=" * 70)

    start_time = datetime.now()

    run_dir = Path(output_path).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if scene_name is None:
        if "outputs" in str(run_dir):
            scene_name = run_dir.parent.parent.name
        else:
            scene_name = run_dir.name

    if compute_metrics:
        _masks_dir = masks_dir if masks_dir else os.path.join(data_dir, "masks_latest")
        if not os.path.isdir(_masks_dir):
            print(f"  Warning: masks dir '{_masks_dir}' not found — disabling metrics.")
            compute_metrics = False

    print(f"CLEAN-GS START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if only_view_selection:
        print("Testing visibility in selected views with all mask pixels set to true")
    else:
        print("Filtering with the selected target masks")

    if psutil:
        mem = psutil.virtual_memory()
        print(
            f"Memory: {mem.percent}% used ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)"
        )
    print(f"Scene: {scene_name}")
    print(f"Output directory: {run_dir}")
    print(f"Steps to run: {steps}")
    print(f"Step 1 Outlier mode: {step1_outlier_mode}")
    print(f"STEP 2 Outlier mode: {step2_outlier_mode}")
    print(f"PLY Path: {ply_path}")
    print(f"Data directory: {data_dir}")
    print(f"Masked images: {masked_images_dir}")
    if surface_distance_threshold is not None:
        print(f"Surface distance threshold: {surface_distance_threshold}")
    print(
        f"Compute metrics: {compute_metrics}  |  Masks dir: {_masks_dir if compute_metrics else 'N/A'}"
    )
    print("=" * 70)

    metrics_log = []

    def save_ply(current_mask, step_label):
        fname = f"{step_label}_{scene_name}_cgspp.ply"
        path = run_dir / fname
        vertex_kept = vertex[current_mask]
        el = PlyElement.describe(vertex_kept, "vertex")
        PlyData([el]).write(str(path))
        fsize = os.path.getsize(str(path)) / (1024 * 1024)
        print(
            f"Saved {np.sum(current_mask):,} Gaussians to {path} ({fsize:.1f} MB)"
        )
        return path

    # Load Gaussians
    if str(ply_path).endswith(".pt"):
        print(f"\nLoading Gaussians from gsplat checkpoint: {ply_path}...")
        ckpt = torch.load(ply_path, map_location="cpu")
        splats = ckpt["splats"] if "splats" in ckpt else ckpt

        means = splats["means"].detach().cpu().numpy()
        opacities = splats["opacities"].detach().cpu().numpy()
        scales = splats["scales"].detach().cpu().numpy()
        quats = splats["quats"].detach().cpu().numpy()
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        num_gaussians = len(means)

        if "sh0" in splats:
            sh0 = splats["sh0"].detach().cpu().numpy().reshape(-1, 3)
            if "shN" in splats:
                shN = splats["shN"].detach().cpu().numpy().reshape(-1, 45)
            else:
                shN = np.zeros((num_gaussians, 45), dtype=np.float32)
        else:
            sh0 = np.zeros((num_gaussians, 3), dtype=np.float32)
            shN = np.zeros((num_gaussians, 45), dtype=np.float32)

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("opacity", np.float32),
            ("scale_0", np.float32),
            ("scale_1", np.float32),
            ("scale_2", np.float32),
            ("rot_0", np.float32),
            ("rot_1", np.float32),
            ("rot_2", np.float32),
            ("rot_3", np.float32),
            ("f_dc_0", np.float32),
            ("f_dc_1", np.float32),
            ("f_dc_2", np.float32),
        ]
        for i in range(45):
            dtype.append((f"f_rest_{i}", np.float32))

        vertex = np.empty(num_gaussians, dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
        vertex["opacity"] = opacities.reshape(-1)
        vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = (
            scales[:, 0],
            scales[:, 1],
            scales[:, 2],
        )
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"] = (
            quats[:, 0],
            quats[:, 1],
            quats[:, 2],
            quats[:, 3],
        )
        vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = (
            sh0[:, 0],
            sh0[:, 1],
            sh0[:, 2],
        )
        for i in range(45):
            vertex[f"f_rest_{i}"] = shN[:, i]

        xyz = means
        sh_features = sh0
        print(f"  Loaded {num_gaussians:,} Gaussians from .pt")
    else:
        print(f"\nLoading Gaussians from {ply_path}...")
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        vertex_names = set(vertex.data.dtype.names)

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
        num_gaussians = len(xyz)

        if "opacity" in vertex_names:
            opacities = np.array(vertex["opacity"]).reshape(-1)
        else:
            opacities = np.full(num_gaussians, 10.0, dtype=np.float32)
            print("  Warning: 'opacity' not found in PLY, using default (10.0)")

        if "scale_0" in vertex_names:
            scales = np.stack(
                [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
            )
        else:
            scales = np.full((num_gaussians, 3), -5.0, dtype=np.float32)
            print("  Warning: 'scale_0/1/2' not found in PLY, using default (-5.0)")

        if "rot_0" in vertex_names:
            quats = np.stack(
                [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
                axis=1,
            )
            quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        else:
            quats = np.zeros((num_gaussians, 4), dtype=np.float32)
            quats[:, 0] = 1.0
            print(
                "  Warning: 'rot_0/1/2/3' not found in PLY, using default identity quaternion"
            )

        sh_features = np.zeros((num_gaussians, 3), dtype=np.float32)
        for i in range(3):
            if f"f_dc_{i}" in vertex_names:
                sh_features[:, i] = vertex[f"f_dc_{i}"]

        shN = np.zeros((num_gaussians, 45), dtype=np.float32)
        for i in range(45):
            if f"f_rest_{i}" in vertex_names:
                shN[:, i] = vertex[f"f_rest_{i}"]

        print(f"  Loaded {num_gaussians:,} Gaussians")


    # Load cameras
    if steps == [0]:
        cameras = []
        print("\nSkipping camera loading for opacity-only mode")
    elif data_dir and os.path.exists(data_dir):
        cameras = load_cameras_from_colmap(
            data_dir=data_dir,
            data_factor=data_factor,
            normalize=normalize,
            test_every=test_every,
        )
    elif cameras_path:
        print(f"\nLoading cameras from {cameras_path} (Fallback)...")
        cameras = load_cameras_json(cameras_path)
    else:
        if 1 in steps or 2 in steps:
            raise ValueError(
                "Either data_dir or cameras_path must be provided for steps 1 or 2."
            )
        cameras = []

    # Optional metrics use the held-out cameras only
    if compute_metrics and data_dir and os.path.exists(data_dir):
        eval_cameras = load_cameras_from_colmap(
            data_dir=data_dir,
            data_factor=data_factor,
            normalize=normalize,
            test_every=test_every,
            split="val",
        )
        if len(eval_cameras) == 0:
            print("  Warning: No 'val' cameras found for metrics — disabling metrics.")
            compute_metrics = False
            eval_cameras = []
        else:
            print(f"  Metric evaluation will use {len(eval_cameras)} val views.")
    else:
        eval_cameras = []

    # Tensors for gsplat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(ply_path).endswith(".pt"):
        t_splats = {k: v.to(device) for k, v in splats.items()}
    else:
        print("\n  Converting PLY data to gsplat tensors...")
        t_splats = {
            "means": torch.from_numpy(xyz).float().to(device),
            "opacities": torch.from_numpy(opacities).float().to(device),
            "scales": torch.from_numpy(scales).float().to(device),
            "quats": torch.from_numpy(quats).float().to(device),
            "sh0": torch.from_numpy(sh_features[:, None, :]).float().to(device),
            "shN": torch.from_numpy(shN.reshape(num_gaussians, 15, 3))
            .float()
            .to(device),
        }

    def save_all_renders(t_splats, cameras, run_dir, sh_degree, device):
        """Save renders of the full model to run_dir/renders/all/."""
        if not cameras or run_dir is None:
            return
        outdir = Path(run_dir) / "renders" / "all"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Saving full-model renders to {outdir} ...")
        psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_m = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
        _render_subset(
            t_splats=t_splats,
            subset_mask=np.ones(num_gaussians, dtype=bool),
            cameras=cameras,
            masks_dir=None,
            sh_degree=sh_degree,
            device=device,
            num_gaussians=num_gaussians,
            psnr_m=psnr_m,
            ssim_m=ssim_m,
            lpips_m=lpips_m,
            save_dir=outdir,
        )

    if compute_metrics and eval_cameras and run_dir is not None:
        save_all_renders(t_splats, eval_cameras, run_dir, sh_degree, device)

    # Step 0: opacity pruning (not used in the paper)
    if 0 in steps and opacity_threshold is not None:
        step0_start = datetime.now()
        print(
            f"\n[STEP 0] Opacity-based pruning (threshold={opacity_threshold}, mode={opacity_mode})..."
        )

        opacity_values = opacities
        if opacity_mode == "sigmoid":
            opacity_values = 1.0 / (1.0 + np.exp(-opacities))

        if opacity_mode in ["threshold", "sigmoid"]:
            keep_mask = opacity_values >= opacity_threshold
            removed = num_gaussians - np.sum(keep_mask)
            print(f"  Removed {removed:,} low-opacity Gaussians (<{opacity_threshold})")
        elif opacity_mode == "percentile":
            threshold_value = np.percentile(opacity_values, opacity_threshold)
            keep_mask = opacity_values >= threshold_value
            removed = num_gaussians - np.sum(keep_mask)
            print(
                f"  Removed bottom {opacity_threshold}% by opacity (threshold={threshold_value:.4f})"
            )
        else:
            print(
                f"  Warning: Unknown opacity_mode '{opacity_mode}', keeping all Gaussians"
            )
            keep_mask = np.ones(num_gaussians, dtype=bool)

        save_ply(keep_mask, "step0_opacity")

        if compute_metrics and eval_cameras:
            prev_mask = np.ones(num_gaussians, dtype=bool)
            evaluate_step(
                step_label="step0_opacity",
                t_splats=t_splats,
                current_keep_mask=keep_mask,
                previous_keep_mask=prev_mask,
                cameras=eval_cameras,
                masks_dir=_masks_dir,
                sh_degree=sh_degree,
                device=device,
                num_gaussians=num_gaussians,
                metrics_log=metrics_log,
                run_dir=run_dir,
            )

        print(f"  [STEP 0 Done] Time: {datetime.now() - step0_start}")
    else:
        keep_mask = np.ones(num_gaussians, dtype=bool)
        if 0 in steps:
            print(f"\n[STEP 0] Skipped (no opacity_threshold specified)")

    # Step 1: multiview vote over the selected masks
    if 1 not in steps:
        print(f"\n[STEP 1] Skipped (not in steps list)")
        whitelist_mask = keep_mask.copy()
    else:
        step1_start = datetime.now()
        print(f"\n[STEP 1] Building whitelist using gsplat projection...")
        print(f"  Using device: {device}")

        whitelist_mask = np.zeros(num_gaussians, dtype=np.bool_)
        view_count = np.zeros(num_gaussians, dtype=np.int32)

        for idx, cam in enumerate(tqdm.tqdm(cameras, desc="Rasterizing views")):
            base_name = Path(cam.image_name).stem
            masked_path = None
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                candidate = os.path.join(masked_images_dir, f"{base_name}{ext}")
                if os.path.exists(candidate):
                    masked_path = candidate
                    break
            if masked_path is None:
                continue

            mask_img = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            if mask_img.shape[0] != cam.height or mask_img.shape[1] != cam.width:
                mask_img = cv2.resize(
                    mask_img, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST
                )
            mask_torch = torch.from_numpy(mask_img > 127).to(device)
            if only_view_selection:
                mask_torch = mask_torch >= 0

            R = torch.from_numpy(cam.R).float().to(device)
            T = torch.from_numpy(cam.T).float().to(device)
            c2w = torch.eye(4, device=device)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            viewmat = torch.linalg.inv(c2w).unsqueeze(0)
            K = torch.from_numpy(cam.K).float().to(device).unsqueeze(0)

            with torch.no_grad():
                sh0 = t_splats["sh0"]
                if sh0.dim() == 2:
                    sh0 = sh0.unsqueeze(1)

                if "shN" in t_splats:
                    shN = t_splats["shN"]
                    if shN.dim() == 2:
                        shN = shN.view(num_gaussians, -1, 3)
                    colors_cat = torch.cat([sh0, shN], dim=1)
                else:
                    colors_cat = sh0

                render, _, info = rasterization(
                    means=t_splats["means"],
                    quats=t_splats["quats"],
                    scales=torch.exp(t_splats["scales"]),
                    opacities=torch.sigmoid(t_splats["opacities"]),
                    colors=colors_cat,
                    viewmats=viewmat,
                    Ks=K,
                    width=cam.width,
                    height=cam.height,
                    packed=False,
                    sh_degree=sh_degree,
                    near_plane=0.01,
                    far_plane=1e10,
                )

            radii = info["radii"].squeeze(0).amax(dim=-1)
            means2d = info["means2d"].squeeze(0)

            visible = radii > 0
            u = means2d[:, 0].long()
            v = means2d[:, 1].long()
            in_bounds = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            valid_pixel_indices = visible & in_bounds

            mask_membership = torch.zeros(
                num_gaussians, dtype=torch.bool, device=device
            )
            if valid_pixel_indices.any():
                v_idx = v[valid_pixel_indices]
                u_idx = u[valid_pixel_indices]
                mask_membership[valid_pixel_indices] = mask_torch[v_idx, u_idx]

            result_np = mask_membership.cpu().numpy()
            if step1_outlier_mode == "multiview" or step1_outlier_mode == "combined":
                view_count += result_np.astype(np.int32)
            else:
                whitelist_mask |= result_np

        if step1_outlier_mode == "multiview" or step1_outlier_mode == "combined":
            whitelist_mask = view_count >= min_views
            print(
                f"  Distribution of view counts: {np.bincount(view_count[view_count > 0])}"
            )

        # Optional: keep Gaussians near COLMAP points (needs read_colmap)
        if (
            surface_distance_threshold is not None
            and read_points3D_binary is not None
            and data_dir is not None
        ):
            print("Loading COLMAP points for the surface proximity check...")
            pts_path = None
            candidates = [
                os.path.join(data_dir, "sparse", "0", "points3D.bin"),
                os.path.join(data_dir, "sparse", "points3D.bin"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    pts_path = c
                    break
            if pts_path is not None:
                print("COLMAP points file: ",pts_path)
                try:
                    points_dict, points_xyz, _, _ = read_points3D_binary(pts_path)
                    if points_xyz.size == 0:
                        print("  Warning: points3D found but empty")
                    else:
                        tree = cKDTree(points_xyz)
                        not_whitelisted_idx = np.where(~whitelist_mask)[0]
                        if len(not_whitelisted_idx) > 0:
                            dists, _ = tree.query(xyz[not_whitelisted_idx], k=1)
                            keep_near = dists <= float(surface_distance_threshold)
                            promoted = not_whitelisted_idx[keep_near]
                            if promoted.size > 0:
                                whitelist_mask[promoted] = True
                                print(
                                    f"  Promoted {promoted.size:,} Gaussians by proximity to COLMAP surface (<={surface_distance_threshold})"
                                )
                except Exception as e:
                    print(
                        f"  Warning: Failed to read/parse points3D for surface proximity: {e}"
                    )
            else:
                print(
                    "  Info: No points3D.bin found under data_dir/sparse — skipping surface proximity check"
                )

        if 0 in steps:
            whitelist_mask = whitelist_mask & keep_mask

        kept_count = np.sum(whitelist_mask)
        print(
            f"  Whitelisted: {kept_count:,} ({100 * kept_count / num_gaussians:.1f}%)"
        )
        save_ply(whitelist_mask, "step1")

        if compute_metrics and eval_cameras:
            prev_mask = keep_mask.copy()  # what existed before step 1
            evaluate_step(
                step_label="step1_whitelist",
                t_splats=t_splats,
                current_keep_mask=whitelist_mask,
                previous_keep_mask=prev_mask,
                cameras=eval_cameras,
                masks_dir=_masks_dir,
                sh_degree=sh_degree,
                device=device,
                num_gaussians=num_gaussians,
                metrics_log=metrics_log,
                run_dir=run_dir,
            )

        print(f"  [STEP 1 Done] Time: {datetime.now() - step1_start}")

    keep_mask = whitelist_mask

    # Step 2: optional outlier removal or clustering (off in the main configuration)
    if 2 in steps and step2_outlier_mode in [
        "spatial",
        "neighbor",
        "cluster",
        "combined",
    ]:
        step2_start = datetime.now()
        print(f"\n[STEP 2] Outlier removal & Clustering ({step2_outlier_mode})...")
        initial_kept = np.sum(keep_mask)
        print(
            f"  [DEBUG] Gaussians entering STEP 2: {initial_kept:,} / {num_gaussians:,}"
        )
        pre_step2_mask = keep_mask.copy()

        if step2_outlier_mode in ["spatial", "combined"]:
            center = np.mean(xyz[keep_mask], axis=0)
            distances = np.linalg.norm(xyz - center, axis=1)
            threshold = np.percentile(distances[keep_mask], spatial_percentile)
            spatial_outliers = (distances > threshold) & keep_mask
            keep_mask = keep_mask & ~spatial_outliers
            print(
                f"  Spatial: Removed {np.sum(spatial_outliers):,} distant Gaussians "
                f"(>{threshold:.2f}, {spatial_percentile}th percentile)"
            )

        if step2_outlier_mode in ["neighbor"]:
            try:
                from sklearn.neighbors import NearestNeighbors

                kept_xyz = xyz[keep_mask]
                print(
                    f"  Neighbor: Finding nearest neighbors for {len(kept_xyz):,} points..."
                )
                nn_fit_start = datetime.now()
                nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(kept_xyz)
                print(f"  Neighbor: fit done in {datetime.now() - nn_fit_start}")
                nn_query_start = datetime.now()
                dists, _ = nbrs.kneighbors(kept_xyz)
                print(
                    f"  Neighbor: kneighbors done in {datetime.now() - nn_query_start}"
                )
                threshold = np.percentile(dists[:, 1], neighbor_percentile)
                isolated_mask = dists[:, 1] > threshold

                current_kept_indices = np.where(keep_mask)[0]
                to_remove = current_kept_indices[isolated_mask]
                keep_mask[to_remove] = False

                print(
                    f"  Neighbor: Removed {len(to_remove):,} isolated Gaussians "
                    f"(>{threshold:.4f}, {neighbor_percentile}th percentile)"
                )
            except Exception as e:
                print(f"  Warning: Neighbor filtering failed: {e}")

        if step2_outlier_mode in ["cluster"]:
            filtered_xyz = xyz[keep_mask]
            print(
                f"  Clustering: Running HDBSCAN on {len(filtered_xyz):,} points (top_n={cluster_top_n})..."
            )
            
            cluster_start = datetime.now()
            if len(filtered_xyz) > 0:
                cluster_mask = run_clustering_hdbscan_sub(
                    filtered_xyz,
                    keep_top_n=cluster_top_n,
                    smoothing=cluster_smoothing,
                    seed=cluster_seed,
                )
                print(
                    f"  Clustering: HDBSCAN completed in {datetime.now() - cluster_start}"
                )
                keep_mask[keep_mask] = cluster_mask
            print(f"  Clustering: Done — kept {np.sum(keep_mask):,} Gaussians")

        final_kept = np.sum(keep_mask)
        print(f"  Total Gaussians removed in STEP 2: {initial_kept - final_kept:,}")
        print(f"  STEP 2 outlier/cluster phase took: {datetime.now() - step2_start}")
        save_ply(keep_mask, "step2")

        if compute_metrics and eval_cameras:
            print(
                f"  [DEBUG] About to evaluate step2 metrics with {len(eval_cameras)} cameras"
            )
            eval_start = datetime.now()
            evaluate_step(
                step_label="step2_outlier",
                t_splats=t_splats,
                current_keep_mask=keep_mask,
                previous_keep_mask=pre_step2_mask,
                cameras=eval_cameras,
                masks_dir=_masks_dir,
                sh_degree=sh_degree,
                device=device,
                num_gaussians=num_gaussians,
                metrics_log=metrics_log,
                run_dir=run_dir,
            )
            print(
                f"  [DEBUG] evaluate_step for step2 took: {datetime.now() - eval_start}"
            )

        print(f"  [STEP 2 Done] Time: {datetime.now() - step2_start}")
    else:
        print(
            f"\n[STEP 2] Skipped because: steps={steps}, step2_mode={step2_outlier_mode}"
        )

    # Final output
    steps_str = "".join(map(str, sorted(steps)))
    final_path = save_ply(keep_mask, steps_str)

    if compute_metrics and metrics_log:
        print_metrics_table(metrics_log)
        save_metrics_csv(metrics_log, run_dir)

    total_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print(f"CLEAN-GS COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time}")
    print(f"Final output: {final_path}")
    print("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter a Gaussian scene using target masks and gsplat projections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ply", required=True, help="Input PLY file or gsplat checkpoint (.pt)")
    parser.add_argument("--data_dir", help="COLMAP scene directory")
    parser.add_argument("--cameras", help="Camera JSON file to use if no COLMAP directory is available")
    parser.add_argument("--masked_images", required=True, help="Directory of target masks for the selected views")
    parser.add_argument("--output_dir", required=True, help="Directory for exported assets and metrics")
    parser.add_argument("--scene_name", help="Scene name")
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Steps to run: 0=opacity pruning, 1=mask filtering, 2=outlier removal or clustering",
    )
    parser.add_argument(
        "--step1_mode", choices=["none", "multiview", "combined"], default="multiview"
    )
    parser.add_argument(
        "--step2_mode",
        choices=["none", "spatial", "neighbor", "cluster", "combined"],
        default="neighbor",
    )
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--spatial_percentile", type=int, default=99)
    parser.add_argument("--neighbor_percentile", type=int, default=95)
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        help="Opacity threshold for step 0",
    )
    parser.add_argument(
        "--opacity_mode",
        choices=["threshold", "sigmoid", "percentile"],
        default="threshold",
    )
    parser.add_argument("--cluster_top_n", type=int, default=1)
    parser.add_argument(
        "--cluster_smoothing", type=int, default=None,
        help="HDBSCAN min_samples; defaults to the automatically chosen minimum cluster size")
    parser.add_argument(
        "--cluster_seed", type=int, default=None,
        help="Random seed for reproducible cluster sampling; unset by default")
    parser.add_argument(
        "--only-view-selection",
        action="store_true",
        help="For diagnostics, set all pixels in each selected mask to true and "
        "filter by visibility in those views. "
    )

    parser.add_argument(
        "--no_metrics",
        dest="compute_metrics",
        action="store_false",
        default=True,
        help="Disable masked metrics evaluation after each step",
    )
    parser.add_argument(
        "--masks_dir",
        help=(
            "Directory of binary target masks for evaluation. "
            "Defaults to <data_dir>/masks_latest."
        ),
    )
    parser.add_argument(
        "--surface_distance_threshold",
        type=float,
        default=None,
        help="Also keep Gaussians within this distance of a COLMAP point, in world units, "
        "even if they lack enough mask votes (requires read_colmap)",
    )

    args = parser.parse_args()

    print("\n[CLEAN-GS] Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    clean_gs(
        ply_path=args.ply,
        data_dir=args.data_dir,
        masked_images_dir=args.masked_images,
        output_path=args.output_dir,
        cameras_path=args.cameras,
        data_factor=args.data_factor,
        normalize=args.normalize,
        test_every=args.test_every,
        sh_degree=args.sh_degree,
        steps=args.steps,
        step1_outlier_mode=args.step1_mode,
        step2_outlier_mode=args.step2_mode,
        min_views=args.min_views,
        spatial_percentile=args.spatial_percentile,
        neighbor_percentile=args.neighbor_percentile,
        cluster_top_n=args.cluster_top_n,
        cluster_smoothing=args.cluster_smoothing,
        cluster_seed=args.cluster_seed,
        opacity_threshold=args.opacity_threshold,
        opacity_mode=args.opacity_mode,
        scene_name=args.scene_name,
        only_view_selection=args.only_view_selection,
        compute_metrics=args.compute_metrics,
        masks_dir=args.masks_dir,
        surface_distance_threshold=args.surface_distance_threshold,
    )


if __name__ == "__main__":
    main()
