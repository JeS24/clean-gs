#!/usr/bin/env python3
"""
Clean-GS - Whitelist-based 3D Gaussian Splatting pruning

Two-stage approach with optional outlier removal:
1. Whitelist: Keep only Gaussians that project to object regions
2. Color validation: Remove floaters using depth buffering and color matching
3. Outlier removal (optional): Neighbor-based, spatial, or multi-view filtering
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

@dataclass
class Camera:
    image_name: str
    width: int
    height: int
    R: np.ndarray
    T: np.ndarray
    K: np.ndarray

def load_cameras(cameras_path):
    with open(cameras_path) as f:
        cameras_data = json.load(f)
    cameras = []
    for cam_data in cameras_data:
        cameras.append(Camera(
            image_name=cam_data['image_name'],
            width=cam_data['width'],
            height=cam_data['height'],
            R=np.array(cam_data['R']),
            T=np.array(cam_data['T']),
            K=np.array(cam_data['K'])
        ))
    return cameras

def project_gaussian_with_depth(xyz, camera):
    """Project 3D point to 2D image coordinates with depth"""
    R_w2c = camera.R.T
    T_w2c = -R_w2c @ camera.T
    xyz_cam = R_w2c @ xyz + T_w2c
    depth = xyz_cam[2]
    if depth <= 0:
        return None
    uv_h = camera.K @ xyz_cam
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    if 0 <= u < camera.width and 0 <= v < camera.height:
        return int(u), int(v), depth
    return None

def sh_to_rgb(sh_features):
    """Convert SH coefficients to RGB"""
    C0 = 0.28209479177387814
    rgb = sh_features[:3] * C0 + 0.5
    return np.clip(rgb, 0, 1)

def color_distance(c1, c2):
    """Euclidean distance between two RGB colors"""
    return np.sqrt(np.sum((c1 - c2) ** 2))

def process_view_whitelist(args):
    """Check if Gaussians project into OBJECT regions"""
    view_idx, cam, masked_images_dir, xyz = args
    num_gaussians = len(xyz)
    base_name = Path(cam.image_name).stem

    # Find mask file matching the image base name (any extension)
    masked_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        candidate = os.path.join(masked_images_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            masked_path = candidate
            break

    if masked_path is None:
        return None
    masked_img = cv2.imread(masked_path)
    if masked_img.shape[0] != cam.height or masked_img.shape[1] != cam.width:
        masked_img = cv2.resize(masked_img, (cam.width, cam.height))
    masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    object_mask = np.any(masked_rgb > 0.05, axis=2)
    projects_to_object = np.zeros(num_gaussians, dtype=np.bool_)
    for g_idx in range(num_gaussians):
        proj = project_gaussian_with_depth(xyz[g_idx], cam)
        if proj is not None:
            u, v, depth = proj
            if object_mask[v, u]:
                projects_to_object[g_idx] = True
    return projects_to_object

def process_view_color_validate(args):
    """Render with depth buffering and validate colors for whitelisted Gaussians"""
    view_idx, cam, masked_images_dir, xyz, rgb_colors, whitelist_mask, color_threshold = args
    whitelisted_indices = np.where(whitelist_mask)[0]
    if len(whitelisted_indices) == 0:
        return None
    base_name = Path(cam.image_name).stem

    # Find mask file matching the image base name (any extension)
    masked_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        candidate = os.path.join(masked_images_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            masked_path = candidate
            break

    if masked_path is None:
        return None
    masked_img = cv2.imread(masked_path)
    if masked_img.shape[0] != cam.height or masked_img.shape[1] != cam.width:
        masked_img = cv2.resize(masked_img, (cam.width, cam.height))
    masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    object_mask = np.any(masked_rgb > 0.05, axis=2)
    depth_buffer = np.full((cam.height, cam.width), np.inf)
    pixel_to_gaussian = {}
    for g_idx in whitelisted_indices:
        proj = project_gaussian_with_depth(xyz[g_idx], cam)
        if proj is not None:
            u, v, depth = proj
            if not object_mask[v, u]:
                continue
            if depth < depth_buffer[v, u]:
                depth_buffer[v, u] = depth
                pixel_to_gaussian[(u, v)] = g_idx
    num_gaussians = len(xyz)
    is_rendered = np.zeros(num_gaussians, dtype=np.bool_)
    color_match = np.zeros(num_gaussians, dtype=np.bool_)
    for (u, v), g_idx in pixel_to_gaussian.items():
        is_rendered[g_idx] = True
        expected_color = masked_rgb[v, u]
        gaussian_color = rgb_colors[g_idx]
        dist = color_distance(gaussian_color, expected_color)
        if dist < color_threshold:
            color_match[g_idx] = True
    return (is_rendered, color_match)

def clean_gs(ply_path, cameras_path, masked_images_dir, output_path,
             outlier_mode='neighbor',
             color_threshold=0.40,
             min_views=1,
             spatial_percentile=99,
             neighbor_percentile=95,
             num_workers=None):
    """
    Clean-GS with outlier removal options

    outlier_mode:
      - 'none': Basic cleaning (no extra outlier removal)
      - 'spatial': Remove far Gaussians (beyond spatial_percentile)
      - 'neighbor': Remove isolated Gaussians (beyond neighbor_percentile, default)
      - 'multiview': Require min_views appearances
      - 'combined': All three methods
    """
    if num_workers is None:
        num_workers = cpu_count()

    mode_name = {
        'none': 'Basic (No outlier removal)',
        'spatial': 'Spatial (Remove far Gaussians)',
        'neighbor': 'Neighbor (Remove isolated)',
        'multiview': 'MultiView (Require multi-view)',
        'combined': 'Combined (All methods)'
    }[outlier_mode]

    print("=" * 70)
    print(f"CLEAN-GS: {mode_name}")
    print("=" * 70)
    print(f"Workers: {num_workers} cores")
    print(f"Color threshold: {color_threshold:.2f}")
    if outlier_mode == 'multiview' or outlier_mode == 'combined':
        print(f"Min masked views: {min_views}")
    if outlier_mode == 'spatial' or outlier_mode == 'combined':
        print(f"Spatial percentile: {spatial_percentile}th")
    if outlier_mode == 'neighbor' or outlier_mode == 'combined':
        print(f"Neighbor percentile: {neighbor_percentile}th")
    print("=" * 70)

    # Load Gaussians
    print(f"\nLoading Gaussians from {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    num_gaussians = len(xyz)
    sh_features = np.zeros((num_gaussians, 3))
    for i in range(3):
        sh_features[:, i] = vertex[f'f_dc_{i}']
    rgb_colors = np.array([sh_to_rgb(sh) for sh in sh_features])
    print(f"  Loaded {num_gaussians:,} Gaussians")

    # Load cameras
    print(f"\nLoading cameras from {cameras_path}...")
    cameras = load_cameras(cameras_path)

    # Count mask files (any image extension)
    mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    masked_images = []
    for ext in mask_extensions:
        masked_images.extend(Path(masked_images_dir).glob(ext))

    print(f"  Loaded {len(cameras)} cameras ({len(masked_images)} with masks)")

    # STEP 1: WHITELIST (count views if multiview mode)
    print(f"\n[STEP 1] Building whitelist...")
    args_list = [(idx, cam, masked_images_dir, xyz) for idx, cam in enumerate(cameras)]
    
    if outlier_mode == 'multiview' or outlier_mode == 'combined':
        # Count how many masked views each Gaussian appears in
        view_count = np.zeros(num_gaussians, dtype=np.int32)
        with Pool(num_workers) as pool:
            for result in pool.imap_unordered(process_view_whitelist, args_list, chunksize=2):
                if result is not None:
                    view_count += result.astype(np.int32)
        whitelist_mask = view_count >= min_views
        print(f"  Whitelisted: {np.sum(whitelist_mask):,} (appear in >={min_views} views)")
    else:
        # Original: appear in at least 1 view
        projects_to_object_any = np.zeros(num_gaussians, dtype=np.bool_)
        with Pool(num_workers) as pool:
            for result in pool.imap_unordered(process_view_whitelist, args_list, chunksize=2):
                if result is not None:
                    projects_to_object_any |= result
        whitelist_mask = projects_to_object_any
        print(f"  Whitelisted: {np.sum(whitelist_mask):,} ({100*np.sum(whitelist_mask)/num_gaussians:.1f}%)")

    # STEP 2: COLOR VALIDATION
    print(f"\n[STEP 2] Color validation...")
    args_list = [
        (idx, cam, masked_images_dir, xyz, rgb_colors, whitelist_mask, color_threshold)
        for idx, cam in enumerate(cameras)
    ]
    rendered_count = np.zeros(num_gaussians, dtype=np.int32)
    match_count = np.zeros(num_gaussians, dtype=np.int32)
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(process_view_color_validate, args_list, chunksize=2):
            if result is not None:
                is_rendered, color_match = result
                rendered_count += is_rendered.astype(np.int32)
                match_count += color_match.astype(np.int32)

    has_renders = rendered_count > 0
    has_matches = match_count > 0
    keep_mask = whitelist_mask & ((~has_renders) | has_matches)
    color_validated = np.sum(keep_mask)
    color_removed = np.sum(whitelist_mask) - color_validated
    print(f"  Removed: {color_removed:,} floaters ({100*color_removed/num_gaussians:.1f}%)")
    print(f"  Kept: {color_validated:,} ({100*color_validated/num_gaussians:.1f}%)")

    # STEP 3: OUTLIER REMOVAL
    if outlier_mode in ['spatial', 'neighbor', 'combined']:
        print(f"\n[STEP 3] Outlier removal...")
        initial_kept = np.sum(keep_mask)
        
        # Spatial outlier removal
        if outlier_mode in ['spatial', 'combined']:
            center = np.mean(xyz[keep_mask], axis=0)
            distances = np.linalg.norm(xyz - center, axis=1)
            threshold = np.percentile(distances[keep_mask], spatial_percentile)
            spatial_outliers = (distances > threshold) & keep_mask
            keep_mask = keep_mask & ~spatial_outliers
            print(f"  Spatial: Removed {np.sum(spatial_outliers):,} far Gaussians (>{threshold:.2f})")
        
        # Neighbor-based outlier removal
        if outlier_mode in ['neighbor', 'combined']:
            try:
                from sklearn.neighbors import NearestNeighbors
                kept_indices = np.where(keep_mask)[0]
                kept_xyz = xyz[keep_mask]
                nbrs = NearestNeighbors(n_neighbors=min(10, len(kept_xyz)), algorithm='ball_tree').fit(kept_xyz)
                distances_knn, _ = nbrs.kneighbors(kept_xyz)
                avg_neighbor_dist = np.mean(distances_knn, axis=1)
                threshold = np.percentile(avg_neighbor_dist, neighbor_percentile)
                isolated_mask = avg_neighbor_dist > threshold
                isolated_indices = kept_indices[isolated_mask]
                keep_mask[isolated_indices] = False
                print(f"  Neighbor: Removed {len(isolated_indices):,} isolated Gaussians")
            except ImportError:
                print("  Warning: scikit-learn not installed, skipping neighbor filtering")
        
        total_removed = initial_kept - np.sum(keep_mask)
        print(f"  Total outliers removed: {total_removed:,}")

    # Final results
    num_keep = np.sum(keep_mask)
    num_remove = num_gaussians - num_keep
    print(f"\nFinal Results:")
    print(f"  Kept: {num_keep:,} ({100*num_keep/num_gaussians:.1f}%)")
    print(f"  Removed: {num_remove:,} ({100*num_remove/num_gaussians:.1f}%)")

    # Save
    print(f"\nSaving to {output_path}...")
    vertex_kept = vertex[keep_mask]
    el = PlyElement.describe(vertex_kept, 'vertex')
    PlyData([el]).write(output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved {num_keep:,} Gaussians")
    print(f"✓ File size: {file_size:.1f} MB")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--masked_images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=['none', 'spatial', 'neighbor', 'multiview', 'combined'], default='neighbor')
    parser.add_argument("--color_threshold", type=float, default=0.40)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--spatial_percentile", type=int, default=99)
    parser.add_argument("--neighbor_percentile", type=int, default=95)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    clean_gs(
        args.ply, args.cameras, args.masked_images, args.output,
        args.mode, args.color_threshold, args.min_views,
        args.spatial_percentile, args.neighbor_percentile, args.workers
    )
