#!/usr/bin/env python3
"""Run ablation study for Clean-GS on Temple dataset."""

import subprocess
import os

# Paths
PLY = "data/models/temple/point_cloud/iteration_30000/point_cloud.ply"
CAMERAS = "data/datasets/temple/cameras.json"
MASKS = "data/masks/temple"
OUTPUT_DIR = "/tmp/ablation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ABLATION STUDY: Temple Dataset")
print("=" * 70)

# Stage 1: Whitelist + Color Validation (mode=none, normal threshold)
print("\n[Stage 1+2] Whitelist + Color Validation (no outlier removal)")
cmd = [
    "python", "clean-gs.py",
    "--ply", PLY,
    "--cameras", CAMERAS,
    "--masked_images", MASKS,
    "--output", f"{OUTPUT_DIR}/stage2.ply",
    "--mode", "none",
    "--color_threshold", "0.40",
    "--workers", "96"
]
subprocess.run(cmd)

# Stage 3: Full pipeline with neighbor removal
print("\n[Stage 3] Full Clean-GS with neighbor removal")
cmd = [
    "python", "clean-gs.py",
    "--ply", PLY,
    "--cameras", CAMERAS,
    "--masked_images", MASKS,
    "--output", f"{OUTPUT_DIR}/stage3.ply",
    "--mode", "neighbor",
    "--color_threshold", "0.40",
    "--workers", "96"
]
subprocess.run(cmd)

print("\n" + "=" * 70)
print("ABLATION COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_DIR}")
