#!/bin/bash

# Run actual timed experiments for paper

PYTHON="/home/smlab/miniconda3/envs/clean-gs/bin/python"
PLY_TEMPLE="data/models/temple/point_cloud/iteration_30000/point_cloud.ply"
CAM_TEMPLE="data/datasets/temple/cameras.json"
MASKS_TEMPLE="data/masks/temple"

PLY_ISHA="data/models/isha/point_cloud/iteration_30000/point_cloud.ply"
CAM_ISHA="data/datasets/isha/cameras.json"
MASKS_ISHA="data/masks/isha"

echo "=== Running Timed Experiments for Paper ==="
echo ""

# Temple - Basic (no outlier removal)
echo "1. Temple - Basic (whitelist + color only)"
/usr/bin/time -f "Time: %E" $PYTHON clean-gs.py \
  --ply "$PLY_TEMPLE" \
  --cameras "$CAM_TEMPLE" \
  --masked_images "$MASKS_TEMPLE" \
  --output results/paper_temple_basic.ply \
  --mode none \
  --workers 96 2>&1 | tee results/timing_temple_basic.log

echo ""
echo "2. Temple - Clean-GS (neighbor)"
/usr/bin/time -f "Time: %E" $PYTHON clean-gs.py \
  --ply "$PLY_TEMPLE" \
  --cameras "$CAM_TEMPLE" \
  --masked_images "$MASKS_TEMPLE" \
  --output results/paper_temple_neighbor.ply \
  --mode neighbor \
  --workers 96 2>&1 | tee results/timing_temple_neighbor.log

echo ""
echo "3. Isha - Clean-GS (neighbor)"
/usr/bin/time -f "Time: %E" $PYTHON clean-gs.py \
  --ply "$PLY_ISHA" \
  --cameras "$CAM_ISHA" \
  --masked_images "$MASKS_ISHA" \
  --output results/paper_isha_neighbor.ply \
  --mode neighbor \
  --workers 96 2>&1 | tee results/timing_isha_neighbor.log

echo ""
echo "=== Timing Summary ==="
grep "Time:" results/timing_*.log
echo ""
echo "=== File Sizes ==="
ls -lh results/paper_*.ply
