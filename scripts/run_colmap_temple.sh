#!/bin/bash
# Run COLMAP on Temple dataset to extract camera poses (fully headless)

DATASET_PATH="/home/smlab/projects/clean-gs/data/datasets/temple"
SOURCE_IMAGES="/home/smlab/projects/clean-gs/data/datasets/advanced/Temple"

# Force headless mode
export QT_QPA_PLATFORM=offscreen

echo "=========================================="
echo "Running COLMAP on Temple dataset (headless)"
echo "=========================================="

# Create dataset directory
mkdir -p "$DATASET_PATH"
cd "$DATASET_PATH"

# Copy images if not already there
if [ ! -d "images" ]; then
    echo "Copying images from $SOURCE_IMAGES..."
    mkdir -p images
    cp "$SOURCE_IMAGES"/*.jpg images/
    echo "Copied $(ls images/*.jpg | wc -l) images"
fi

# Clean previous attempts
rm -f database.db
rm -rf sparse
rm -rf sparse_undistorted

# Step 1: Feature extraction
echo ""
echo "Step 1/5: Feature extraction..."
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.num_threads 96

# Step 2: Feature matching
echo ""
echo "Step 2/5: Feature matching..."
colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 0 \
    --SiftMatching.num_threads 96

# Step 3: Sparse reconstruction
echo ""
echo "Step 3/5: Sparse reconstruction..."
mkdir -p sparse
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse

# Step 4: Image undistortion (convert SIMPLE_RADIAL to PINHOLE)
echo ""
echo "Step 4/5: Undistorting images (converting to PINHOLE model)..."
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path sparse_undistorted \
    --output_type COLMAP

# Step 5: Move undistorted results to sparse/0
echo ""
echo "Step 5/5: Replacing sparse reconstruction with undistorted version..."
rm -rf sparse/0
mv sparse_undistorted/sparse sparse/0

# Convert COLMAP binary to JSON for 3DGS
echo ""
echo "Converting COLMAP to JSON format..."
/home/smlab/miniconda3/envs/clean-gs/bin/python /home/smlab/projects/clean-gs/scripts/colmap_to_json.py \
    --colmap_dir sparse/0 \
    --output cameras.json

# Check if successful
if [ -f "sparse/0/cameras.bin" ] && [ -f "cameras.json" ]; then
    echo ""
    echo "✓ COLMAP reconstruction complete!"
    echo ""
    echo "Statistics:"
    ls -lh sparse/0/
    echo ""
    echo "Cameras JSON created: cameras.json"
else
    echo ""
    echo "✗ COLMAP reconstruction failed!"
    exit 1
fi
