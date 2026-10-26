# Clean-GS: Semantic Floater Removal for 3D Gaussian Splatting

Official implementation of **Clean-GS**, a method for removing background clutter and floaters from 3D Gaussian Splatting reconstructions using sparse semantic masks.

## Overview

3D Gaussian Splatting produces high-quality scene reconstructions but generates hundreds of thousands of spurious Gaussians (floaters) scattered throughout the environment. Clean-GS addresses this by combining whitelist-based spatial filtering with color-guided validation and outlier removal to achieve 60-80% model compression while preserving object quality.

## Features

- **Sparse Supervision**: Uses as few as 3 segmentation masks (1% of views)
- **Three-Stage Pipeline**:
  1. Whitelist filtering via projection to masked regions
  2. Depth-buffered color validation
  3. Neighbor-based outlier removal (k-NN)
- **High Compression**: 60-80% reduction in Gaussians with maintained quality
- **Monument Isolation**: Extract clean objects from complex outdoor scenes

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/clean-gs.git
cd clean-gs

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch
- Open3D
- NumPy
- SciPy
- PIL

See [requirements.txt](requirements.txt) for complete dependencies.

## Data Structure

Your data should be organized as follows:

```
data/
├── datasets/
│   └── temple/                    # Scene name
│       ├── images/                # Input images
│       ├── sparse/0/              # COLMAP sparse reconstruction
│       │   ├── cameras.bin
│       │   ├── images.bin
│       │   └── points3D.bin
│       ├── cameras.json           # Camera parameters (generated)
│       └── masks/                 # Semantic segmentation masks (PNG)
│           ├── 000000.png
│           ├── 000001.png
│           └── ...
└── models/
    └── temple/                    # Trained 3DGS model
        └── point_cloud/
            └── iteration_30000/
                └── point_cloud.ply
```

### Creating Masks

Use [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) to create semantic masks for your objects of interest. You only need 3-5 masks for effective floater removal.

## Usage

### Basic Usage

```bash
python clean-gs.py \
    --scene temple \
    --masks_dir data/datasets/temple/masks \
    --input_ply data/models/temple/point_cloud/iteration_30000/point_cloud.ply \
    --output_ply data/models/temple/point_cloud/iteration_30000/clean.ply
```

### With Custom Parameters

```bash
python clean-gs.py \
    --scene temple \
    --masks_dir data/datasets/temple/masks \
    --input_ply data/models/temple/point_cloud/iteration_30000/point_cloud.ply \
    --output_ply clean_output.ply \
    --color_threshold 0.3 \
    --k_neighbors 5 \
    --neighbor_threshold 0.7
```

### Parameters

- `--scene`: Scene name (must match directory in data/datasets/)
- `--masks_dir`: Directory containing semantic masks
- `--input_ply`: Input 3DGS point cloud file
- `--output_ply`: Output cleaned point cloud file
- `--color_threshold`: Color validation threshold (default: 0.3)
- `--k_neighbors`: Number of neighbors for outlier detection (default: 5)
- `--neighbor_threshold`: Outlier removal threshold (default: 0.7)

## Results

### Temple (Tanks and Temples)

- **Original**: 526K Gaussians, 125 MB
- **Clean-GS**: 198K Gaussians, 47 MB (62% reduction)
- **Quality**: Maintained rendering quality with PSNR preserved

### Isha Statue

- **Original**: 1,112K Gaussians, 263.6 MB
- **Clean-GS**: 223K Gaussians, 52.8 MB (80% reduction)

## Pipeline Stages

| Stage | Removed | Remaining | Compression |
|-------|---------|-----------|-------------|
| Original | - | 526K | 0% |
| + Whitelist | 163K | 363K | 31% |
| + Color Validation | 155K | 208K | 60% |
| + Neighbor Removal | 10K | 198K | 62% |

## Quick Start

### 1. Prepare Data

See [data/README.md](data/README.md) for detailed data preparation instructions. Quick summary:

1. **Capture or download multi-view images** (100-300 views)
2. **Run COLMAP** to get sparse reconstruction
   ```bash
   bash scripts/run_colmap_temple.sh
   ```
3. **Train 3D Gaussian Splatting** model
   ```bash
   python train.py -s data/datasets/YOUR_SCENE -m data/models/YOUR_SCENE
   ```
4. **Generate 3-5 semantic masks** using [SAM](https://github.com/facebookresearch/segment-anything)
5. **Convert camera parameters**
   ```bash
   python scripts/colmap_to_json.py --colmap_path data/datasets/YOUR_SCENE/sparse/0 --output_json data/datasets/YOUR_SCENE/cameras.json
   ```

### 2. Run Clean-GS

```bash
python clean-gs.py \
    --scene YOUR_SCENE \
    --masks_dir data/datasets/YOUR_SCENE/masks \
    --input_ply data/models/YOUR_SCENE/point_cloud/iteration_30000/point_cloud.ply \
    --output_ply clean_output.ply
```

## Citation

If you use Clean-GS in your research, please cite:

```bibtex
@article{cleangs2026,
  title={Clean-GS: Semantic Floater Removal for 3D Gaussian Splatting},
  author={Subhankar Mishra},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the base representation
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for mask generation
- [Tanks and Temples](https://www.tanksandtemples.org/) for benchmark datasets

## Contact

For questions or issues, please open an issue on GitHub.
