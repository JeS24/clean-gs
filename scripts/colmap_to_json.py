#!/usr/bin/env python3
"""
Convert COLMAP camera data to Clean-GS cameras.json format using pycolmap
"""

import argparse
import json
import numpy as np
import pycolmap
from pathlib import Path


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def convert_colmap_to_cameras_json(colmap_dir, output_json):
    """
    Convert COLMAP binary format to Clean-GS cameras.json

    Args:
        colmap_dir: Path to COLMAP sparse reconstruction
        output_json: Output path for cameras.json
    """
    colmap_dir = Path(colmap_dir)

    print(f"Reading COLMAP reconstruction from {colmap_dir}...")
    reconstruction = pycolmap.Reconstruction(str(colmap_dir))

    print(f"Found {len(reconstruction.cameras)} camera(s) and {len(reconstruction.images)} image(s)")

    # Convert to Clean-GS format
    cameras_json = []

    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]

        # Get camera intrinsics
        width = camera.width
        height = camera.height

        # Get focal length and principal point
        # COLMAP cameras have different param layouts depending on model
        model = str(camera.model)

        if "PINHOLE" in model and len(camera.params) >= 4:
            fx = camera.params[0]
            fy = camera.params[1]
            cx = camera.params[2]
            cy = camera.params[3]
        elif ("SIMPLE" in model or "RADIAL" in model) and len(camera.params) >= 3:
            fx = fy = camera.params[0]
            cx = camera.params[1]
            cy = camera.params[2]
        else:
            # Default: assume first 4 params are fx, fy, cx, cy
            fx = camera.params[0] if len(camera.params) > 0 else width / 2
            fy = camera.params[1] if len(camera.params) > 1 else height / 2
            cx = camera.params[2] if len(camera.params) > 2 else width / 2
            cy = camera.params[3] if len(camera.params) > 3 else height / 2

        # Get camera extrinsics
        # image.cam_from_world() gives world-to-camera transform
        cam_from_world = image.cam_from_world()

        # Get the transformation matrix (3x4)
        w2c_matrix = cam_from_world.matrix()  # 3x4 world-to-camera

        # We need camera-to-world (inverse)
        # Extract R and t from world-to-camera
        R_w2c = w2c_matrix[:, :3]  # 3x3 rotation
        t_w2c = w2c_matrix[:, 3]    # 3x1 translation

        # Invert: R_c2w = R_w2c^T, t_c2w = -R_c2w @ t_w2c
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        # Create camera intrinsic matrix K (3x3)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        camera_data = {
            "image_name": image.name,
            "width": int(width),
            "height": int(height),
            "R": R_c2w.tolist(),  # 3x3 rotation (camera-to-world)
            "T": t_c2w.tolist(),  # 3x1 translation (camera-to-world)
            "K": K.tolist()       # 3x3 intrinsic matrix
        }

        cameras_json.append(camera_data)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(cameras_json, f, indent=2)

    print(f"\n✓ Saved {len(cameras_json)} cameras to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP to cameras.json")
    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="Path to COLMAP sparse reconstruction directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for cameras.json"
    )

    args = parser.parse_args()

    convert_colmap_to_cameras_json(args.colmap_dir, args.output)


if __name__ == "__main__":
    main()
