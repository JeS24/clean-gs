"""Load cameras from COLMAP or JSON and project world points into images."""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List

from ._vendor.gsplat_colmap.colmap import Parser


@dataclass
class Camera:
    """Store a camera-to-world pose, image details and undistorted intrinsics."""

    image_name: str
    image_path: str
    width: int
    height: int
    R: np.ndarray  # rotation (3x3)
    T: np.ndarray  # camera center in world coordinates
    K: np.ndarray  # intrinsics (3x3)


def load_cameras_from_colmap(
    data_dir: str,
    data_factor: int = 1,
    normalize: bool = True,
    test_every: int = 8,
    split: str = "all",
) -> List[Camera]:
    print(f"Parsing COLMAP data from {data_dir} (split={split})...")
    colmap_parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=normalize,
        test_every=test_every,
    )

    indices = range(len(colmap_parser.image_names))
    if split == "train":
        indices = [x for x in indices if x % test_every != 0]
    elif split == "val":
        indices = [x for x in indices if x % test_every == 0]
    else:
        pass

    print("Building camera records...")
    cameras = []
    for i in indices:
        img_name = colmap_parser.image_names[i]
        cam_id = colmap_parser.camera_ids[i]
        c2w = colmap_parser.camtoworlds[i]
        K = colmap_parser.Ks_dict[cam_id]
        w, h = colmap_parser.imsize_dict[cam_id]
        img_path = os.path.join(data_dir, "images", img_name)
        if not os.path.exists(img_path):
            if data_factor > 1:
                img_path = os.path.join(data_dir, f"images_{data_factor}", img_name)

        cameras.append(
            Camera(
                image_name=img_name,
                width=w,
                height=h,
                R=c2w[:3, :3],
                T=c2w[:3, 3],
                K=K,
                image_path=img_path if os.path.exists(img_path) else None,
            )
        )
    return cameras


def load_cameras_json(cameras_path: str) -> List[Camera]:
    """Load camera records with image_name, width, height, R, T and K from JSON."""
    with open(cameras_path) as f:
        cameras_data = json.load(f)

    cameras = []
    for cam_data in cameras_data:
        cameras.append(
            Camera(
                image_name=cam_data["image_name"],
                width=cam_data["width"],
                height=cam_data["height"],
                R=np.array(cam_data["R"]),
                T=np.array(cam_data["T"]),
                K=np.array(cam_data["K"]),
            )
        )
    return cameras
