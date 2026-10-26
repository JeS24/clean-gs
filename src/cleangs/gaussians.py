"""Load Gaussians from a gsplat checkpoint (.pt) or a PLY file."""

from typing import Dict, Tuple

import numpy as np
import torch
from plyfile import PlyData

# Order of the 45 f_rest values in a PLY:
#   gsplat  coefficient-major, as written by gsplat (CleanGS outputs)
#   3dgs    channel-major, as in the original 3DGS code (FlashSplat, GaussianCut)
SH_LAYOUTS = ("gsplat", "3dgs")


def load_gs_from_pt(path: str) -> Tuple[Dict[str, torch.Tensor], int]:
    print(f"Loading Gaussians from gsplat checkpoint: {path}...")
    ckpt = torch.load(path, map_location="cpu")
    splats = ckpt["splats"] if "splats" in ckpt else ckpt
    return splats, len(splats["means"])


def load_gs_from_ply(path: str, sh_layout: str = "gsplat") -> Tuple[Dict[str, torch.Tensor], int]:
    print(f"Loading Gaussians from {path}...")
    vertex = PlyData.read(path)["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    num_gaussians = len(xyz)

    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    available_props = set(vertex.data.dtype.names)
    sh_rest = []
    for i in range(45):
        prop_name = f"f_rest_{i}"
        if prop_name not in available_props:
            break
        sh_rest.append(vertex[prop_name])
    if sh_rest:
        shN = np.stack(sh_rest, axis=1)
    else:
        shN = np.zeros((num_gaussians, 45), dtype=np.float32)

    shN = torch.from_numpy(shN).float()
    if sh_layout == "3dgs":
        if shN.shape[1] == 45:
            shN = shN.view(num_gaussians, 3, 15).permute(0, 2, 1).contiguous()
        else:
            shN = shN.view(num_gaussians, -1, 3)
    else:
        shN = shN.view(num_gaussians, 15, 3)

    opacity = vertex["opacity"]
    scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    rot = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
    )

    splats = {
        "means": torch.from_numpy(xyz).float(),
        "sh0": torch.from_numpy(sh0[:, None, :]).float(),
        "shN": shN,
        "opacities": torch.from_numpy(opacity).float(),
        "scales": torch.from_numpy(scale).float(),
        "quats": torch.from_numpy(rot).float(),
    }
    return splats, num_gaussians


def load_gs(
    path: str, device: str = "cpu", sh_layout: str = "gsplat"
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Load Gaussians onto device as gsplat tensors.

    sh_layout specifies the SH coefficient order for PLY files.
    """
    if sh_layout not in SH_LAYOUTS:
        raise ValueError(f"sh_layout must be one of {SH_LAYOUTS}, got {sh_layout!r}")
    if str(path).endswith(".pt"):
        splats, n = load_gs_from_pt(path)
    else:
        splats, n = load_gs_from_ply(path, sh_layout)
    return {k: v.to(device) for k, v in splats.items()}, n
