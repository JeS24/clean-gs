#!/usr/bin/env python3
"""Stage 3: evaluate an exported Gaussian asset at the original and wider framing."""

import os
import numpy as np
import cv2
import torch
import tqdm
import json
from pathlib import Path
from datetime import datetime

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .camera_utils import load_cameras_from_colmap
from .gaussians import SH_LAYOUTS, load_gs
from gsplat.rendering import rasterization


def apply_zoom_2d(img, zoom_factor):
    """Shrink the image about its center and pad to the original size."""
    if zoom_factor <= 1.0:
        return img
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w / zoom_factor)), max(1, int(h / zoom_factor))
    interp = cv2.INTER_LINEAR if len(img.shape) > 2 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    if len(img.shape) > 2:
        padded = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    else:
        padded = np.zeros((h, w), dtype=img.dtype)
    padded[top : top + new_h, left : left + new_w] = resized
    return padded


def apply_zoom_2d_faceon(img, zoom_factor, cx, cy):
    """Shrink the image about the principal point to match a shorter focal length."""
    if zoom_factor <= 1.0:
        return img
    h, w = img.shape[:2]
    s = 1.0 / zoom_factor
    M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], dtype=np.float32)
    interp = cv2.INTER_LINEAR if len(img.shape) > 2 else cv2.INTER_NEAREST
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderValue=0)


def get_masked_metrics(psnr_m, ssim_m, lpips_m, render, gt, mask=None, device="cuda"):
    """Compute masked PSNR, SSIM and LPIPS for images shaped [1, 3, H, W].

    Return three None values if no mask is supplied.
    """
    if mask is None:
        return None, None, None

    mask = mask.to(device)

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
        return 0.0, 0.0, 1.0

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


def render_frame(splats, viewmat, K, width, height, sh_degree, device, near=0.01):
    """Render RGB [H, W, 3], accumulated depth [H, W] and alpha [H, W]."""
    rendered, alphas, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
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


def compute_silhouette_leakage(alpha_map, fg_mask, tau=0.1):
    """Measure rendered opacity outside the target mask.

    Return mean alpha (l_leak, reported in the paper) and the fraction of pixels
    outside the mask with alpha > tau. Both values are NaN if the mask is missing
    or covers the whole image.
    """
    if fg_mask is None:
        return float("nan"), float("nan")

    bg = 1.0 - fg_mask
    bg_sum = bg.sum().item()
    if bg_sum < 1:
        return float("nan"), float("nan")

    l_leak = ((alpha_map * bg).sum() / (bg_sum + 1e-10)).item()
    l_leak_tau = ((alpha_map.gt(tau).float() * bg).sum() / (bg_sum + 1e-10)).item()
    return l_leak, l_leak_tau


def _apply_jitter(c2w, rot_deg, trans_noise, device):
    """Rotate a camera-to-world pose by up to rot_deg and translate it by trans_noise."""
    axis = torch.randn(3, device=device)
    axis = axis / (axis.norm() + 1e-8)
    angle = (torch.rand(1, device=device).item() * 2 - 1) * rot_deg * (np.pi / 180.0)
    K_skew = torch.zeros(3, 3, device=device)
    K_skew[0, 1] = -axis[2]
    K_skew[0, 2] = axis[1]
    K_skew[1, 0] = axis[2]
    K_skew[1, 2] = -axis[0]
    K_skew[2, 0] = -axis[1]
    K_skew[2, 1] = axis[0]
    R_jit = (
        torch.eye(3, device=device)
        + np.sin(angle) * K_skew
        + (1 - np.cos(angle)) * (K_skew @ K_skew)
    )

    c2w_new = c2w.clone()
    c2w_new[:3, :3] = R_jit @ c2w[:3, :3]

    t_noise = torch.randn(3, device=device)
    t_noise = t_noise / (t_noise.norm() + 1e-8) * trans_noise
    c2w_new[:3, 3] = c2w[:3, 3] + t_noise

    return c2w_new


def compute_depth_stability_step(
    depth_base, c2w_base, K_base, depth_jit, c2w_jit, K_jit, H, W, eps_abs, device
):
    """Measure depth agreement after reprojecting the jittered view into the base view.

    Return the fraction of valid projections whose depths differ by less than eps_abs.
    """
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pts2d = torch.stack(
        [xs.reshape(-1), ys.reshape(-1), torch.ones(H * W, device=device)], dim=1
    )

    # Unproject the jittered view and reproject into the base view
    K_jit_inv = torch.linalg.inv(K_jit)
    rays_cam = (K_jit_inv @ pts2d.T).T

    d_flat = depth_jit.reshape(-1)
    valid = d_flat > 0.0
    pts3d_cam_jit = rays_cam * d_flat.unsqueeze(1)

    R_jit = c2w_jit[:3, :3]
    t_jit = c2w_jit[:3, 3]
    pts3d_world = (R_jit @ pts3d_cam_jit.T).T + t_jit

    w2c_base = torch.linalg.inv(c2w_base)
    pts3d_cam_base = (w2c_base[:3, :3] @ pts3d_world.T).T + w2c_base[:3, 3]
    z_in_base = pts3d_cam_base[:, 2]

    proj = (K_base @ pts3d_cam_base.T).T
    u_b = proj[:, 0] / (proj[:, 2] + 1e-8)
    v_b = proj[:, 1] / (proj[:, 2] + 1e-8)

    valid = (
        valid
        & (z_in_base > 0)
        & (u_b >= 0)
        & (u_b < W - 1)
        & (v_b >= 0)
        & (v_b < H - 1)
    )
    if valid.sum() == 0:
        return 0.0

    u_v = u_b[valid]
    v_v = v_b[valid]
    z_exp = z_in_base[valid]

    # Bilinear lookup of the base depth
    u0 = u_v.long().clamp(0, W - 2)
    v0 = v_v.long().clamp(0, H - 2)
    du = u_v - u0.float()
    dv = v_v - v0.float()
    d00 = depth_base[v0, u0]
    d10 = depth_base[v0, u0 + 1]
    d01 = depth_base[v0 + 1, u0]
    d11 = depth_base[v0 + 1, u0 + 1]
    d_sampled = (
        d00 * (1 - du) * (1 - dv)
        + d10 * du * (1 - dv)
        + d01 * (1 - du) * dv
        + d11 * du * dv
    )

    agree = (torch.abs(d_sampled - z_exp) < eps_abs).float()
    return agree.mean().item()


def compute_background_variance(renders_list, bg_mask):
    """Average color variance across renders over pixels outside the target mask."""
    if bg_mask is None or bg_mask.sum() < 1:
        return None
    stack = np.stack(renders_list, axis=0)
    var_map = stack.var(axis=0)
    var_gray = var_map.mean(axis=2)
    bg_var = float((var_gray * bg_mask).sum() / (bg_mask.sum() + 1e-10))
    return bg_var


def load_clip_model(model_name="openai/clip-vit-base-patch32", device="cuda"):
    """Load a Hugging Face CLIP model and processor using transformers."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def compute_clip_iqa(
    render_np,
    clip_model,
    clip_processor,
    device,
    pos_prompt="Good photo.",
    neg_prompt="Bad photo.",
):
    """Score image quality with a softmax over positive and negative CLIP similarities."""
    from PIL import Image

    pil_img = Image.fromarray(render_np)
    inputs = clip_processor(
        text=[pos_prompt, neg_prompt],
        images=[pil_img],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        out = clip_model(**inputs)
        img_emb = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        text_emb = torch.nn.functional.normalize(out.text_embeds, dim=-1)

    s_pos = (img_emb @ text_emb[0:1].T).squeeze().item()
    s_neg = (img_emb @ text_emb[1:2].T).squeeze().item()
    score = np.exp(s_pos) / (np.exp(s_pos) + np.exp(s_neg) + 1e-10)
    return float(score)


def evaluate_gs_more(
    ply_path,
    data_dir,
    output_dir,
    masks_dir=None,
    data_factor=1,
    test_every=8,
    sh_degree=3,
    zoom_factor=1.0,
    zoom_mode="optical",
    device="cuda",
    scene_name=None,
    save_renders=True,
    normalize_cameras=True,
    sh_layout="gsplat",
    jitter_k=5,
    jitter_rot_deg=2.0,
    jitter_trans_frac=0.02,
    depth_eps_rel=0.05,
    leakage_tau=0.1,
    clip_iqa=False,
    clip_model_name="openai/clip-vit-base-patch32",
    clip_pos_prompt="Good photo.",
    clip_neg_prompt="Bad photo.",
):
    start_time = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scene_name is None:
        p = Path(data_dir)
        scene_name = p.parent.name if p.name in ("colmap", "sparse") else p.name

    print(f"\n{'=' * 70}")
    print(f"CleanGS evaluation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scene:              {scene_name}")
    print(f"PLY Path:           {ply_path}")
    print(f"Data Dir:           {data_dir}")
    print(f"Masks Dir:          {masks_dir}")
    print(f"Zoom Factor:        {zoom_factor} ({zoom_mode})")
    print(
        f"Jitter K:           {jitter_k}  rot={jitter_rot_deg}deg  trans_frac={jitter_trans_frac}"
    )
    print(f"Depth eps (rel):    {depth_eps_rel}")
    print(f"Leakage tau:        {leakage_tau}")
    print(f"CLIP-IQA:           {clip_iqa}")
    print(f"Output Dir:         {output_dir}")
    print(f"{'=' * 70}\n")

    # 1. Load Gaussians
    splats, num_gaussians = load_gs(ply_path, device=device, sh_layout=sh_layout)
    print(f"Loaded {num_gaussians:,} Gaussians.")

    # 2. Held-out cameras
    cameras = load_cameras_from_colmap(
        data_dir,
        data_factor=data_factor,
        normalize=normalize_cameras,
        test_every=test_every,
        split="val",
    )
    if len(cameras) == 0:
        print("No held-out cameras found; evaluating all cameras.")
        cameras = load_cameras_from_colmap(
            data_dir,
            data_factor=data_factor,
            normalize=normalize_cameras,
            test_every=test_every,
            split="all",
        )
    print(f"Evaluating on {len(cameras)} views.")

    # Scene scale for the translation jitter
    cam_positions = np.stack([c.T for c in cameras], axis=0)
    scene_scale = float(np.std(cam_positions)) + 1e-3

    # 3. Metrics setup
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    clip_model_obj = clip_proc_obj = None
    if clip_iqa:
        print("Loading CLIP model…")
        try:
            clip_model_obj, clip_proc_obj = load_clip_model(clip_model_name, device)
            print(f"  Loaded {clip_model_name}")
        except Exception as e:
            print(f"  WARNING: CLIP-IQA disabled — {e}")
            clip_iqa = False

    render_save_dir = output_dir / "renders"
    if save_renders:
        render_save_dir.mkdir(parents=True, exist_ok=True)

    metrics_results = []

    # Masks are matched by case-insensitive file stem
    mask_index = {}
    if masks_dir:
        for p in Path(masks_dir).iterdir():
            if p.is_file():
                mask_index.setdefault(p.stem.lower(), p)

    # 4. Per-view evaluation loop
    for i, cam in enumerate(tqdm.tqdm(cameras, desc="Evaluating views")):
        if cam.image_name is None or not os.path.exists(cam.image_path):
            print("Skipping view: image name or image file is missing.")
            continue

        gt_img = cv2.imread(cam.image_path)
        if gt_img is None:
            print(f"  Warning: could not read reference image: {cam.image_path}")
            continue
        if gt_img.shape[0] != cam.height or gt_img.shape[1] != cam.width:
            gt_img = cv2.resize(gt_img, (cam.width, cam.height))
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        mask_np = None
        if masks_dir:
            mp = mask_index.get(Path(cam.image_name).stem.lower())
            if mp is not None:
                mask_np = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask_np is not None:
                if mask_np.shape[0] != cam.height or mask_np.shape[1] != cam.width:
                    mask_np = cv2.resize(
                        mask_np,
                        (cam.width, cam.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_np = (mask_np > 127).astype(np.float32)

        R = torch.from_numpy(cam.R).float().to(device)
        T = torch.from_numpy(cam.T).float().to(device)
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        K_t = torch.from_numpy(cam.K).float().to(device)
        K = K_t.unsqueeze(0)

        with torch.no_grad():
            rgb_base, depth_base, alpha_base = render_frame(
                splats, viewmat, K, cam.width, cam.height, sh_degree, device
            )

        # A. Masked photometric metrics
        gt_tensor = torch.from_numpy(gt_rgb).float().to(device) / 255.0
        gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0)
        render_tensor = rgb_base.permute(2, 0, 1).unsqueeze(0)

        mask_tensor = None
        if mask_np is not None:
            mask_tensor = (
                torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
            )

        psnr_val, ssim_val, lpips_val = get_masked_metrics(
            psnr_metric,
            ssim_metric,
            lpips_metric,
            render_tensor,
            gt_tensor,
            mask_tensor,
            device,
        )
        if psnr_val is None:
            continue

        view_results = {
            "image_name": cam.image_name,
            "standard": {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val},
        }

        # B. Silhouette leakage
        fg_mask_t = (
            mask_tensor.squeeze(0).squeeze(0) if mask_tensor is not None else None
        )
        l_leak, l_leak_tau = compute_silhouette_leakage(
            alpha_base, fg_mask_t, tau=leakage_tau
        )
        view_results["leakage"] = {"l_leak": l_leak, "l_leak_tau": l_leak_tau}

        # C. Depth stability and background variance under small pose jitter
        trans_noise = scene_scale * jitter_trans_frac
        # Depth tolerance relative to the median rendered depth
        med_depth = (
            float(torch.median(depth_base[depth_base > 0]).item())
            if (depth_base > 0).any()
            else 1.0
        )
        eps_abs = depth_eps_rel * med_depth

        jit_stability_acc = 0.0
        renders_for_var = [rgb_base.cpu().numpy()]

        torch.manual_seed(42 + i)
        for k in range(jitter_k):
            c2w_jit = _apply_jitter(c2w, jitter_rot_deg, trans_noise, device)
            vm_jit = torch.linalg.inv(c2w_jit).unsqueeze(0)

            with torch.no_grad():
                rgb_jit, depth_jit, _ = render_frame(
                    splats, vm_jit, K, cam.width, cam.height, sh_degree, device
                )

            stab = compute_depth_stability_step(
                depth_base,
                c2w,
                K_t,
                depth_jit,
                c2w_jit,
                K_t,
                cam.height,
                cam.width,
                eps_abs,
                device,
            )
            jit_stability_acc += stab
            renders_for_var.append(rgb_jit.cpu().numpy())

        s_depth = jit_stability_acc / max(1, jitter_k)

        bg_mask_np = (1.0 - mask_np) if mask_np is not None else None
        var_bg = compute_background_variance(renders_for_var, bg_mask_np)

        view_results["stability"] = {
            "s_depth": s_depth,
            "var_bg": var_bg,
            "jitter_k": jitter_k,
            "jitter_rot_deg": jitter_rot_deg,
            "eps_abs": eps_abs,
        }

        # D. Widened framing ("faceon": focal length divided by zoom_factor)
        if zoom_factor > 1.0:
            if zoom_mode == "optical":
                backward = R[:, 2]
                d = backward.dot(T)
                T_zoom = T + backward * d * (zoom_factor - 1)
                c2w_zoom = torch.eye(4, device=device)
                c2w_zoom[:3, :3] = R
                c2w_zoom[:3, 3] = T_zoom
                vm_zoom = torch.linalg.inv(c2w_zoom).unsqueeze(0)
                K_zoom = K
                K_t_zoom = K_t
            else:  # faceon
                K_zoom = K.clone()
                K_zoom[0, 0, 0] /= zoom_factor
                K_zoom[0, 1, 1] /= zoom_factor
                vm_zoom = viewmat
                c2w_zoom = c2w
                K_t_zoom = K_zoom.squeeze(0)

            with torch.no_grad():
                rgb_zoom, depth_zoom, alpha_zoom = render_frame(
                    splats, vm_zoom, K_zoom, cam.width, cam.height, sh_degree, device
                )

            if zoom_mode == "faceon":
                cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()
                gt_rgb_zoom = apply_zoom_2d_faceon(gt_rgb, zoom_factor, cx, cy)
                mask_np_zoom = (
                    apply_zoom_2d_faceon(mask_np, zoom_factor, cx, cy)
                    if mask_np is not None
                    else None
                )
            else:
                gt_rgb_zoom = apply_zoom_2d(gt_rgb, zoom_factor)
                mask_np_zoom = (
                    apply_zoom_2d(mask_np, zoom_factor) if mask_np is not None else None
                )

            gt_t_zoom = torch.from_numpy(gt_rgb_zoom).float().to(device) / 255.0
            gt_t_zoom = gt_t_zoom.permute(2, 0, 1).unsqueeze(0)
            rz_tensor = rgb_zoom.permute(2, 0, 1).unsqueeze(0)
            mz_tensor = (
                torch.from_numpy(mask_np_zoom)
                .float()
                .to(device)
                .unsqueeze(0)
                .unsqueeze(0)
                if mask_np_zoom is not None
                else None
            )

            psnr_z, ssim_z, lpips_z = get_masked_metrics(
                psnr_metric,
                ssim_metric,
                lpips_metric,
                rz_tensor,
                gt_t_zoom,
                mz_tensor,
                device,
            )
            view_results["zoomed"] = {"psnr": psnr_z, "ssim": ssim_z, "lpips": lpips_z}

            fg_mask_z = (
                torch.from_numpy(mask_np_zoom).float().to(device)
                if mask_np_zoom is not None
                else None
            )
            l_leak_z, l_leak_tau_z = compute_silhouette_leakage(
                alpha_zoom, fg_mask_z, tau=leakage_tau
            )

            med_depth_z = (
                float(torch.median(depth_zoom[depth_zoom > 0]).item())
                if (depth_zoom > 0).any()
                else 1.0
            )
            eps_abs_z = depth_eps_rel * med_depth_z
            jit_stab_z_acc = 0.0
            renders_z_for_var = [rgb_zoom.cpu().numpy()]
            torch.manual_seed(42 + i + 1000)
            for k in range(jitter_k):
                c2w_jit_z = _apply_jitter(c2w_zoom, jitter_rot_deg, trans_noise, device)
                vm_jit_z = torch.linalg.inv(c2w_jit_z).unsqueeze(0)
                with torch.no_grad():
                    rgb_jit_z, depth_jit_z, _ = render_frame(
                        splats, vm_jit_z, K_zoom, cam.width, cam.height, sh_degree, device
                    )
                stab_z = compute_depth_stability_step(
                    depth_zoom,
                    c2w_zoom,
                    K_t_zoom,
                    depth_jit_z,
                    c2w_jit_z,
                    K_t_zoom,
                    cam.height,
                    cam.width,
                    eps_abs_z,
                    device,
                )
                jit_stab_z_acc += stab_z
                renders_z_for_var.append(rgb_jit_z.cpu().numpy())

            s_depth_z = jit_stab_z_acc / max(1, jitter_k)
            bg_mask_np_zoom = (1.0 - mask_np_zoom) if mask_np_zoom is not None else None
            var_bg_z = compute_background_variance(renders_z_for_var, bg_mask_np_zoom)

            clip_score_z = None
            if clip_iqa and clip_model_obj is not None:
                render_zoom_uint8 = (
                    (rgb_zoom.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                clip_score_z = compute_clip_iqa(
                    render_zoom_uint8,
                    clip_model_obj,
                    clip_proc_obj,
                    device,
                    clip_pos_prompt,
                    clip_neg_prompt,
                )

            view_results["zoomed"].update({
                "l_leak_z": l_leak_z,
                "l_leak_tau_z": l_leak_tau_z,
                "s_depth_z": s_depth_z,
                "var_bg_z": var_bg_z,
                "clip_iqa_z": clip_score_z,
            })

            if save_renders:
                stem_z = Path(cam.image_name).stem
                render_zoom_np = (rgb_zoom.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    str(render_save_dir / f"{stem_z}_zoom_render.png"),
                    cv2.cvtColor(render_zoom_np, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(render_save_dir / f"{stem_z}_zoom_gt.png"),
                    cv2.cvtColor(gt_rgb_zoom, cv2.COLOR_RGB2BGR),
                )
                if mask_np_zoom is not None:
                    gt_zoom_masked = (gt_rgb_zoom * mask_np_zoom[:, :, None]).astype(
                        np.uint8
                    )
                    r_zoom_masked = (render_zoom_np * mask_np_zoom[:, :, None]).astype(
                        np.uint8
                    )
                    cv2.imwrite(
                        str(render_save_dir / f"{stem_z}_zoom_gt_masked.png"),
                        cv2.cvtColor(gt_zoom_masked, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(render_save_dir / f"{stem_z}_zoom_render_masked.png"),
                        cv2.cvtColor(r_zoom_masked, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        str(render_save_dir / f"{stem_z}_zoom_mask.png"),
                        (mask_np_zoom * 255).astype(np.uint8),
                    )

        # E. CLIP-IQA
        if clip_iqa and clip_model_obj is not None:
            render_uint8 = (rgb_base.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            clip_score = compute_clip_iqa(
                render_uint8,
                clip_model_obj,
                clip_proc_obj,
                device,
                clip_pos_prompt,
                clip_neg_prompt,
            )
            view_results["clip_iqa"] = clip_score

        # F. Save renders
        if save_renders:
            stem_name = Path(cam.image_name).stem
            render_np = (rgb_base.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                str(render_save_dir / f"{stem_name}_render.png"),
                cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(render_save_dir / f"{stem_name}_gt.png"),
                cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR),
            )
            alpha_np = (alpha_base.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(render_save_dir / f"{stem_name}_alpha.png"), alpha_np)
            # Depth visualization: near is bright
            d_np = depth_base.cpu().numpy()
            valid = d_np > 0
            d_p99 = float(np.percentile(d_np[valid], 99)) if valid.any() else 1.0
            d_norm = np.zeros_like(d_np)
            d_norm[valid] = np.clip(1.0 - d_np[valid] / (d_p99 + 1e-3), 0.0, 1.0)
            d_vis = cv2.applyColorMap(
                (d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            cv2.imwrite(str(render_save_dir / f"{stem_name}_depth.png"), d_vis)
            if mask_np is not None:
                gt_masked = (gt_rgb * mask_np[:, :, None]).astype(np.uint8)
                r_masked = (render_np * mask_np[:, :, None]).astype(np.uint8)
                cv2.imwrite(
                    str(render_save_dir / f"{stem_name}_gt_masked.png"),
                    cv2.cvtColor(gt_masked, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(render_save_dir / f"{stem_name}_render_masked.png"),
                    cv2.cvtColor(r_masked, cv2.COLOR_RGB2BGR),
                )

        metrics_results.append(view_results)

    # 5. Aggregate and print
    if not metrics_results:
        print("No views could be evaluated with the supplied images and masks.")
        return

    def _mean(key_path, results):
        """Average a nested metric such as ('standard', 'psnr'), skipping None and NaN."""
        vals = []
        for m in results:
            obj = m
            for k in key_path:
                obj = obj.get(k)
                if obj is None:
                    break
            if obj is not None and not (isinstance(obj, float) and np.isnan(obj)):
                vals.append(obj)
        return float(np.mean(vals)) if vals else float("nan")

    avg_std = {
        k: _mean(("standard", k), metrics_results) for k in ["psnr", "ssim", "lpips"]
    }

    avg_leak = {
        "l_leak": _mean(("leakage", "l_leak"), metrics_results),
        "l_leak_tau": _mean(("leakage", "l_leak_tau"), metrics_results),
    }
    avg_stab = {
        "s_depth": _mean(("stability", "s_depth"), metrics_results),
        "var_bg": _mean(("stability", "var_bg"), metrics_results),
    }

    print(f"\n{'=' * 70}")
    print(f"Averages for {scene_name}")
    print(
        f"  [Standard masked]     PSNR={avg_std['psnr']:.4f}  SSIM={avg_std['ssim']:.4f}  LPIPS={avg_std['lpips']:.4f}"
    )
    print(
        f"  [Silhouette leakage]  L_leak={avg_leak['l_leak']:.4f}  L_leak_tau={avg_leak['l_leak_tau']:.4f}  (tau={leakage_tau})"
    )
    print(
        f"  [Depth stability]     S_depth={avg_stab['s_depth']:.4f}  (eps_rel={depth_eps_rel}, K={jitter_k})"
    )
    print(f"  [BG variance]         Var_bg={avg_stab['var_bg']:.6f}")

    if zoom_factor > 1.0:
        avg_zoom = {
            k: _mean(("zoomed", k), metrics_results) for k in ["psnr", "ssim", "lpips"]
        }
        avg_zoom.update({
            "l_leak_z":     _mean(("zoomed", "l_leak_z"),     metrics_results),
            "l_leak_tau_z": _mean(("zoomed", "l_leak_tau_z"), metrics_results),
            "s_depth_z":    _mean(("zoomed", "s_depth_z"),    metrics_results),
            "var_bg_z":     _mean(("zoomed", "var_bg_z"),     metrics_results),
            "clip_iqa_z":   _mean(("zoomed", "clip_iqa_z"),   metrics_results),
        })
        print(
            f"  [Zoomed {zoom_factor}x {zoom_mode}]   PSNR={avg_zoom['psnr']:.4f}  SSIM={avg_zoom['ssim']:.4f}  LPIPS={avg_zoom['lpips']:.4f}"
        )
        print(
            f"  [Zoom leakage]        L_leak_z={avg_zoom['l_leak_z']:.4f}  L_leak_tau_z={avg_zoom['l_leak_tau_z']:.4f}  (tau={leakage_tau})"
        )
        print(
            f"  [Zoom depth stab]     S_depth_z={avg_zoom['s_depth_z']:.4f}  (eps_rel={depth_eps_rel}, K={jitter_k})"
        )
        print(f"  [Zoom BG variance]    Var_bg_z={avg_zoom['var_bg_z']:.6f}")
        if not np.isnan(avg_zoom["clip_iqa_z"]):
            print(f"  [Zoom CLIP-IQA]       {avg_zoom['clip_iqa_z']:.4f}")
    else:
        avg_zoom = None

    avg_clip = _mean(("clip_iqa",), metrics_results) if clip_iqa else None
    if avg_clip is not None and not np.isnan(avg_clip):
        print(f"  [CLIP-IQA]            {avg_clip:.4f}")
    print(f"{'=' * 70}\n")

    # 6. Save JSON
    results_json = {
        "scene_name": scene_name,
        "ply_path": str(ply_path),
        "data_dir": str(data_dir),
        "eval_params": {
            "zoom_factor": zoom_factor,
            "zoom_mode": zoom_mode,
            "jitter_k": jitter_k,
            "jitter_rot_deg": jitter_rot_deg,
            "jitter_trans_frac": jitter_trans_frac,
            "depth_eps_rel": depth_eps_rel,
            "leakage_tau": leakage_tau,
            "clip_iqa": clip_iqa,
        },
        "averages": {
            "standard": avg_std,
            "zoomed": avg_zoom,
            "leakage": avg_leak,
            "stability": avg_stab,
            "clip_iqa": avg_clip,
        },
        "per_view": metrics_results,
    }

    json_path = output_dir / f"{scene_name}_metrics_more.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)

    print(f"Results saved to {json_path}")
    print(f"Total Time: {datetime.now() - start_time}")
    print(f"{'=' * 70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate masked image quality, silhouette leakage, depth stability "
        "and background variance, with optional CLIP-IQA."
    )
    parser.add_argument("--ply", required=True, help="Input PLY or .pt file")
    parser.add_argument("--data_dir", required=True, help="COLMAP data directory")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for metrics and renders"
    )
    parser.add_argument("--masks", help="Directory containing binary foreground masks")
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=1.0,
        help="Factor by which to widen the framing (1 keeps the original framing)",
    )
    parser.add_argument("--zoom_mode", choices=["optical", "faceon"], default="optical")
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_save_renders", action="store_false", dest="save_renders")
    parser.add_argument(
        "--no_normalize_cameras",
        action="store_true",
        help="Skip camera normalization for assets in the original 3DGS coordinate system",
    )
    parser.add_argument(
        "--sh_layout",
        choices=SH_LAYOUTS,
        default="gsplat",
        help="SH coefficient order in a PLY: gsplat for CleanGS outputs, or "
        "3dgs for original 3DGS, FlashSplat and GaussianCut exports",
    )
    parser.add_argument(
        "--jitter_k",
        type=int,
        default=5,
        help="Number of perturbed camera poses per view for stability and variance metrics",
    )
    parser.add_argument(
        "--jitter_rot_deg",
        type=float,
        default=2.0,
        help="Maximum rotation in degrees for each perturbed camera pose",
    )
    parser.add_argument(
        "--jitter_trans_frac",
        type=float,
        default=0.02,
        help="Translation distance as a fraction of the estimated scene scale",
    )
    parser.add_argument(
        "--depth_eps_rel",
        type=float,
        default=0.05,
        help="Depth agreement tolerance as a fraction of the median positive depth in each view",
    )
    parser.add_argument(
        "--leakage_tau",
        type=float,
        default=0.1,
        help="Opacity threshold for counting leakage pixels outside the target mask",
    )
    parser.add_argument(
        "--clip_iqa",
        action="store_true",
        help="Compute CLIP-IQA scores (install transformers and accelerate first)",
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name for CLIP-IQA",
    )
    parser.add_argument("--clip_pos_prompt", default="Good photo.")
    parser.add_argument("--clip_neg_prompt", default="Bad photo.")

    args = parser.parse_args()

    evaluate_gs_more(
        ply_path=args.ply,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        masks_dir=args.masks,
        data_factor=args.data_factor,
        test_every=args.test_every,
        sh_degree=args.sh_degree,
        zoom_factor=args.zoom_factor,
        zoom_mode=args.zoom_mode,
        device=args.device,
        save_renders=args.save_renders,
        normalize_cameras=not args.no_normalize_cameras,
        sh_layout=args.sh_layout,
        jitter_k=args.jitter_k,
        jitter_rot_deg=args.jitter_rot_deg,
        jitter_trans_frac=args.jitter_trans_frac,
        depth_eps_rel=args.depth_eps_rel,
        leakage_tau=args.leakage_tau,
        clip_iqa=args.clip_iqa,
        clip_model_name=args.clip_model,
        clip_pos_prompt=args.clip_pos_prompt,
        clip_neg_prompt=args.clip_neg_prompt,
    )


if __name__ == "__main__":
    main()
