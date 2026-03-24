#!/usr/bin/env python3
"""
Cumulative Rendering Script for HunyuanWorld-Mirror

This script takes the output directory from infer.py and creates cumulative renders:
- Load splats for view 0, render for view 0
- Load splats for view 1, concatenate to view 0 splats, render for views 0 and 1
- And so on...

Usage:
    python cumulative_render.py /path/to/infer_output /path/to/render_output
"""

import argparse
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from src.models.models.rasterization import GaussianSplatRenderer
import torch
from PIL import Image
from plyfile import PlyData

from src.models.models.worldmirror import WorldMirror
from src.utils.save_utils import save_image_png
from src.models.utils.sh_utils import RGB2SH


def load_splats_from_ply(ply_path):
    """
    Load Gaussian splats from PLY file and convert to tensor format expected by rasterizer.

    Args:
        ply_path: Path to PLY file

    Returns:
        Dictionary with splat tensors in format expected by rasterize_batches
    """
    # Load PLY data
    plydata = PlyData.read(ply_path)
    vert = plydata["vertex"]

    # Extract data and convert back to tensor format
    means = torch.tensor(np.column_stack([vert["x"], vert["y"], vert["z"]]), dtype=torch.float32)
    colors = torch.tensor(np.column_stack([vert["f_dc_0"], vert["f_dc_1"], vert["f_dc_2"]]), dtype=torch.float32)
    opacities = torch.tensor(vert["opacity"], dtype=torch.float32)
    scales = torch.exp(torch.tensor(np.column_stack([vert["scale_0"], vert["scale_1"], vert["scale_2"]]), dtype=torch.float32))
    quats = torch.tensor(np.column_stack([vert["rot_0"], vert["rot_1"], vert["rot_2"], vert["rot_3"]]), dtype=torch.float32)

    # Convert RGB colors back to SH coefficients (as stored in prepare_splats)
    sh = RGB2SH(colors).unsqueeze(1)  # [N, 1, 3]

    return {
        "means": means.unsqueeze(0),      # [1, N, 3]
        "quats": quats.unsqueeze(0),      # [1, N, 4]
        "scales": scales.unsqueeze(0),    # [1, N, 3]
        "opacities": opacities.unsqueeze(0), # [1, N]
        "sh": sh.unsqueeze(0)             # [1, N, 1, 3]
    }


def load_splats_from_zip(zip_path):
    """
    Load Gaussian splats from zip file containing numpy arrays and convert to tensor format.

    Args:
        zip_path: Path to zip file containing splat tensors

    Returns:
        Dictionary with splat tensors in format expected by rasterize_batches
    """
    splats_dict = {}
    mask = None

    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in z.namelist():
            if file_name.endswith(".npy"):
                with z.open(file_name) as f:
                    arr = np.load(f)
                    key = file_name[:-4]  # Remove .npy extension
                    if key == "mask":
                        mask = arr.astype(bool)
                    else:
                        splats_dict[key] = torch.tensor(arr, dtype=torch.float32)

    # Apply mask if present
    if mask is not None:
        for key in splats_dict:
            if key in ["means", "scales", "quats", "opacities", "sh", "colors"]:
                # Flatten to [H*W, D], apply mask, then reshape back to [1, N, D]
                original_shape = splats_dict[key].shape
                flat = splats_dict[key].reshape(-1, original_shape[-1])
                masked = flat[mask.flatten()]
                splats_dict[key] = masked.unsqueeze(0)  # [1, N, D]
    else:
        # No mask, just flatten and add batch dim
        for key in splats_dict:
            if key in ["means", "scales", "quats", "opacities", "sh", "colors"]:
                flat = splats_dict[key].reshape(-1, splats_dict[key].shape[-1])
                splats_dict[key] = flat.unsqueeze(0)  # [1, N, D]
    
    return splats_dict


def main():
    parser = argparse.ArgumentParser(description="Cumulative rendering for HunyuanWorld-Mirror")
    parser.add_argument("--infer_dir", type=str, help="Path to infer.py output directory")
    parser.add_argument("--render_dir", type=str, help="Path to save cumulative renders")
    parser.add_argument("--height", type=int, help="Height of the rendered images")
    parser.add_argument("--width", type=int, help="Width of the rendered images")
    args = parser.parse_args()

    infer_dir = Path(args.infer_dir)
    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading from: {infer_dir}")
    print(f"📂 Saving to: {render_dir}")

    # Load camera parameters
    cam_params_path = infer_dir / "camera_params.json"
    if not cam_params_path.exists():
        raise FileNotFoundError(f"Camera parameters not found: {cam_params_path}")

    with open(cam_params_path) as f:
        cam_data = json.load(f)

    extrinsics = [torch.tensor(cam["matrix"], dtype=torch.float32) for cam in cam_data["extrinsics"]]
    intrinsics = [torch.tensor(cam["matrix"], dtype=torch.float32) for cam in cam_data["intrinsics"]]
    num_views = len(extrinsics)

    print(f"📷 Loaded {num_views} camera parameters")

    H,W=args.height, args.width
    print(f"📐 Image dimensions: {W} x {H}")

    # Load model for rendering
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs_renderer = GaussianSplatRenderer().to(device)

    print("🎨 Starting cumulative rendering...")

    for i in range(num_views):
        print(f"\n🔄 Processing cumulative views 0-{i}")

        # Load splats for views 0 to i
        all_splats = []
        for j in range(i + 1):
            zip_path = infer_dir / f"splats_view_{j}.zip"
            ply_path = infer_dir / f"splats_view_{j}.ply"

            if zip_path.exists():
                splats_j = load_splats_from_zip(zip_path)
                print(f"  📦 Loaded splats_view_{j}.zip: {splats_j['means'].shape[1]} splats")
            elif ply_path.exists():
                splats_j = load_splats_from_ply(ply_path)
                print(f"  📄 Loaded splats_view_{j}.ply: {splats_j['means'].shape[1]} splats")
            else:
                raise FileNotFoundError(f"Neither splats_view_{j}.zip nor splats_view_{j}.ply found")

            all_splats.append(splats_j)

        # Concatenate splats along the N dimension (dim=1)
        combined_splats = {
            "means": torch.cat([s["means"].to(device) for s in all_splats], dim=1),
            "quats": torch.cat([s["quats"].to(device) for s in all_splats], dim=1),
            "scales": torch.cat([s["scales"].to(device) for s in all_splats], dim=1),
            "opacities": torch.cat([s["opacities"].to(device) for s in all_splats], dim=1),
            "sh": torch.cat([s["sh"].to(device) for s in all_splats], dim=1),
        }

        total_splats = combined_splats["means"].shape[1]
        print(f"  🔗 Combined splats: {total_splats} total")

        # Prepare cameras for views 0 to i
        viewmats = torch.stack(extrinsics[:i+1]).unsqueeze(0).to(device)  # [1, i+1, 4, 4]
        Ks = torch.stack(intrinsics[:i+1]).unsqueeze(0).to(device)        # [1, i+1, 3, 3]

        # Render using the same call as render_interpolated_video
        with torch.no_grad():
            colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
                combined_splats["means"], combined_splats["quats"], combined_splats["scales"],
                combined_splats["opacities"].squeeze(-1), combined_splats["sh"].unsqueeze(-2),
                viewmats, Ks, width=W, height=H, sh_degree=0
            )

        # Save renders for each camera
        for cam_idx in range(i + 1):
            rgb_tensor = colors[0, cam_idx]  # [H, W, 3] - already correct for PIL
            render_path = render_dir / f"cumulative_view_{i}_cam_{cam_idx}.png"
            save_image_png(render_path, rgb_tensor)
            print(f"  💾 Saved: {render_path}")

    print("\n✅ Cumulative rendering complete!")
    print(f"📂 Renders saved to: {render_dir}")


if __name__ == "__main__":
    main()