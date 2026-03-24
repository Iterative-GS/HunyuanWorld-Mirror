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

import numpy as np
from src.models.models.rasterization import GaussianSplatRenderer
import torch
from PIL import Image
from plyfile import PlyData
import zipfile
import OpenEXR
import Imath

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


def load_splats_from_exr(exr_zip_path):
    """
    Load Gaussian splats from EXR zip file and convert to tensor format expected by rasterizer.

    Args:
        exr_zip_path: Path to EXR zip file

    Returns:
        Dictionary with splat tensors in format expected by rasterize_batches
    """
    with zipfile.ZipFile(exr_zip_path, 'r') as z:
        exr_filename = z.namelist()[0]
        with z.open(exr_filename) as f:
            exr = OpenEXR.InputFile(f)
            header = exr.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Read channels
            means_0 = np.frombuffer(exr.channel('means_0'), dtype=np.float16).reshape(height, width)
            means_1 = np.frombuffer(exr.channel('means_1'), dtype=np.float16).reshape(height, width)
            means_2 = np.frombuffer(exr.channel('means_2'), dtype=np.float16).reshape(height, width)
            means = np.stack([means_0, means_1, means_2], axis=-1).astype(np.float32)  # H x W x 3

            quats_0 = np.frombuffer(exr.channel('quats_0'), dtype=np.float16).reshape(height, width)
            quats_1 = np.frombuffer(exr.channel('quats_1'), dtype=np.float16).reshape(height, width)
            quats_2 = np.frombuffer(exr.channel('quats_2'), dtype=np.float16).reshape(height, width)
            quats_3 = np.frombuffer(exr.channel('quats_3'), dtype=np.float16).reshape(height, width)
            quats = np.stack([quats_0, quats_1, quats_2, quats_3], axis=-1).astype(np.float32)  # H x W x 4

            scales_0 = np.frombuffer(exr.channel('scales_0'), dtype=np.float16).reshape(height, width)
            scales_1 = np.frombuffer(exr.channel('scales_1'), dtype=np.float16).reshape(height, width)
            scales_2 = np.frombuffer(exr.channel('scales_2'), dtype=np.float16).reshape(height, width)
            scales = np.stack([scales_0, scales_1, scales_2], axis=-1).astype(np.float32)  # H x W x 3

            opacities_0 = np.frombuffer(exr.channel('opacities_0'), dtype=np.float16).reshape(height, width)
            opacities = opacities_0.astype(np.float32)  # H x W

            sh_0_0 = np.frombuffer(exr.channel('sh_0_0'), dtype=np.float16).reshape(height, width)
            sh_0_1 = np.frombuffer(exr.channel('sh_0_1'), dtype=np.float16).reshape(height, width)
            sh_0_2 = np.frombuffer(exr.channel('sh_0_2'), dtype=np.float16).reshape(height, width)
            sh = np.stack([sh_0_0, sh_0_1, sh_0_2], axis=-1).astype(np.float32)  # H x W x 3

            # Flatten to N x ...
            N = height * width
            means_flat = means.reshape(N, 3)
            quats_flat = quats.reshape(N, 4)
            scales_flat = scales.reshape(N, 3)
            opacities_flat = opacities.reshape(N)
            sh_flat = sh.reshape(N, 3)

            # Convert to torch
            return {
                "means": torch.tensor(means_flat, dtype=torch.float32).unsqueeze(0),      # [1, N, 3]
                "quats": torch.tensor(quats_flat, dtype=torch.float32).unsqueeze(0),      # [1, N, 4]
                "scales": torch.tensor(scales_flat, dtype=torch.float32).unsqueeze(0),    # [1, N, 3]
                "opacities": torch.tensor(opacities_flat, dtype=torch.float32).unsqueeze(0), # [1, N]
                "sh": torch.tensor(sh_flat, dtype=torch.float32).unsqueeze(1).unsqueeze(0)  # [1, N, 1, 3]
            }


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
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.gs_renderer.enable_prune = False


    print("🎨 Starting cumulative rendering...")

    for i in range(num_views):
        print(f"\n🔄 Processing cumulative views 0-{i}")

        # Load splats for views 0 to i
        all_splats = []
        for j in range(i + 1):
            zip_path = infer_dir / f"splats_view_{j}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"Splat file not found: {zip_path}")

            splats_j = load_splats_from_exr(zip_path)
            all_splats.append(splats_j)
            print(f"  📄 Loaded splats_view_{j}.zip: {splats_j['means'].shape[1]} splats")

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
            colors, depths, _ = model.gs_renderer.rasterizer.rasterize_batches(
                combined_splats["means"], combined_splats["quats"], combined_splats["scales"],
                combined_splats["opacities"], combined_splats["sh"],
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