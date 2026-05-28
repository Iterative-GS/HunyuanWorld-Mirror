"""
Video utilities for visualization.

"""

import os
from pathlib import Path
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import subprocess
try:
    from PIL import Image
except Exception:
    Image = None

def _rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    trace = float(np.trace(R_rel))
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(angle_rad))


def _select_frames_by_trajectory_arclength(
    poses,
    max_frames: int,
    *,
    traj_step_frac_scene: float | None = 0.05,
    traj_step_frac_path: float | None = None,
    traj_min_step_abs: float = 1e-3,
    traj_smooth_window: int = 0,
    traj_min_rot_deg: float | None = None,
    traj_redundant_trans_frac_scene: float = 0.02,
    traj_redundant_rot_deg: float = 7.5,
):
    """
    Select frames by sampling roughly uniformly along the camera trajectory (arc-length)
    in time order, with a local redundancy gate.

    Returns:
        (selected_indices, meta)
    """
    import math
    from src.utils.pose_overlap import compute_scene_scale

    frame_indices = sorted(poses.keys())
    if not frame_indices:
        return [], {"reason": "no_poses"}

    # Positions in time order
    pos = np.stack([poses[i][:3, 3].astype(np.float64) for i in frame_indices], axis=0)

    if traj_smooth_window and traj_smooth_window > 1:
        w = int(traj_smooth_window)
        half = w // 2
        pos_smooth = pos.copy()
        for i in range(len(pos)):
            lo = max(0, i - half)
            hi = min(len(pos), i + half + 1)
            pos_smooth[i] = pos[lo:hi].mean(axis=0)
        pos_use = pos_smooth
    else:
        pos_use = pos

    # Arc-length along trajectory
    deltas = np.linalg.norm(pos_use[1:] - pos_use[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    total_path_len = float(s[-1])

    scene_scale = float(compute_scene_scale(pos_use))
    eps = 1e-9

    # Determine step size
    if traj_step_frac_scene is not None:
        step = float(traj_step_frac_scene) * max(scene_scale, eps)
        step_mode = "scene_scale"
    elif traj_step_frac_path is not None:
        step = float(traj_step_frac_path) * max(total_path_len, eps)
        step_mode = "path_length"
    else:
        if max_frames >= 2:
            step = max(total_path_len / float(max_frames - 1), eps)
        else:
            step = math.inf
        step_mode = "derived_from_max_frames"

    step = max(step, float(traj_min_step_abs))

    # Degenerate: no motion => uniform-in-time indices
    if total_path_len <= eps:
        if max_frames <= 1:
            sel = [frame_indices[0]]
        else:
            # Evenly spaced in time
            k = min(max_frames, len(frame_indices))
            picks = np.linspace(0, len(frame_indices) - 1, k)
            sel = [frame_indices[int(round(x))] for x in picks]
            sel = sorted(set(sel), key=sel.index)
        meta = {
            "mode": "trajectory",
            "reason": "degenerate_path_len",
            "total_path_length": total_path_len,
            "scene_scale": scene_scale,
            "trajectory_step": step,
            "step_mode": step_mode,
            "redundancy_gate_applied_count": 0,
        }
        return sel, meta

    # Greedy arc-length sampling with redundancy gate
    selected = [frame_indices[0]]
    redundancy_skips = 0
    next_target = step

    for idx_in_seq in range(1, len(frame_indices)):
        if len(selected) >= max_frames:
            break

        if float(s[idx_in_seq]) < next_target:
            continue

        cand_idx = frame_indices[idx_in_seq]
        last_idx = selected[-1]

        # Local redundancy gate vs most recent kept
        trans_redundant = float(traj_redundant_trans_frac_scene) * max(scene_scale, eps)
        dtrans = float(np.linalg.norm(poses[cand_idx][:3, 3] - poses[last_idx][:3, 3]))
        drot = _rotation_geodesic_deg(poses[last_idx][:3, :3], poses[cand_idx][:3, :3])

        if dtrans <= trans_redundant and drot <= float(traj_redundant_rot_deg):
            redundancy_skips += 1
            continue

        if traj_min_rot_deg is not None and drot < float(traj_min_rot_deg) and dtrans < step:
            redundancy_skips += 1
            continue

        selected.append(cand_idx)
        next_target += step

    # Always include last frame if there's room and it's not already included
    if len(selected) < max_frames and frame_indices[-1] not in selected:
        selected.append(frame_indices[-1])

    meta = {
        "mode": "trajectory",
        "total_path_length": total_path_len,
        "scene_scale": scene_scale,
        "trajectory_step": step,
        "step_mode": step_mode,
        "traj_step_frac_scene": traj_step_frac_scene,
        "traj_step_frac_path": traj_step_frac_path,
        "traj_min_step_abs": traj_min_step_abs,
        "traj_smooth_window": traj_smooth_window,
        "traj_min_rot_deg": traj_min_rot_deg,
        "traj_redundant_trans_frac_scene": traj_redundant_trans_frac_scene,
        "traj_redundant_rot_deg": traj_redundant_rot_deg,
        "redundancy_gate_applied_count": redundancy_skips,
    }
    return selected, meta


def video_to_image_frames(input_video_path, save_directory=None, fps=1):
    """
    Extracts image frames from a video file at the specified frame rate and saves them as JPEG format.
    Supports regular video files, webcam captures, WebM files, and GIF files, including incomplete files.
    
    Args:
        input_video_path: Path to the input video file
        save_directory: Directory to save extracted frames (default: None)
        fps: Number of frames to extract per second (default: 1)
    
    Returns: List of file paths to extracted frames
    """
    extracted_frame_paths = []
    
    # For GIF files, use PIL library for better handling
    if input_video_path.lower().endswith('.gif'):
        try:
            print(f"Processing GIF file using PIL: {input_video_path}")
            
            with Image.open(input_video_path) as gif_img:
                # Get GIF properties
                frame_duration_ms = gif_img.info.get('duration', 100)  # Duration per frame in milliseconds
                gif_frame_rate = 1000.0 / frame_duration_ms if frame_duration_ms > 0 else 10.0  # Convert to frame rate
                
                print(f"GIF properties: {gif_img.n_frames} frames, {gif_frame_rate:.2f} FPS, {frame_duration_ms}ms per frame")
                
                # Calculate sampling interval
                sampling_interval = max(1, int(gif_frame_rate / fps)) if fps < gif_frame_rate else 1
                
                saved_count = 0
                for current_frame_index in range(gif_img.n_frames):
                    gif_img.seek(current_frame_index)
                    
                    # Sample frames based on desired frame rate
                    if current_frame_index % sampling_interval == 0:
                        # Convert to RGB format if necessary
                        rgb_frame = gif_img.convert('RGB')
                        
                        # Convert PIL image to numpy array
                        frame_ndarray = np.array(rgb_frame)
                        
                        # Save frame as JPEG format
                        frame_output_path = os.path.join(save_directory, f"frame_{saved_count:06d}.jpg")
                        pil_image = Image.fromarray(frame_ndarray)
                        pil_image.save(frame_output_path, 'JPEG', quality=95)
                        extracted_frame_paths.append(frame_output_path)
                        saved_count += 1
                
                if extracted_frame_paths:
                    print(f"Successfully extracted {len(extracted_frame_paths)} frames from GIF using PIL")
                    return extracted_frame_paths
                    
        except Exception as error:
            print(f"PIL GIF extraction error: {str(error)}, falling back to OpenCV")
    
    # For WebM files, use FFmpeg directly for more stable processing
    if input_video_path.lower().endswith('.webm'):
        try:
            print(f"Processing WebM file using FFmpeg: {input_video_path}")
            
            # Create a unique output pattern for the frames
            output_frame_pattern = os.path.join(save_directory, "frame_%04d.jpg")
            
            # Use FFmpeg to extract frames at specified frame rate
            ffmpeg_command = [
                "ffmpeg", 
                "-i", input_video_path,
                "-vf", f"fps={fps}",  # Specified frames per second
                "-q:v", "2",     # High quality
                output_frame_pattern
            ]
            
            # Run FFmpeg process
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            process_stdout, process_stderr = ffmpeg_process.communicate()
            
            # Collect all extracted frames
            for filename in sorted(os.listdir(save_directory)):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    full_frame_path = os.path.join(save_directory, filename)
                    extracted_frame_paths.append(full_frame_path)
            
            if extracted_frame_paths:
                print(f"Successfully extracted {len(extracted_frame_paths)} frames from WebM using FFmpeg")
                return extracted_frame_paths
            
            print("FFmpeg extraction failed, falling back to OpenCV")
        except Exception as error:
            print(f"FFmpeg extraction error: {str(error)}, falling back to OpenCV")
    
    # Standard OpenCV method for non-WebM files or as fallback
    try:
        video_capture = cv2.VideoCapture(input_video_path)
        
        # For WebM files, try setting more robust decoder options
        if input_video_path.lower().endswith('.webm'):
            video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'VP80'))
        
        source_fps = video_capture.get(cv2.CAP_PROP_FPS)
        extraction_interval = max(1, int(source_fps / fps))  # Extract at specified frame rate
        processed_frame_count = 0
        
        # Set error mode to suppress console warnings
        cv2.setLogLevel(0)
        
        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break
                
            if processed_frame_count % extraction_interval == 0:
                try:
                    # Additional check for valid frame data
                    if current_frame is not None and current_frame.size > 0:
                        rgb_converted_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        frame_output_path = os.path.join(save_directory, f"frame_{processed_frame_count:06d}.jpg")
                        cv2.imwrite(frame_output_path, cv2.cvtColor(rgb_converted_frame, cv2.COLOR_RGB2BGR))
                        extracted_frame_paths.append(frame_output_path)
                except Exception as error:
                    print(f"Warning: Failed to process frame {processed_frame_count}: {str(error)}")
                    
            processed_frame_count += 1
            
            # Safety limit to prevent infinite loops
            if processed_frame_count > 1000:
                break
                
        video_capture.release()
        print(f"Extracted {len(extracted_frame_paths)} frames from video using OpenCV")
        
    except Exception as error:
        print(f"Error extracting frames: {str(error)}")
            
    return extracted_frame_paths

def _select_frames_by_pose_constraints(poses, n):
    """
    Select n frames using pose-based constraints.
    
    Algorithm:
    - Start with frame 0
    - For each subsequent frame i (1 to n-1):
        - Rotation threshold: (i+1) * (180/n) degrees
        - Find frame with max translation from frame 0 that has rotation <= threshold
        - If no such frame exists, take the frame with max translation overall
        - Mark selected frame as used
    """
    frame_indices = sorted(poses.keys())
    selected_indices = []
    remaining_indices = set(frame_indices)
    
    # Always start with first frame
    selected_indices.append(0)
    remaining_indices.discard(0)

    ref_pose = poses[0]
    ref_position = ref_pose[:3, 3]

    # Keep list of currently selected positions; used to compute distance to the set
    selected_positions = [ref_position]

    for i in range(1, n):
        # Rotation threshold for this frame: (i+1) * (180/n) degrees
        rotation_threshold_deg = (i + 1) * (180.0 / n)
        rotation_threshold_rad = np.deg2rad(rotation_threshold_deg)
        
        # Find frame with max translation within rotation constraint
        best_idx = None
        best_dist = -1
        best_dist_unconstrained = -1
        best_idx_unconstrained = None
        
        for idx in remaining_indices:
            pose = poses[idx]
            position = pose[:3, 3]
            # Compute distance to the set of already-selected frames: use minimum distance
            dists_to_selected = [np.linalg.norm(position - p) for p in selected_positions]
            distance = float(np.min(dists_to_selected))

            # Compute rotation angle between ref_pose (start) and this pose
            R_rel = ref_pose[:3, :3].T @ pose[:3, :3]
            trace = np.trace(R_rel)
            # Clamp trace to [-1, 1] to avoid numerical issues
            trace = np.clip(trace, -1, 1)
            rotation_angle = np.arccos((trace - 1) / 2.0)

            # Track best unconstrained frame (furthest from selected set)
            if distance > best_dist_unconstrained:
                best_dist_unconstrained = distance
                best_idx_unconstrained = idx

            # Check if within rotation constraint (relative to starting frame)
            if rotation_angle <= rotation_threshold_rad:
                if distance > best_dist:
                    best_dist = distance
                    best_idx = idx
        
        # Select frame: prefer constrained frame, fall back to unconstrained
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            selected_choice = best_idx
            selected_dist = best_dist
            print(f"   Frame {i}: selected idx={best_idx} (dist_to_selected_set={best_dist:.3f}, rot<{rotation_threshold_deg:.1f}°)")
        else:
            selected_indices.append(best_idx_unconstrained)
            remaining_indices.discard(best_idx_unconstrained)
            selected_choice = best_idx_unconstrained
            selected_dist = best_dist_unconstrained
            print(f"   Frame {i}: selected idx={best_idx_unconstrained} (dist_to_selected_set={best_dist_unconstrained:.3f}, no rot constraint satisfied, threshold was {rotation_threshold_deg:.1f}°)")

        # Add newly selected position to the selected_positions list
        selected_positions.append(poses[selected_choice][:3, 3])
        
        if len(remaining_indices) == 0:
            print(f" Ran out of frames; selected {len(selected_indices)} out of {n}")
            break
    
    # Return indices sorted to preserve original temporal order in the video
    return sorted(selected_indices)
def select_frames_from_dl3dv(
    dataset_dir,
    n=10,
    output_dir=None,
    *,
    frame_selection: str = "trajectory",
    dedupe_overlap=True,
    overlap_rot_deg=10.0,
    overlap_trans_frac_scene=0.08,
    overlap_trans_frac_path=0.20,
    overlap_min_views=3,
    # Trajectory selection params
    traj_step_frac_scene: float | None = 0.05,
    traj_step_frac_path: float | None = None,
    traj_min_step_abs: float = 1e-3,
    traj_smooth_window: int = 0,
    traj_min_rot_deg: float | None = None,
    traj_redundant_trans_frac_scene: float = 0.02,
    traj_redundant_rot_deg: float = 7.5,
):
    """
    Select n frames from a DL3DV-10K dataset directory using pre-computed COLMAP poses.
    
    Structure expected:
    dataset_dir/
    ├── transforms.json          (COLMAP camera poses)
    └── images_4/                (or images/, images_2/, etc.)
        ├── frame_00001.png
        ├── frame_00002.png
        ...
    
    Args:
        dataset_dir: Path to DL3DV dataset directory
        n: Number of frames to select
        output_dir: Directory to save selected frames (default: dataset_dir/selected_frames)
    
    Returns:
        List of paths to selected frames (sorted by frame index)
    """
    import json
    
    dataset_dir = Path(dataset_dir)
    transforms_path = dataset_dir / "transforms.json"
    if not transforms_path.exists():
        print(f"❌ transforms.json not found in {dataset_dir}")
        return None
    images_dirs = sorted(dataset_dir.glob("images*"))
    if not images_dirs:
        print(f"❌ No images* directory found in {dataset_dir}")
        return None
    images_dir = None
    for candidate in ["images_4", "images_8", "images"]:
        candidate_path = dataset_dir / candidate
        if candidate_path.is_dir():
            images_dir = candidate_path
            break
    
    if images_dir is None:
        images_dir = images_dirs[-1]
    
    print(f" Using images directory: {images_dir.name}")

    print(f"Loading camera poses from transforms.json...")
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
    except Exception as e:
        print(f" Error loading transforms.json: {e}")
        return None
    
    # Extract camera frames
    frames_data = transforms.get("frames", [])
    if not frames_data:
        print(f"No frames found in transforms.json")
        return None
    
    print(f"   Found {len(frames_data)} frames in transforms.json")
    
    # Get all frame paths from images directory
    all_frame_paths = sorted(images_dir.glob("frame_*.png"))
    if not all_frame_paths:
        all_frame_paths = sorted(images_dir.glob("*.png"))
    
    if not all_frame_paths:
        print(f"No PNG files found in {images_dir}")
        return None
    
    print(f"   Found {len(all_frame_paths)} image files")
    
    if len(all_frame_paths) < n:
        print(f"Dataset has only {len(all_frame_paths)} frames but {n} requested. Returning all frames.")
        return all_frame_paths
    poses = {}
    for frame_idx, frame_data in enumerate(frames_data):
        if "transform_matrix" in frame_data:
            pose_matrix = np.array(frame_data["transform_matrix"], dtype=np.float32)
            if pose_matrix.shape == (4, 4):
                poses[frame_idx] = pose_matrix
    
    if not poses:
        print(f"Could not extract valid poses from transforms.json")
        return None
    
    print(f"Extracted {len(poses)} valid camera poses")
    
    initial_indices = []
    selected_indices = []
    drop_log = []
    traj_meta = {}

    if frame_selection == "legacy":
        # Select frames using pose constraints
        print(f"Selecting {n} frames by pose constraints...")
        initial_indices = _select_frames_by_pose_constraints(poses, n)
        selected_indices = list(initial_indices)

        if dedupe_overlap and len(selected_indices) > 1:
            from src.utils.pose_overlap import dedupe_frames_by_pose_overlap

            print(
                f" Deduplicating overlapping views (rot<={overlap_rot_deg}°, "
                f"trans_frac_scene={overlap_trans_frac_scene}, trans_frac_path={overlap_trans_frac_path})..."
            )
            selected_indices, drop_log = dedupe_frames_by_pose_overlap(
                selected_indices,
                poses,
                rot_deg_thresh=overlap_rot_deg,
                frac_scene=overlap_trans_frac_scene,
                frac_path=overlap_trans_frac_path,
                min_kept_views=overlap_min_views,
            )
            print(
                f"   Overlap dedupe: kept {len(selected_indices)}/{len(initial_indices)} frames "
                f"(dropped {len(drop_log)})"
            )
            for rec in drop_log:
                print(
                    f"     dropped idx={rec['index']} (closest_kept={rec.get('closest_kept')}, "
                    f"rot={rec.get('rot_deg', 0):.2f}°, trans={rec.get('trans_dist', 0):.4f}, "
                    f"thresh={rec.get('trans_thresh', 0):.4f})"
                )
    else:
        print(f"Selecting up to {n} frames by trajectory arc-length sampling...")
        selected_indices, traj_meta = _select_frames_by_trajectory_arclength(
            poses,
            n,
            traj_step_frac_scene=traj_step_frac_scene,
            traj_step_frac_path=traj_step_frac_path,
            traj_min_step_abs=traj_min_step_abs,
            traj_smooth_window=traj_smooth_window,
            traj_min_rot_deg=traj_min_rot_deg,
            traj_redundant_trans_frac_scene=traj_redundant_trans_frac_scene,
            traj_redundant_rot_deg=traj_redundant_rot_deg,
        )
        initial_indices = list(selected_indices)

        # In trajectory mode, overlap dedupe is optional but off by default upstream.
        if dedupe_overlap and len(selected_indices) > 1:
            from src.utils.pose_overlap import dedupe_frames_by_pose_overlap

            print(
                f" Optional global overlap dedupe (rot<={overlap_rot_deg}°, "
                f"trans_frac_scene={overlap_trans_frac_scene}, trans_frac_path={overlap_trans_frac_path})..."
            )
            selected_indices, drop_log = dedupe_frames_by_pose_overlap(
                selected_indices,
                poses,
                rot_deg_thresh=overlap_rot_deg,
                frac_scene=overlap_trans_frac_scene,
                frac_path=overlap_trans_frac_path,
                min_kept_views=overlap_min_views,
            )
            print(
                f"   Overlap dedupe: kept {len(selected_indices)}/{len(initial_indices)} frames "
                f"(dropped {len(drop_log)})"
            )

    # Copy selected frames to output directory
    if output_dir is None:
        output_dir = dataset_dir / "selected_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_meta = {
        "pose_source": "transforms.json transform_matrix",
        "frame_selection_mode": frame_selection,
        "initial_selection": initial_indices,
        "final_selection": selected_indices,
        "dropped": drop_log,
        "final_view_count": len(selected_indices),
        "dedupe_overlap": bool(dedupe_overlap),
        "overlap_rot_deg": overlap_rot_deg,
        "overlap_trans_frac_scene": overlap_trans_frac_scene,
        "overlap_trans_frac_path": overlap_trans_frac_path,
        "overlap_min_views": overlap_min_views,
        "trajectory": traj_meta,
        "pose_units_note": "All translation thresholds/steps are in transforms.json translation units.",
    }
    selection_meta_path = output_dir / "frame_selection.json"
    with open(selection_meta_path, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)
    print(f"   Wrote frame selection log to {selection_meta_path}")
    
    print(f"\n Saving selected frames to {output_dir}...")
    selected_paths = []
    for out_idx, frame_idx in enumerate(selected_indices):
        src = all_frame_paths[frame_idx]
        dst = output_dir / f"frame_{out_idx:06d}.png"
        import shutil
        shutil.copy2(src, dst)
        selected_paths.append(str(dst))
        print(f"   Frame {frame_idx} ({src.name}) → {dst.name}")
    
    print(f"Selected {len(selected_paths)} frames")
    return selected_paths