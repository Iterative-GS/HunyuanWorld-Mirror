"""
Video utilities for visualization.

"""

import os
from zipfile import Path
import cv2
import numpy as np
import subprocess
from PIL import Image


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
def select_frames_from_dl3dv(dataset_dir, n=10, output_dir=None):
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
    
    # Select frames using pose constraints
    print(f"Selecting {n} frames by pose constraints...")
    selected_indices = _select_frames_by_pose_constraints(poses, n)
    
    # Copy selected frames to output directory
    if output_dir is None:
        output_dir = dataset_dir / "selected_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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