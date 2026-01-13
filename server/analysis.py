import cv2
import numpy as np
import torch
import json
import base64
from io import BytesIO
from PIL import Image
# Assuming sam2.build_sam and its dependencies are correctly set up
# from sam2.build_sam import build_sam2_camera_predictor

# Mocking the predictor for standalone execution if sam2 is not available
# In your actual environment, uncomment the sam2 imports and checkpoint loading
try:
    from sam2.build_sam import build_sam2_camera_predictor
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt" # Adjust path as needed
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml" # Adjust path as needed
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint) # UNCOMMENT THIS LINE
    
except ImportError:
    print("sam2.build_sam could not be imported. Using a dummy predictor for demonstration.")
    class DummyPredictor:
        def load_first_frame(self, frame): print("DummyPredictor: Loaded first frame.")
        def add_new_prompt(self, frame_idx, obj_id, points, labels): print(f"DummyPredictor: Added new prompt for obj_id {obj_id} at frame {frame_idx}."); return None, [obj_id], torch.rand(1, 1, 128, 128) > 0.5 # Mock output
        def track(self, frame): return [1], torch.rand(1, 1, frame.shape[0], frame.shape[1]) > 0.5 # Mock output
    predictor = DummyPredictor()
except Exception as e:
    print(f"An error occurred during PyTorch or SAM2 setup: {e}")
    print("Using a dummy predictor for demonstration.")
    class DummyPredictor:
        def load_first_frame(self, frame): print("DummyPredictor: Loaded first frame.")
        def add_new_prompt(self, frame_idx, obj_id, points, labels): print(f"DummyPredictor: Added new prompt for obj_id {obj_id} at frame {frame_idx}."); return None, [obj_id], torch.rand(1, 1, 128, 128) > 0.5 # Mock output
        def track(self, frame): return [1], torch.rand(1, 1, frame.shape[0], frame.shape[1]) > 0.5 # Mock output
    predictor = DummyPredictor()


def get_mask_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

def point_side(p1, p2, p):
    if p is None: # Handle cases where center might be None
        return 0 # Or some other default behavior, like assuming it hasn't crossed
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

def generate_mask_from_points(input_video: str,
                               video_time: float,
                               points: list[tuple[float, float]],
                               labels: list[int]) -> str:
    """
    Generate a mask from positive and negative point prompts.

    Args:
        input_video: Path to the video file
        video_time: Time in seconds to extract the frame
        points: List of (x, y) tuples in relative coordinates (0-1)
        labels: List of labels (1 for positive, 0 for negative)

    Returns:
        Base64-encoded PNG image of the mask
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_width == 0 or video_height == 0:
        raise RuntimeError(f"Video dimensions ({video_width}x{video_height}) are invalid.")

    # Seek to the target frame
    target_frame_idx = int(video_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Could not read frame at {video_time}s (frame {target_frame_idx})")

    # Convert BGR to RGB for SAM
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert relative coordinates to absolute pixel coordinates
    points_abs = np.array([
        [p[0] * video_width, p[1] * video_height] for p in points
    ], dtype=np.float32)

    labels_array = np.array(labels, dtype=np.int32)

    print(f"Generating mask with {len(points)} points at frame {target_frame_idx}")
    print(f"Points (absolute): {points_abs}")
    print(f"Labels: {labels_array}")

    # Load frame into predictor and generate mask
    predictor.load_first_frame(frame_rgb)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=target_frame_idx,
        obj_id=1,
        points=points_abs,
        labels=labels_array
    )

    # Convert mask to numpy array
    mask = (out_mask_logits[0] > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
    mask_gray = np.squeeze(mask)
    if mask_gray.ndim == 3 and mask_gray.shape[-1] == 1:
        mask_gray = mask_gray[:, :, 0]

    # Convert mask to PIL Image and then to base64
    mask_pil = Image.fromarray(mask_gray)
    buffered = BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return mask_base64

def run_full_analysis(input_video: str,
                      line_pts: list[list[float]], # Changed to float for relative input
                      meas_pts: list[list[float]], # Changed to float for relative input
                      click_pt: tuple[float,float], # Changed to float for relative input
                      known_length: float,
                      output_video: str,
                      output_metrics: str,
                      video_time: float):
    """
    1. Reads input_video
    2. Draws angle/measurement lines on first frame
    3. Initializes SAM2 tracking
    4. Tracks object, computes elapsed time, etc.
    5. Writes annotated video to output_video (e.g. via cv2.VideoWriter)
    6. Dumps metrics JSON to output_metrics, including relative object positions
    """
    
    start_crossed = False
    finish_crossed = False
    start_frame_idx = None
    finish_frame_idx = None
    start_line_abs=None # Absolute pixel coordinates
    finish_line_abs=None # Absolute pixel coordinates
    elapsed_frames = 0
    elapsed_time = 0
    start_center_abs = None
    finish_center_abs = None
    
    object_positions_relative = [] # To store relative positions frame by frame

    print(f"Input video: {input_video}")
    print(f"Line points (relative): {line_pts}")
    print(f"Click point (relative): {click_pt}")
    print(f"Known length: {known_length} m")
    print(f"Measurement points (relative): {meas_pts}")
    print(f"Video time for initial click: {video_time}")


    

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get video dimensions ONCE and use these for all relative calculations
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if video_width == 0 or video_height == 0:
        raise RuntimeError(f"Video dimensions ({video_width}x{video_height}) are invalid.")

    print(f"Video Width: {video_width}, Video Height: {video_height}, FPS: {fps}")


    meas1_abs = np.array([meas_pts[0][0] * video_width,
                            meas_pts[0][1] * video_height], dtype=np.float32)
    meas2_abs = np.array([meas_pts[1][0] * video_width,
                           meas_pts[1][1] * video_height], dtype=np.float32)
    
    meas_px = np.linalg.norm(meas2_abs - meas1_abs)

    scale_m_per_px = known_length / meas_px if meas_px > 0 else 0.0

    # === Define and draw start/finish lines (convert relative to absolute) ===
    # line_pts are expected to be [[x1_rel, y1_rel], [x2_rel, y2_rel]]
    p1_abs = np.array([line_pts[0][0] * video_width, line_pts[0][1] * video_height], dtype=np.float32)
    p2_abs = np.array([line_pts[1][0] * video_width, line_pts[1][1] * video_height], dtype=np.float32)
    print(f"Start line (absolute): {p1_abs}, Finish line (absolute): {p2_abs}")

    direction = p2_abs - p1_abs
    length = np.linalg.norm(direction)
    if length != 0:
        direction /= length
        perp_dir = np.array([-direction[1], direction[0]])
        line_display_len = 50 # Length of the perpendicular segment to display

        start_line_abs = (tuple((p1_abs - perp_dir * line_display_len / 2).astype(int)),
                          tuple((p1_abs + perp_dir * line_display_len / 2).astype(int)))
        finish_line_abs = (tuple((p2_abs - perp_dir * line_display_len / 2).astype(int)),
                           tuple((p2_abs + perp_dir * line_display_len / 2).astype(int)))
    else:
        print("Warning: Start and finish points for the line are the same.")
        # Handle degenerate case, perhaps by not drawing lines or setting them to point values
        start_line_abs = (tuple(p1_abs.astype(int)), tuple(p1_abs.astype(int)))
        finish_line_abs = (tuple(p2_abs.astype(int)), tuple(p2_abs.astype(int)))


    target_frame_idx = int(video_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    success, frame_at_time = cap.read()
    
    if not success:
        # Try to read the first frame if the target frame fails (e.g., video_time is out of bounds)
        print(f"Warning: Could not read frame at {video_time}s (frame {target_frame_idx}). Trying first frame.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_frame_idx = 0
        success, frame_at_time = cap.read()
        if not success:
             raise RuntimeError(f"Could not read frame at {video_time}s (frame {target_frame_idx}) or the first frame.")

    # Use the initially fetched video_width and video_height for consistency in relative coordinate calculations
    # current_frame_height, current_frame_width = frame_at_time.shape[:2] # This is redundant if video_width/height are global

    frame_rgb = cv2.cvtColor(frame_at_time, cv2.COLOR_BGR2RGB)
    predictor.load_first_frame(frame_rgb) # Use RGB for SAM

    ann_obj_id = 1
    
    # Convert relative click_pt to absolute pixel coordinates
    click_pt_abs = np.array([click_pt[0] * video_width, click_pt[1] * video_height], dtype=np.float32)
    
    points_for_sam = np.array([click_pt_abs], dtype=np.float32)
    labels_for_sam = np.array([1], dtype=np.int32)

    print(f"Points for SAM (absolute): {points_for_sam}")
    print(f"Labels for SAM: {labels_for_sam}")

    
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=target_frame_idx, obj_id=ann_obj_id, points=points_for_sam, labels=labels_for_sam
        )
    
    print(f"Mask: {out_mask_logits}")
    mask = (out_mask_logits[0] > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
    mask_gray = np.squeeze(mask) # Squeeze to make it 2D if it's 3D with one channel
    if mask_gray.ndim == 3 and mask_gray.shape[-1] == 1: # Ensure it's 2D
        mask_gray = mask_gray[:,:,0]

    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    # Use video_width and video_height for the output video dimensions
    out = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to the beginning for processing all frames
    current_frame_num = 0 
    print(f"Processed Start Line (absolute): {start_line_abs}")
    print(f"Processed Finish Line (absolute): {finish_line_abs}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # The frame read is BGR. SAM expects RGB.
        frame_rgb_for_tracking = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        out_obj_ids, out_mask_logits = predictor.track(frame_rgb_for_tracking) # Pass RGB frame

        mask = (out_mask_logits[0] > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        mask_gray = np.squeeze(mask) # Squeeze to make it 2D if it's 3D with one channel
        if mask_gray.ndim == 3 and mask_gray.shape[-1] == 1: # Ensure it's 2D
            mask_gray = mask_gray[:,:,0]

        center_abs = get_mask_center(mask_gray) # This is in absolute pixel coordinates

        relative_x, relative_y = None, None
        if center_abs is not None:
            relative_x = center_abs[0] / video_width
            relative_y = center_abs[1] / video_height
            # Draw circle on the BGR frame for output video
            frame = cv2.circle(frame, center_abs, 5, (255, 0, 0), -1) # Blue circle
        
        object_positions_relative.append({
            "frame": current_frame_num,
            "relative_x": relative_x,
            "relative_y": relative_y,
            "absolute_x": center_abs[0] if center_abs else None, # Optional: also store absolute for debugging
            "absolute_y": center_abs[1] if center_abs else None, # Optional: also store absolute for debugging
        })
            
        if start_line_abs and finish_line_abs and center_abs is not None: # Check if lines are defined and center is found
            if not start_crossed:
                side = point_side(start_line_abs[0], start_line_abs[1], center_abs)
                if side < 0: # Assuming crossing when object moves to the "negative" side
                    start_crossed = True
                    start_frame_idx = current_frame_num
                    start_center_abs = center_abs
                    print(f"Crossed Start at frame {current_frame_num}")

            if start_crossed and not finish_crossed:
                side = point_side(finish_line_abs[0], finish_line_abs[1], center_abs)
                if side < 0: # Assuming crossing condition
                    finish_crossed = True
                    finish_frame_idx = current_frame_num
                    elapsed_frames = finish_frame_idx - start_frame_idx if start_frame_idx is not None else 0
                    elapsed_time = elapsed_frames / fps if fps > 0 else 0
                    finish_center_abs = center_abs
                    print(f"Crossed Finish at frame {current_frame_num}")
                    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
        
        # Draw lines on the BGR frame
        if start_line_abs:
            cv2.line(frame, start_line_abs[0], start_line_abs[1], (0, 255, 0), 2) # Green start line
        if finish_line_abs:
            cv2.line(frame, finish_line_abs[0], finish_line_abs[1], (0, 0, 255), 2) # Red finish line
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {current_frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if start_crossed:
            cv2.putText(frame, "Start Crossed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if finish_crossed:
            cv2.putText(frame, f"Finish Crossed! Time: {elapsed_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        out.write(frame) # Write the BGR frame with annotations
        current_frame_num += 1
    
    cap.release()
    out.release()
    print(f"Video processing complete. Output video saved to: {output_video}")
    
    if start_center_abs is not None and finish_center_abs is not None:
        px_travel = np.linalg.norm(np.array(finish_center_abs) - np.array(start_center_abs))
        real_travel = px_travel * scale_m_per_px
    else:
        real_travel = None

    metrics = {
        "video_width": video_width,
        "video_height": video_height,
        "fps": fps,
        "initial_prompt_frame": target_frame_idx,
        "start_line_crossed_frame": start_frame_idx,
        "finish_line_crossed_frame": finish_frame_idx,
        "elapsed_frames_between_lines": elapsed_frames if finish_crossed else None,
        "elapsed_time_seconds": elapsed_time if finish_crossed else None,
        "known_length_for_measurement": real_travel,
        # "angle_degrees": 45.0, # Example, calculate if needed
        # "measured_length_pixels": 10.2, # Example, calculate if needed
        "object_positions": object_positions_relative # Add the list of positions
    }

    # Save metrics
    with open(output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {output_metrics}")
