import cv2
import subprocess
import tempfile
import os
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import json
import time
import shutil

# Paths
base_path = "/home/aun/Desktop/2nd paper/codes/sam2approach/New"
input_video = os.path.join(base_path, "part and path.mp4")
repaired_path = os.path.join(base_path, "sam2_repair.mp4")
output_video_path = os.path.join(base_path, "sam2_annotated.mp4")
output_json_path = os.path.join(base_path, "sam2_border_pixels.json")
txt_path = os.path.join(base_path, "Pics")
os.makedirs(txt_path, exist_ok=True)

START_FRAME = 401

print("=" * 50)
print(f"STEP 1: Extract frames from {START_FRAME} onward (repaired video starts at frame 0)")
print("=" * 50)

start_total = time.time()
start_repair = time.time()

# Get video info
result = subprocess.run(
    ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
     '-show_entries', 'stream=r_frame_rate,width,height', 
     '-of', 'default=noprint_wrappers=1', input_video],
    capture_output=True, text=True
)

info = {}
for line in result.stdout.strip().split('\n'):
    if '=' in line:
        k, v = line.split('=')
        info[k] = v

if 'r_frame_rate' in info:
    parts = info['r_frame_rate'].split('/')
    fps = float(parts[0]) / float(parts[1]) if len(parts) > 1 else float(parts[0])
else:
    fps = 30.0

width = int(info.get('width', 1236))
height = int(info.get('height', 682))

print(f"Original FPS: {fps:.2f}")
print(f"Resolution: {width}x{height}")

# Get total frames
probe_result = subprocess.run(
    ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', 
     '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', input_video],
    capture_output=True, text=True
)
try:
    total_video_frames = int(probe_result.stdout.strip())
except:
    total_video_frames = 1000

print(f"Total frames in video: {total_video_frames}")
print(f"Extracting frames {START_FRAME} to {total_video_frames} (total: {total_video_frames - START_FRAME + 1} frames)")

# Extract frames from START_FRAME to end - frames will be numbered 0, 1, 2, ...
temp_dir = tempfile.mkdtemp()
subprocess.run([
    'ffmpeg', '-err_detect', 'ignore_err', '-i', input_video,
    '-vsync', 'cfr', '-r', '30', 
    '-start_number', '0',
    '-frames:v', str(total_video_frames - START_FRAME + 1),
    '-skip_frame', 'nokey',
    '-vsync', '0',
    f'{temp_dir}/frame_%05d.jpg',
    '-hide_banner', '-loglevel', 'error', '-y'
], capture_output=True)

# Alternative: Use seek to start from specific frame
temp_dir = tempfile.mkdtemp()
subprocess.run([
    'ffmpeg', '-err_detect', 'ignore_err', 
    '-ss', f'{START_FRAME/30:.2f}',  # Seek to time
    '-i', input_video,
    '-vsync', 'cfr', '-r', '30', 
    '-start_number', '0',
    '-frames:v', str(total_video_frames - START_FRAME + 1),
    f'{temp_dir}/frame_%05d.jpg',
    '-hide_banner', '-loglevel', 'error', '-y'
])

frames = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
print(f"Extracted {len(frames)} frames (numbered 0 to {len(frames)-1})")

# Load frames
frame_list = []
for f in frames:
    img = cv2.imread(os.path.join(temp_dir, f))
    if img is not None:
        frame_list.append(img)

print(f"Loaded {len(frame_list)} frames")

# Save repaired video at 30 FPS (starts from frame 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(repaired_path, fourcc, 30, (width, height))
for frame in frame_list:
    writer.write(frame)
writer.release()

repair_time = time.time() - start_repair
print(f"Repaired video saved: {repaired_path} (30 FPS, starts at frame 0)")
shutil.rmtree(temp_dir, ignore_errors=True)

print("\n" + "=" * 50)
print("STEP 2: SAM2 Segmentation (Border Extraction)")
print("=" * 50)

# Load SAM2
print("Loading SAM2 on GPU...")
torch.set_grad_enabled(False)
predictor = SAM2VideoPredictor.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

total_frames = len(frame_list)

# Data structures
border_pixels_data = {}
timing_data = {'frame_times': [], 'gpu_memory_usage': []}

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

def extract_border_pixels(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], None
    
    all_border_points = []
    for contour in contours:
        points = contour.reshape(-1, 2).tolist()
        all_border_points.extend(points)
    
    return all_border_points, contours[0] if contours else None

# Create temp directory for SAM2
temp_sam2 = tempfile.mkdtemp()
for i, frame in enumerate(frame_list):
    cv2.imwrite(os.path.join(temp_sam2, f"{i:05d}.jpg"), frame)

state = predictor.init_state(video_path=temp_sam2)

# Click selection
click_point = None
tracking = False

def mouse_cb(event, x, y, flags, param):
    global click_point, state
    if event == cv2.EVENT_LBUTTONDOWN and not tracking:
        click_point = (x, y)
        print(f"Clicked at ({x}, {y})")
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            points=np.array([click_point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32)
        )
        print("Press 't' to start tracking")

cv2.namedWindow('SAM2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SAM2', 1280, 720)
cv2.setMouseCallback('SAM2', mouse_cb)

frame0 = frame_list[0].copy()
cv2.putText(frame0, "CLICK ON OBJECT TO SEGMENT", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.putText(frame0, f"Frames: Original {START_FRAME} to {START_FRAME + total_frames - 1}", (50, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.imshow('SAM2', frame0)

while not tracking:
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        out_writer.release()
        cv2.destroyAllWindows()
        shutil.rmtree(temp_sam2)
        exit(0)
    elif key == ord('t') and click_point is not None:
        tracking = True
        print("\nTracking started!")

# Process frames - idx is the frame number in repaired video
for idx in range(total_frames):
    frame = frame_list[idx].copy()
    actual_frame_num = START_FRAME + idx
    
    start_inference = time.time()
    
    try:
        for out_idx, obj_ids, masks in predictor.propagate_in_video(
            state, start_frame_idx=idx, max_frame_num_to_track=1
        ):
            if out_idx == idx and len(masks) > 0:
                mask = (masks[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
                
                border_points, contour_for_viz = extract_border_pixels(mask)
                border_pixels_data[actual_frame_num] = border_points
                
                inference_time = time.time() - start_inference
                timing_data['frame_times'].append(inference_time)
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    timing_data['gpu_memory_usage'].append(memory_used)
                else:
                    timing_data['gpu_memory_usage'].append(0)
                
                # Light blue overlay
                frame[mask == 1] = frame[mask == 1] * 0.5 + np.array([255, 255, 0]) * 0.5
                
                # Yellow border
                if contour_for_viz is not None:
                    cv2.drawContours(frame, [contour_for_viz], -1, (0, 255, 255), 3)
                break
        else:
            border_pixels_data[actual_frame_num] = []
            inference_time = time.time() - start_inference
            timing_data['frame_times'].append(inference_time)
            timing_data['gpu_memory_usage'].append(0)
            
    except Exception as e:
        print(f"Error on frame {actual_frame_num}: {e}")
        border_pixels_data[actual_frame_num] = []
        inference_time = time.time() - start_inference
        timing_data['frame_times'].append(inference_time)
        timing_data['gpu_memory_usage'].append(0)
    
    # Display ONLY current frame number (actual original frame number)
    cv2.putText(frame, f"{actual_frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out_writer.write(frame)
    cv2.imshow('SAM2', frame)
    
    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx+1}/{total_frames} (Original frame {actual_frame_num})")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

out_writer.release()
cv2.destroyAllWindows()
shutil.rmtree(temp_sam2, ignore_errors=True)
torch.cuda.empty_cache()

sam2_total_time = time.time() - start_repair

# ============================================================
# SAVE JSON
# ============================================================
print("\nSaving JSON...")
json_output = {
    'video_info': {
        'width': width,
        'height': height,
        'fps': 30,
        'start_frame': START_FRAME,
        'total_frames': total_frames
    },
    'border_pixels': {}
}
for frame_idx, coords in border_pixels_data.items():
    json_output['border_pixels'][str(frame_idx)] = coords if coords else []

with open(output_json_path, 'w') as f:
    json.dump(json_output, f)
print(f"JSON saved: {output_json_path}")

# ============================================================
# SAVE TEXT FILES
# ============================================================
print("\nSaving text files...")

time_txt = os.path.join(txt_path, "processing_time_per_frame.txt")
with open(time_txt, 'w') as f:
    f.write("Frame,Time(seconds)\n")
    for i, t in enumerate(timing_data['frame_times']):
        f.write(f"{START_FRAME + i},{t:.6f}\n")
print(f"  ✅ {time_txt}")

gpu_txt = os.path.join(txt_path, "gpu_memory_usage.txt")
with open(gpu_txt, 'w') as f:
    f.write("Frame,GPU_Memory(GB)\n")
    for i, mem in enumerate(timing_data['gpu_memory_usage']):
        f.write(f"{START_FRAME + i},{mem:.6f}\n")
print(f"  ✅ {gpu_txt}")

# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - start_total

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"✅ Repaired video (30 FPS, starts at frame 0): {repaired_path}")
print(f"✅ Annotated video (border YELLOW, mask LIGHT BLUE): {output_video_path}")
print(f"✅ JSON (border pixels): {output_json_path}")
print(f"\n📊 Text files saved in: {txt_path}")
print(f"  - processing_time_per_frame.txt")
print(f"  - gpu_memory_usage.txt")
print(f"\nStatistics:")
print(f"  Original frames: {START_FRAME} to {START_FRAME + total_frames - 1}")
print(f"  Total frames processed: {total_frames}")
print(f"  Total inference time: {sum(timing_data['frame_times']):.2f} seconds")
print(f"  Average time per frame: {np.mean(timing_data['frame_times']):.3f} seconds")
print(f"  Peak GPU memory: {max(timing_data['gpu_memory_usage']):.3f} GB")
print(f"  Frames with border detected: {sum(1 for v in border_pixels_data.values() if v)}")
print("=" * 50)