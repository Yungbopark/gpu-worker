import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

# =========================================================
# Python path (required for 4D-Humans stack)
# =========================================================

sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")
sys.path.insert(0, "/home/yungbopark/gpu-worker/4D-Humans")
sys.path.insert(0, "/home/yungbopark/gpu-worker/detectron2")

# =========================================================
# Paths
# =========================================================

BASE_DIR = "/home/yungbopark/gpu-worker"

REPO = f"{BASE_DIR}/4D-Humans"
UPLOAD_DIR = f"{BASE_DIR}/upload_inbox"
IMG_FOLDER = f"{BASE_DIR}/temp_frames"
OUT_BASE = f"{BASE_DIR}/output"

PY = sys.executable
DEFAULT_SMOOTH_WINDOW = 3

# =========================================================
# Joint mapping
# =========================================================

# SMPL body-24 joint names. The raw SMPL output usually contains 45 joints,
# but the first 24 are the stable body joints needed for downstream motion JSON.
BODY24_INDICES = list(range(24))
BODY24_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

LEFT_HIP_INDEX = BODY24_NAMES.index("left_hip")
RIGHT_HIP_INDEX = BODY24_NAMES.index("right_hip")


# =========================================================
# Utils
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_folder(path):
    ensure_dir(path)

    for f in glob.glob(os.path.join(path, "*")):
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)


def pick_latest_video():
    vids = glob.glob(os.path.join(UPLOAD_DIR, "*.mp4"))
    vids += glob.glob(os.path.join(UPLOAD_DIR, "*.mov"))

    if not vids:
        raise RuntimeError("No video found in upload_inbox (*.mp4, *.mov)")

    vids.sort(key=os.path.getmtime, reverse=True)
    return vids[0]


def count_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png")

    total = 0
    for e in exts:
        total += len(glob.glob(os.path.join(folder, e)))

    return total


def resolve_segment_args(args):
    start_sec = args.start if args.start is not None else args.start_sec
    end_sec = args.end if args.end is not None else args.end_sec

    if start_sec is not None and start_sec < 0:
        raise RuntimeError("start_sec must be >= 0")
    if end_sec is not None and end_sec < 0:
        raise RuntimeError("end_sec must be >= 0")
    if start_sec is not None and end_sec is not None and end_sec < start_sec:
        raise RuntimeError("end_sec must be greater than or equal to start_sec")

    return start_sec, end_sec


def describe_segment(start_sec, end_sec):
    if start_sec is None and end_sec is None:
        return "full video", None
    if start_sec is None:
        return "0.0s ~ %.3fs" % end_sec, float(end_sec)
    if end_sec is None:
        return "%.3fs ~ end" % start_sec, None
    return "%.3fs ~ %.3fs" % (start_sec, end_sec), float(end_sec - start_sec)


# =========================================================
# Single Person filter
# =========================================================

def keep_single_person(folder):
    files = glob.glob(os.path.join(folder, "*.png"))

    grouped = {}

    for f in files:
        name = os.path.basename(f)
        frame_id = name.split("_")[0]
        grouped.setdefault(frame_id, []).append(f)

    for frame_id, file_list in grouped.items():
        keep = [f for f in file_list if "_0.png" in f]

        if not keep:
            continue

        for f in file_list:
            if f not in keep:
                os.remove(f)


def move_png_outputs(out_folder):
    png_dir = os.path.join(out_folder, "png")
    ensure_dir(png_dir)

    png_files = sorted(glob.glob(os.path.join(out_folder, "*.png")))
    for src in png_files:
        dst = os.path.join(png_dir, os.path.basename(src))
        shutil.move(src, dst)

    print("[INFO] PNG outputs moved to:", png_dir)
    print("[INFO] PNG count:", len(glob.glob(os.path.join(png_dir, "*.png"))))
    return png_dir


# =========================================================
# GPU Monitor
# =========================================================

class GPUMonitor:

    def __init__(self, interval=2.0):
        self.interval = interval
        self.stop_signal = False
        self.thread = None

    def query(self):
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"

        try:
            output = subprocess.check_output(
                ["bash", "-lc", cmd],
                text=True
            ).strip()

            util, mem = output.split(",")
            return "util=%s%% mem=%sMiB" % (util.strip(), mem.strip())
        except Exception:
            return "nvidia-smi error"

    def run(self):
        while not self.stop_signal:
            print("[GPU]", self.query())
            time.sleep(self.interval)

    def start(self):
        self.stop_signal = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_signal = True


# =========================================================
# Frame Extraction
# =========================================================

def extract_frames(video_file, fps, max_frames=None, start_sec=None, end_sec=None):
    clear_folder(IMG_FOLDER)

    vf = "fps=%s" % fps

    if max_frames:
        vf += ",select='lt(n\\,%s)'" % max_frames

    cmd = ["ffmpeg", "-y"]

    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]

    cmd += ["-i", video_file]

    if start_sec is not None and end_sec is not None:
        duration = max(0, end_sec - start_sec)
        cmd += ["-t", str(duration)]
    elif end_sec is not None:
        cmd += ["-to", str(end_sec)]

    cmd += [
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", "2",
        os.path.join(IMG_FOLDER, "%06d.jpg")
    ]

    print("[INFO] Extracting frames...")
    print("[INFO] start_sec:", start_sec)
    print("[INFO] end_sec:", end_sec)
    print("[RUN]", " ".join(cmd))

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("ffmpeg failed with exit code %d" % r.returncode)

    n = count_images(IMG_FOLDER)

    print("[OK] Extracted %d frames (%.1fs)" % (n, dt))

    if n == 0:
        raise RuntimeError("No frames extracted")


# =========================================================
# 4D Humans Run
# =========================================================

def run_4dhumans(out_folder, enable_gpu_monitor=True):
    ensure_dir(out_folder)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        "%s/chumpy:%s/detectron2:%s:%s"
        % (BASE_DIR, BASE_DIR, REPO, env.get("PYTHONPATH", ""))
    )
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    cmd = [
        PY,
        "demo.py",
        "--img_folder", IMG_FOLDER,
        "--out_folder", out_folder
    ]

    print("[RUN]", " ".join(cmd))

    gpu = GPUMonitor()
    if enable_gpu_monitor:
        gpu.start()

    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        print("[4D]", line.rstrip())

    rc = proc.wait()
    gpu.stop()

    if rc != 0:
        raise RuntimeError("demo.py failed (exit=%d)" % rc)


# =========================================================
# Motion NPZ -> joints
# =========================================================

def load_motion_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    motion = {}

    for key in data.files:
        motion[key] = data[key]

    return motion


def find_motion_npz(out_folder):
    npz_path = os.path.join(out_folder, "motion_data.npz")
    if not os.path.exists(npz_path):
        raise RuntimeError("motion_data.npz not found in output folder: %s" % out_folder)
    return npz_path


def filter_motion_single_person(motion):
    if "person_id" not in motion:
        return motion

    person_ids = np.asarray(motion["person_id"]).reshape(-1)
    keep_mask = person_ids == 0

    if not np.any(keep_mask):
        unique_ids = sorted(set(person_ids.tolist()))
        raise RuntimeError("No person_id=0 found in motion_data.npz. Available ids: %s" % unique_ids)

    filtered = {}
    for key, value in motion.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == keep_mask.shape[0]:
            filtered[key] = arr[keep_mask]
        else:
            filtered[key] = value

    print("[INFO] Single-person filter applied to npz: kept %d / %d rows" % (int(keep_mask.sum()), len(keep_mask)))
    return filtered


def sort_motion_by_frame(motion):
    if "img_name" not in motion:
        return motion

    img_names = np.asarray(motion["img_name"])
    order = np.argsort(img_names.astype(str), kind="stable")

    sorted_motion = {}
    for key, value in motion.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == order.shape[0]:
            sorted_motion[key] = arr[order]
        else:
            sorted_motion[key] = value

    return sorted_motion


def load_joints3d(motion):
    if "joints3d" not in motion:
        raise RuntimeError(
            "motion_data.npz does not contain joints3d. "
            "Re-run inference with updated demo.py or migrate this file."
        )

    joints3d = np.asarray(motion["joints3d"], dtype=np.float32)

    if joints3d.ndim != 3 or joints3d.shape[2] != 3:
        raise RuntimeError(
            "joints3d must have shape (detections, joints, 3), got %s"
            % (joints3d.shape,)
        )

    if "person_id" not in motion:
        raise RuntimeError("motion_data.npz does not contain person_id")
    if "img_name" not in motion:
        raise RuntimeError("motion_data.npz does not contain img_name")

    if joints3d.shape[0] != len(motion["person_id"]):
        raise RuntimeError(
            "joints3d/person_id row mismatch: %d != %d"
            % (joints3d.shape[0], len(motion["person_id"]))
        )
    if joints3d.shape[0] != len(motion["img_name"]):
        raise RuntimeError(
            "joints3d/img_name row mismatch: %d != %d"
            % (joints3d.shape[0], len(motion["img_name"]))
        )

    print("[INFO] Using joints3d from npz")
    print("[INFO] joints3d shape:", joints3d.shape)

    return joints3d


def select_body24_joints(joints3d):
    joints = np.asarray(joints3d, dtype=np.float32)

    if joints.ndim != 3 or joints.shape[2] != 3:
        raise RuntimeError("joints3d must have shape (frames, joints, 3), got %s" % (joints.shape,))

    if joints.shape[1] < len(BODY24_INDICES):
        raise RuntimeError("Not enough joints for BODY24 subset")

    body24 = joints[:, BODY24_INDICES, :]
    print("[INFO] Using SMPL body-24 subset for motion.json")
    return body24


def moving_average_joints(joints_xy, window):
    joints = np.asarray(joints_xy, dtype=np.float32)

    if window <= 1:
        return joints.copy()

    if window < 1:
        raise RuntimeError("smooth_window must be >= 1")

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(joints, ((pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
    smoothed = np.zeros_like(joints, dtype=np.float32)

    for frame_idx in range(joints.shape[0]):
        smoothed[frame_idx] = padded[frame_idx:frame_idx + window].mean(axis=0)

    return smoothed


def build_coordinate_views(joints_xy):
    raw_xy = np.asarray(joints_xy, dtype=np.float32).copy()
    raw_xy[:, :, 1] *= -1.0

    hip_center = (
        raw_xy[:, LEFT_HIP_INDEX, :] + raw_xy[:, RIGHT_HIP_INDEX, :]
    ) * 0.5
    relative_xy = raw_xy - hip_center[:, None, :]

    min_xy = raw_xy.reshape(-1, 2).min(axis=0)
    max_xy = raw_xy.reshape(-1, 2).max(axis=0)
    size_xy = max_xy - min_xy
    size_xy[size_xy < 1e-8] = 1.0
    normalized_xy = (raw_xy - min_xy[None, None, :]) / size_xy[None, None, :]
    normalized_xy = np.clip(normalized_xy, 0.0, 1.0)

    print("[INFO] raw_xy min:", min_xy.tolist())
    print("[INFO] raw_xy max:", max_xy.tolist())
    print("[INFO] hip_center first frame:", hip_center[0].tolist())

    return {
        "raw_xy": raw_xy.astype(np.float32),
        "normalized_xy": normalized_xy.astype(np.float32),
        "relative_xy": relative_xy.astype(np.float32),
        "hip_center_xy": hip_center.astype(np.float32),
        "normalize_min_xy": min_xy.astype(np.float32),
        "normalize_max_xy": max_xy.astype(np.float32),
    }


def build_motion_json(coordinates, motion, fps_value):
    raw_xy = coordinates["raw_xy"]
    normalized_xy = coordinates["normalized_xy"]
    relative_xy = coordinates["relative_xy"]
    hip_center_xy = coordinates["hip_center_xy"]
    frame_count = int(raw_xy.shape[0])
    frames = []

    for idx in range(frame_count):
        joints_dict = {}
        for joint_idx, joint_name in enumerate(BODY24_NAMES):
            raw = raw_xy[idx, joint_idx]
            normalized = normalized_xy[idx, joint_idx]
            relative = relative_xy[idx, joint_idx]
            joints_dict[joint_name] = {
                "raw": [float(raw[0]), float(raw[1])],
                "normalized": [float(normalized[0]), float(normalized[1])],
                "relative": [float(relative[0]), float(relative[1])],
            }

        frames.append({
            "frame_index": int(idx),
            "hip_center": [float(hip_center_xy[idx, 0]), float(hip_center_xy[idx, 1])],
            "joints": joints_dict,
        })

    return {
        "meta": {
            "fps": float(fps_value),
            "frame_count": frame_count,
        },
        "frames": frames,
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] Wrote:", path)


def validate_output_counts(out_folder, motion_frame_count):
    png_dir = os.path.join(out_folder, "png")
    png_count = len(glob.glob(os.path.join(png_dir, "*.png")))

    print("[INFO] motion.json frame count:", motion_frame_count)
    print("[INFO] PNG count:", png_count)

    if png_count != 0 and png_count != motion_frame_count:
        print("[WARN] PNG count and motion frame count differ: %d vs %d" % (png_count, motion_frame_count))
        print("[WARN] Possible cause: 4DHumans dropped detections on some frames or PNG naming/filtering differs from npz rows.")


def create_motion_json(
    out_folder,
    fps_value,
    video_file,
    smooth_window,
    start_sec,
    end_sec,
):
    npz_path = find_motion_npz(out_folder)
    motion = load_motion_npz(npz_path)
    motion = filter_motion_single_person(motion)
    motion = sort_motion_by_frame(motion)

    joints3d = load_joints3d(motion)
    body24 = select_body24_joints(joints3d)
    smoothed_xy = moving_average_joints(body24[:, :, :2], smooth_window)
    coordinates = build_coordinate_views(smoothed_xy)

    motion_json = build_motion_json(coordinates, motion, fps_value)
    motion_json_path = os.path.join(out_folder, "motion.json")
    write_json(motion_json_path, motion_json)

    meta = {
        "video_file": video_file,
        "motion_npz": npz_path,
        "fps": float(fps_value),
        "frame_count": int(coordinates["raw_xy"].shape[0]),
        "segment": {
            "start_sec": None if start_sec is None else float(start_sec),
            "end_sec": None if end_sec is None else float(end_sec),
        },
        "joint_count": len(BODY24_NAMES),
        "joint_names": BODY24_NAMES,
        "joint_source": "joints3d",
        "joint_source_shape": list(joints3d.shape),
        "joint_subset": "SMPL body-24",
        "coordinate_views": ["raw", "normalized", "relative"],
        "relative_reference": "hip_center = average(left_hip, right_hip)",
        "smoothing": {
            "method": "moving_average",
            "window": int(smooth_window),
        },
        "normalize_min_xy": [
            float(coordinates["normalize_min_xy"][0]),
            float(coordinates["normalize_min_xy"][1]),
        ],
        "normalize_max_xy": [
            float(coordinates["normalize_max_xy"][0]),
            float(coordinates["normalize_max_xy"][1]),
        ],
    }
    meta_path = os.path.join(out_folder, "meta.json")
    write_json(meta_path, meta)

    print("[INFO] motion.json joint count:", len(BODY24_NAMES))
    validate_output_counts(out_folder, motion_json["meta"]["frame_count"])


# =========================================================
# SwiftUI output contract postprocess
# =========================================================

def safe_postprocess_step(label, fn):
    try:
        fn()
    except Exception as exc:
        print("[WARN] %s failed: %s" % (label, exc))


def count_png_frames_for_contract(out_folder):
    return len(list_contract_png_frames(out_folder))


def list_contract_png_frames(out_folder):
    png_dir = os.path.join(out_folder, "png")
    return sorted(glob.glob(os.path.join(png_dir, "*.png")))


def read_motion_frame_count(out_folder):
    motion_json_path = os.path.join(out_folder, "motion.json")
    if not os.path.exists(motion_json_path):
        return None

    with open(motion_json_path, "r", encoding="utf-8") as f:
        motion_json = json.load(f)

    meta = motion_json.get("meta", {})
    frame_count = meta.get("frame_count")
    if frame_count is None:
        return None

    return int(frame_count)


def read_motion_npz_fps(out_folder):
    npz_path = os.path.join(out_folder, "motion_data.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    if "fps" not in data.files:
        return None

    fps_arr = np.asarray(data["fps"]).reshape(-1)
    if fps_arr.size == 0:
        return None

    fps_value = float(fps_arr[0])
    if fps_value <= 0:
        return None

    return fps_value


def read_motion_json_fps(out_folder):
    motion_json_path = os.path.join(out_folder, "motion.json")
    if not os.path.exists(motion_json_path):
        return None

    with open(motion_json_path, "r", encoding="utf-8") as f:
        motion_json = json.load(f)

    fps_value = motion_json.get("meta", {}).get("fps")
    if fps_value is None:
        return None

    fps_value = float(fps_value)
    if fps_value <= 0:
        return None

    return fps_value


def resolve_contract_fps(out_folder, fallback_fps):
    for reader in (read_motion_npz_fps, read_motion_json_fps):
        try:
            fps_value = reader(out_folder)
        except Exception as exc:
            print("[WARN] fps metadata read failed: %s" % exc)
            fps_value = None

        if fps_value is not None:
            return fps_value

    return float(fallback_fps)


def ffmpeg_concat_escape(path):
    return path.replace("'", "'\\''")


def write_ffmpeg_concat_list(out_folder, png_frames, fps_value):
    list_path = os.path.join(out_folder, ".swiftui_frames.txt")
    frame_duration = 1.0 / float(fps_value)

    with open(list_path, "w", encoding="utf-8") as f:
        for frame_path in png_frames:
            f.write("file '%s'\n" % ffmpeg_concat_escape(os.path.abspath(frame_path)))
            f.write("duration %.10f\n" % frame_duration)

        # ffmpeg concat demuxer needs the last file repeated for the final duration.
        f.write("file '%s'\n" % ffmpeg_concat_escape(os.path.abspath(png_frames[-1])))

    return list_path


def create_output_mp4(out_folder, fps_value):
    png_frames = list_contract_png_frames(out_folder)
    if not png_frames:
        raise RuntimeError("No PNG frames found for output.mp4")

    output_mp4 = os.path.join(out_folder, "output.mp4")
    list_path = write_ffmpeg_concat_list(out_folder, png_frames, fps_value)

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_mp4,
    ]

    print("[INFO] Creating SwiftUI output video...")
    print("[INFO] PNG frame count:", len(png_frames))
    print("[INFO] First PNG frame:", png_frames[0])
    print("[RUN]", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr)
            raise RuntimeError("ffmpeg output.mp4 failed with exit code %d" % result.returncode)
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

    print("[OK] Wrote:", output_mp4)


def create_thumbnail(out_folder):
    png_frames = list_contract_png_frames(out_folder)
    if not png_frames:
        print("[WARN] thumbnail source frame not found in:", os.path.join(out_folder, "png"))
        return

    source = png_frames[0]
    target = os.path.join(out_folder, "thumbnail.jpg")

    shutil.copyfile(source, target)
    print("[INFO] Thumbnail source:", source)
    print("[OK] Wrote:", target)


def create_analysis_json(out_folder, fps_value):
    result_id = os.path.basename(os.path.abspath(out_folder))

    frame_count = read_motion_frame_count(out_folder)
    if frame_count is None:
        frame_count = count_png_frames_for_contract(out_folder)

    fps_float = float(fps_value)
    duration_sec = float(frame_count) / fps_float if fps_float > 0 else 0.0

    analysis = {
        "result_id": result_id,
        "status": "completed",
        "fps": fps_float,
        "frame_count": int(frame_count),
        "duration_sec": duration_sec,
        "exercise_type": "squat",
        "files": {
            "video": "output.mp4",
            "thumbnail": "thumbnail.jpg",
        },
        "motion": {
            "motion_json": "motion.json",
        },
    }

    analysis_path = os.path.join(out_folder, "analysis.json")
    write_json(analysis_path, analysis)


def create_swiftui_output_contract(out_folder, fps_value):
    contract_fps = resolve_contract_fps(out_folder, fps_value)
    print("[INFO] SwiftUI contract FPS:", contract_fps)

    safe_postprocess_step(
        "output.mp4 generation",
        lambda: create_output_mp4(out_folder, contract_fps),
    )
    safe_postprocess_step(
        "thumbnail.jpg generation",
        lambda: create_thumbnail(out_folder),
    )
    safe_postprocess_step(
        "analysis.json generation",
        lambda: create_analysis_json(out_folder, contract_fps),
    )


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)

    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)

    parser.add_argument("--smooth_window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--disable_gpu_monitor", action="store_true")

    args = parser.parse_args()

    start_sec, end_sec = resolve_segment_args(args)
    segment_label, effective_duration = describe_segment(start_sec, end_sec)

    if args.video:
        video_file = args.video

        if not os.path.exists(video_file):
            raise RuntimeError("Video not found: %s" % video_file)
    else:
        video_file = pick_latest_video()

    print("[INFO] Video:", video_file)
    print("[INFO] start_sec:", start_sec)
    print("[INFO] end_sec:", end_sec)
    print("[INFO] fps:", args.fps)
    print("[INFO] Using segment:", segment_label)
    if effective_duration is None:
        print("[INFO] Effective duration: open-ended")
    else:
        print("[INFO] Effective duration: %.3fs" % effective_duration)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(OUT_BASE, "raw_%s" % run_id)

    ensure_dir(out_folder)

    extract_frames(
        video_file,
        fps=args.fps,
        max_frames=args.max_frames,
        start_sec=start_sec,
        end_sec=end_sec
    )

    run_4dhumans(
        out_folder,
        enable_gpu_monitor=(not args.disable_gpu_monitor)
    )

    keep_single_person(out_folder)
    move_png_outputs(out_folder)

    create_motion_json(
        out_folder=out_folder,
        fps_value=args.fps,
        video_file=video_file,
        smooth_window=args.smooth_window,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    create_swiftui_output_contract(
        out_folder=out_folder,
        fps_value=args.fps,
    )

    print("\n==============================")
    print("Initial inference finished")
    print("Output folder:", out_folder)
    print("==============================")


# =========================================================

if __name__ == "__main__":
    main()
