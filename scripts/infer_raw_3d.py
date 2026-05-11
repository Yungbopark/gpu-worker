import os
import sys
import glob
import time
import subprocess
import argparse
import threading
from datetime import datetime

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


def pick_latest_video():
    vids = glob.glob(os.path.join(UPLOAD_DIR, "*.mp4"))
    vids += glob.glob(os.path.join(UPLOAD_DIR, "*.mov"))

    if not vids:
        raise RuntimeError("No video found in upload_inbox")

    vids.sort(key=os.path.getmtime, reverse=True)
    return vids[0]


def count_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png")

    total = 0
    for e in exts:
        total += len(glob.glob(os.path.join(folder, e)))

    return total


# =========================================================
# 🔥 Single Person 필터
# =========================================================

def keep_single_person(folder):
    files = glob.glob(os.path.join(folder, "*.png"))

    grouped = {}

    for f in files:
        name = os.path.basename(f)
        frame_id = name.split("_")[0]
        grouped.setdefault(frame_id, []).append(f)

    for frame_id, file_list in grouped.items():

        # 가장 간단한 방식: _0만 남김
        keep = [f for f in file_list if "_0.png" in f]

        if not keep:
            continue

        for f in file_list:
            if f not in keep:
                os.remove(f)


# =========================================================
# GPU Monitor
# =========================================================

class GPUMonitor:

    def __init__(self, interval=2.0):
        self.interval = interval
        self.stop_signal = False

    def query(self):
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"

        try:
            output = subprocess.check_output(
                ["bash", "-lc", cmd],
                text=True
            ).strip()

            util, mem = output.split(",")

            return f"util={util.strip()}% mem={mem.strip()}MiB"

        except:
            return "nvidia-smi error"

    def run(self):
        while not self.stop_signal:
            print("[GPU]", self.query())
            time.sleep(self.interval)

    def start(self):
        self.stop_signal = False
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        self.thread = t

    def stop(self):
        self.stop_signal = True


# =========================================================
# Frame Extraction
# =========================================================

def extract_frames(video_file, fps, max_frames=None, start_sec=None, end_sec=None):

    clear_folder(IMG_FOLDER)

    vf = f"fps={fps}"

    if max_frames:
        vf += f",select='lt(n\\,{max_frames})'"

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

    t0 = time.time()

    r = subprocess.run(cmd, capture_output=True, text=True)

    dt = time.time() - t0

    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("ffmpeg failed")

    n = count_images(IMG_FOLDER)

    print(f"[OK] Extracted {n} frames ({dt:.1f}s)")

    if n == 0:
        raise RuntimeError("No frames extracted")


# =========================================================
# 4D Humans Run
# =========================================================

def run_4dhumans(out_folder):

    ensure_dir(out_folder)

    env = os.environ.copy()

    env["PYTHONPATH"] = (
        f"{BASE_DIR}/chumpy:"
        f"{BASE_DIR}/detectron2:"
        f"{REPO}:"
        + env.get("PYTHONPATH", "")
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
        raise RuntimeError(f"demo.py failed (exit={rc})")


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

    args = parser.parse_args()

    start_sec = args.start if args.start is not None else args.start_sec
    end_sec = args.end if args.end is not None else args.end_sec

    if args.video:
        video_file = args.video

        if not os.path.exists(video_file):
            raise RuntimeError(f"Video not found: {video_file}")
    else:
        video_file = pick_latest_video()

    print("[INFO] Video:", video_file)
    print("[INFO] start_sec:", start_sec)
    print("[INFO] end_sec:", end_sec)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_folder = os.path.join(OUT_BASE, f"raw_{run_id}")

    ensure_dir(out_folder)

    extract_frames(
        video_file,
        fps=args.fps,
        max_frames=args.max_frames,
        start_sec=start_sec,
        end_sec=end_sec
    )

    run_4dhumans(out_folder)

    # 🔥 여기 추가
    keep_single_person(out_folder)

    print("\n==============================")
    print("Inference finished")
    print("Output folder:", out_folder)
    print("==============================")


# =========================================================

if __name__ == "__main__":
    main()