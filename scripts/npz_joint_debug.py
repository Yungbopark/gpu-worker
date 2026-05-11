import numpy as np
import os
import glob
import matplotlib.pyplot as plt

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"


# ---------------------------------------------------
# npz 찾기
# ---------------------------------------------------
def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    for d in raw_dirs:
        path = os.path.join(d, "motion_fixed_v5_safe.npz")
        if os.path.exists(path):
            return path

    raise RuntimeError("npz not found")


# ---------------------------------------------------
# load
# ---------------------------------------------------
def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data


# ---------------------------------------------------
# 전체 관절 pose 시각화
# ---------------------------------------------------
def plot_all_joint_pose(body_pose):

    frames = np.arange(body_pose.shape[0])

    # rotmat → scalar로 단순화 (norm 사용)
    pose_value = np.linalg.norm(body_pose, axis=(2,3))  # (N, 23)

    plt.figure(figsize=(18, 10))

    for j in range(pose_value.shape[1]):
        plt.plot(frames, pose_value[:, j], alpha=0.3)

    plt.title("All Joint Pose (Rotation Magnitude)")
    plt.xlabel("Frame")
    plt.ylabel("Rotation magnitude")
    plt.grid()

    save_path = os.path.join(OUTPUT_DIR, "all_joint_pose.png")
    plt.savefig(save_path)

    print("[SAVED]", save_path)


# ---------------------------------------------------
# min/max envelope
# ---------------------------------------------------
def plot_pose_envelope(body_pose):

    frames = np.arange(body_pose.shape[0])
    pose_value = np.linalg.norm(body_pose, axis=(2,3))

    min_line = pose_value.min(axis=1)
    max_line = pose_value.max(axis=1)

    plt.figure(figsize=(18, 10))

    plt.plot(frames, min_line, label="min_joint", color="blue")
    plt.plot(frames, max_line, label="max_joint", color="red")

    plt.title("Joint Pose Envelope")
    plt.xlabel("Frame")
    plt.ylabel("Rotation magnitude")
    plt.legend()
    plt.grid()

    save_path = os.path.join(OUTPUT_DIR, "joint_pose_envelope.png")
    plt.savefig(save_path)

    print("[SAVED]", save_path)


# ---------------------------------------------------
# outlier detection
# ---------------------------------------------------
def detect_outliers(body_pose):

    pose_value = np.linalg.norm(body_pose, axis=(2,3))
    max_line = pose_value.max(axis=1)

    median = np.median(max_line)
    std = np.std(max_line)

    threshold = std * 2

    mask = np.abs(max_line - median) > threshold

    print("\n[INFO] ===== OUTLIERS =====")
    print("threshold:", threshold)
    print("count:", np.sum(mask))
    print("frames:", np.where(mask)[0][:30])
    print("[INFO] =====================\n")


# ---------------------------------------------------
# main
# ---------------------------------------------------
def main():

    path = find_latest_npz()
    print("[INFO] NPZ:", path)

    data = load_npz(path)

    body_pose = data["body_pose"]

    plot_all_joint_pose(body_pose)
    plot_pose_envelope(body_pose)
    detect_outliers(body_pose)


if __name__ == "__main__":
    main()