import os
import glob
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MAX_ROT_DEG_PER_FRAME = 10.0

SMOOTH_TRANSL_XZ = 5
SMOOTH_BODY = 3

# XZ에서 너무 큰 이동 스파이크 제거용
VELOCITY_CLAMP_XZ = 2.0

# Y drift 제거용
# 큰 값일수록 "느린 드리프트"만 제거하고,
# 작은 값일수록 상하 움직임도 죽을 수 있음
Y_BASELINE_WINDOW = 61

# baseline 제거 강도 (0~1)
# 1.0이면 baseline을 전부 제거
# 0.6~0.85 정도가 보통 더 자연스러움
Y_BASELINE_REMOVE_RATIO = 0.75

# frame0 기준 상대좌표로 둘지 여부
USE_RELATIVE_FRAME0 = True


# ---------------------------------------------------
# IO
# ---------------------------------------------------

def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )

    preferred = [
        "motion_fixed_v6_groundlock.npz",
        "motion_fixed_v5_safe.npz",
        "motion_fixed_v4.npz",
        "motion_data.npz",
    ]

    for d in raw_dirs:
        for name in preferred:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p

    raise RuntimeError("No motion npz found")


def load_motion(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    motion = {
        "global_orient": d["global_orient"],
        "body_pose": d["body_pose"],
        "betas": d["betas"],
        "transl": d["transl"],
        "fps": int(d["fps"][0]) if "fps" in d else 30,
    }

    print("[INFO] loaded:", npz_path)
    print("[INFO] global_orient:", motion["global_orient"].shape)
    print("[INFO] body_pose    :", motion["body_pose"].shape)
    print("[INFO] betas        :", motion["betas"].shape)
    print("[INFO] transl       :", motion["transl"].shape)
    print("[INFO] fps          :", motion["fps"])
    return motion


def save_motion_npz(out_path, motion):
    np.savez_compressed(
        out_path,
        global_orient=motion["global_orient"],
        body_pose=motion["body_pose"],
        betas=motion["betas"],
        transl=motion["transl"],
        fps=np.array([motion["fps"]], dtype=np.int32),
    )
    print("[SAVE]", out_path)


# ---------------------------------------------------
# Math / utils
# ---------------------------------------------------

def moving_average_1d(x, window):
    if window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def moving_average_2d(x, window):
    if window <= 1:
        return x.copy()
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[1]):
        out[:, i] = moving_average_1d(x[:, i], window)
    return out


def velocity_norm(x):
    out = np.zeros(x.shape[0], dtype=np.float32)
    if x.shape[0] > 1:
        out[1:] = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return out


def rotation_delta_deg(rotmats):
    n = rotmats.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        rel = rotmats[i - 1].T @ rotmats[i]
        out[i] = np.rad2deg(R.from_matrix(rel).magnitude())
    return out


# ---------------------------------------------------
# FIX 1: root rotation stabilization
# ---------------------------------------------------

def stabilize_root_rotation(global_orient, max_rot_deg_per_frame=10.0):
    g = global_orient.copy().astype(np.float32)
    root = g[:, 0].copy()

    before = rotation_delta_deg(root)

    for i in range(1, len(root)):
        rel = root[i - 1].T @ root[i]
        angle = np.rad2deg(R.from_matrix(rel).magnitude())

        if angle > max_rot_deg_per_frame:
            root[i] = root[i - 1]

    after = rotation_delta_deg(root)
    g[:, 0] = root

    summary = {
        "before_root_rot_mean": float(np.mean(before)),
        "before_root_rot_max": float(np.max(before)),
        "after_root_rot_mean": float(np.mean(after)),
        "after_root_rot_max": float(np.max(after)),
    }

    return g.astype(np.float32), summary


# ---------------------------------------------------
# FIX 2: body pose light smoothing
# ---------------------------------------------------

def smooth_body_pose(body_pose, window=3):
    b = body_pose.copy().astype(np.float32)
    n = b.shape[0]
    flat = b.reshape(n, -1)
    flat = moving_average_2d(flat, window)
    return flat.reshape(b.shape).astype(np.float32)


# ---------------------------------------------------
# FIX 3: transl X/Z stabilization
# ---------------------------------------------------

def stabilize_transl_xz(transl, smooth_window=5, velocity_clamp_xz=2.0):
    t = transl.copy().astype(np.float32)

    if USE_RELATIVE_FRAME0:
        t = t - t[0:1]

    xz = t[:, [0, 2]].copy()
    vel_before = velocity_norm(xz)

    spike = vel_before > velocity_clamp_xz
    for i in range(1, len(xz)):
        if spike[i]:
            xz[i] = xz[i - 1]

    xz = moving_average_2d(xz, smooth_window)

    t[:, 0] = xz[:, 0]
    t[:, 2] = xz[:, 1]

    vel_after = velocity_norm(t[:, [0, 2]])

    summary = {
        "xz_vel_before_mean": float(np.mean(vel_before)),
        "xz_vel_before_max": float(np.max(vel_before)),
        "xz_vel_after_mean": float(np.mean(vel_after)),
        "xz_vel_after_max": float(np.max(vel_after)),
        "xz_spike_count": int(np.sum(spike)),
    }

    return t.astype(np.float32), summary


# ---------------------------------------------------
# FIX 4: transl Y drift removal (핵심)
# ---------------------------------------------------

def stabilize_transl_y_drift(
    transl,
    baseline_window=61,
    remove_ratio=0.75,
):
    """
    완전 고정이 아니라,
    느린 Y drift만 제거하고 빠른 상하 움직임은 남긴다.
    """
    t = transl.copy().astype(np.float32)
    y = t[:, 1].copy()

    # 느린 baseline 추정
    baseline = moving_average_1d(y, baseline_window).astype(np.float32)

    # frame0 기준 baseline offset
    baseline_rel = baseline - baseline[0]

    # baseline 일부만 제거
    y_fixed = y - (baseline_rel * remove_ratio)

    t[:, 1] = y_fixed

    summary = {
        "ground_mode": "y_drift_removal",
        "baseline_window": int(baseline_window),
        "remove_ratio": float(remove_ratio),
        "y_before_min": float(np.min(y)),
        "y_before_max": float(np.max(y)),
        "y_before_std": float(np.std(y)),
        "y_after_min": float(np.min(y_fixed)),
        "y_after_max": float(np.max(y_fixed)),
        "y_after_std": float(np.std(y_fixed)),
        "baseline_rel_min": float(np.min(baseline_rel)),
        "baseline_rel_max": float(np.max(baseline_rel)),
    }

    return t.astype(np.float32), summary


# ---------------------------------------------------
# Main fix
# ---------------------------------------------------

def fix_motion_v7(motion):
    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"].copy(),
        "transl": motion["transl"].copy(),
        "fps": motion["fps"],
    }

    fixed["global_orient"], rot_summary = stabilize_root_rotation(
        fixed["global_orient"],
        max_rot_deg_per_frame=MAX_ROT_DEG_PER_FRAME,
    )

    fixed["body_pose"] = smooth_body_pose(
        fixed["body_pose"],
        window=SMOOTH_BODY,
    )

    fixed["transl"], xz_summary = stabilize_transl_xz(
        fixed["transl"],
        smooth_window=SMOOTH_TRANSL_XZ,
        velocity_clamp_xz=VELOCITY_CLAMP_XZ,
    )

    fixed["transl"], y_summary = stabilize_transl_y_drift(
        fixed["transl"],
        baseline_window=Y_BASELINE_WINDOW,
        remove_ratio=Y_BASELINE_REMOVE_RATIO,
    )

    summary = {
        "rotation": rot_summary,
        "transl_xz": xz_summary,
        "transl_y": y_summary,
    }

    return fixed, summary


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()
    motion = load_motion(npz_path)

    fixed, summary = fix_motion_v7(motion)

    raw_dir = os.path.dirname(npz_path)
    out_npz = os.path.join(raw_dir, "motion_fixed_v7.npz")
    out_json = os.path.join(raw_dir, "motion_fixed_v7_summary.json")

    save_motion_npz(out_npz, fixed)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[DONE]")


if __name__ == "__main__":
    main()