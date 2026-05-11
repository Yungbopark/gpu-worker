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

# 회전은 v5_safe와 비슷하게 "급회전만 제거"
MAX_ROT_DEG_PER_FRAME = 10.0

# transl X/Z smoothing
SMOOTH_TRANSL_XZ = 5

# body pose smoothing
SMOOTH_BODY = 3

# X/Z 속도 clamp
VELOCITY_CLAMP_XZ = 2.0

# 바닥 고정 방식
# "lock_frame0_y" : transl[:,1]을 첫 프레임과 동일하게 고정 (가장 강함)
# "median_y"      : transl[:,1]을 전체 중앙값으로 고정
# "smooth_y"      : transl[:,1]을 강하게 smoothing만 함 (덜 공격적)
GROUND_LOCK_MODE = "lock_frame0_y"

# smooth_y 모드일 때 사용할 window
SMOOTH_TRANSL_Y = 21


# ---------------------------------------------------
# IO
# ---------------------------------------------------

def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )

    candidates = [
        "motion_fixed_v5_safe.npz",
        "motion_fixed_v4.npz",
        "motion_data.npz",
    ]

    for d in raw_dirs:
        for name in candidates:
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


def save_npz(path, motion):
    np.savez_compressed(
        path,
        global_orient=motion["global_orient"],
        body_pose=motion["body_pose"],
        betas=motion["betas"],
        transl=motion["transl"],
        fps=np.array([motion["fps"]], dtype=np.int32),
    )
    print("[SAVE]", path)


# ---------------------------------------------------
# UTILS
# ---------------------------------------------------

def moving_average_1d(x, w):
    if w <= 1:
        return x.copy()
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, np.ones(w) / w, mode="valid")


def moving_average_2d(x, w):
    if w <= 1:
        return x.copy()
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[1]):
        out[:, i] = moving_average_1d(x[:, i], w)
    return out


def rotation_delta_deg(rotmats):
    n = rotmats.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        rel = rotmats[i - 1].T @ rotmats[i]
        ang = R.from_matrix(rel).magnitude()
        out[i] = np.rad2deg(ang)
    return out


def velocity_norm(x):
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n > 1:
        out[1:] = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return out


# ---------------------------------------------------
# FIX 1: ROOT ROTATION STABILIZATION
# ---------------------------------------------------

def stabilize_root_rotation(global_orient, max_rot_deg_per_frame=10.0):
    """
    global_orient: (N,1,3,3)
    급격한 회전 jump만 제거
    """
    g = global_orient.copy()
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
# FIX 2: BODY POSE LIGHT SMOOTHING
# ---------------------------------------------------

def smooth_body_pose(body_pose, window=3):
    """
    rotmat을 직접 평균내는 건 이상적이지 않을 수 있음.
    그래도 이전 safe 버전과 호환되게 최소 smoothing만 적용.
    """
    b = body_pose.copy()
    n = b.shape[0]
    flat = b.reshape(n, -1).astype(np.float32)
    flat = moving_average_2d(flat, window)
    return flat.reshape(b.shape).astype(np.float32)


# ---------------------------------------------------
# FIX 3: TRANSL X/Z STABILIZATION
# ---------------------------------------------------

def stabilize_transl_xz(transl, smooth_window=5, velocity_clamp_xz=2.0):
    t = transl.copy().astype(np.float32)

    # frame0 기준 상대값으로 변환
    t = t - t[0:1]

    # X/Z만 다룸
    xz = t[:, [0, 2]].copy()

    # 급격한 spike clamp
    vel = velocity_norm(xz)
    spike = vel > velocity_clamp_xz
    for i in range(1, len(xz)):
        if spike[i]:
            xz[i] = xz[i - 1]

    # smoothing
    xz = moving_average_2d(xz, smooth_window)

    t[:, 0] = xz[:, 0]
    t[:, 2] = xz[:, 1]

    summary = {
        "xz_vel_before_mean": float(np.mean(vel)),
        "xz_vel_before_max": float(np.max(vel)),
        "xz_spike_count": int(np.sum(spike)),
    }

    return t.astype(np.float32), summary


# ---------------------------------------------------
# FIX 4: GROUND LOCK (핵심)
# ---------------------------------------------------

def apply_ground_lock(transl, mode="lock_frame0_y", smooth_y_window=21):
    """
    발이 항상 바닥에 붙어 보이게 만들기 위한 근사.
    실제 foot IK가 아니라 transl Y를 강하게 안정화한다.
    """
    t = transl.copy().astype(np.float32)
    y_before = t[:, 1].copy()

    if mode == "lock_frame0_y":
        ref_y = float(t[0, 1])
        t[:, 1] = ref_y

    elif mode == "median_y":
        ref_y = float(np.median(t[:, 1]))
        t[:, 1] = ref_y

    elif mode == "smooth_y":
        t[:, 1] = moving_average_1d(t[:, 1], smooth_y_window).astype(np.float32)

    else:
        raise ValueError(f"Unsupported GROUND_LOCK_MODE={mode}")

    y_after = t[:, 1].copy()

    summary = {
        "ground_lock_mode": mode,
        "y_before_min": float(np.min(y_before)),
        "y_before_max": float(np.max(y_before)),
        "y_before_std": float(np.std(y_before)),
        "y_after_min": float(np.min(y_after)),
        "y_after_max": float(np.max(y_after)),
        "y_after_std": float(np.std(y_after)),
    }

    return t.astype(np.float32), summary


# ---------------------------------------------------
# MAIN FIX
# ---------------------------------------------------

def fix_motion_v6_groundlock(motion):
    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"].copy(),
        "transl": motion["transl"].copy(),
        "fps": motion["fps"],
    }

    # 1) root rotation 안정화
    fixed["global_orient"], rot_summary = stabilize_root_rotation(
        fixed["global_orient"],
        max_rot_deg_per_frame=MAX_ROT_DEG_PER_FRAME
    )

    # 2) body pose 약한 smoothing
    fixed["body_pose"] = smooth_body_pose(
        fixed["body_pose"],
        window=SMOOTH_BODY
    )

    # 3) transl X/Z 안정화
    fixed["transl"], xz_summary = stabilize_transl_xz(
        fixed["transl"],
        smooth_window=SMOOTH_TRANSL_XZ,
        velocity_clamp_xz=VELOCITY_CLAMP_XZ
    )

    # 4) transl Y ground lock
    fixed["transl"], y_summary = apply_ground_lock(
        fixed["transl"],
        mode=GROUND_LOCK_MODE,
        smooth_y_window=SMOOTH_TRANSL_Y
    )

    summary = {
        "rotation": rot_summary,
        "transl_xz": xz_summary,
        "ground_lock": y_summary,
    }

    return fixed, summary


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()
    motion = load_motion(npz_path)

    fixed, summary = fix_motion_v6_groundlock(motion)

    out_dir = os.path.dirname(npz_path)
    out_npz = os.path.join(out_dir, "motion_fixed_v6_groundlock.npz")
    out_json = os.path.join(out_dir, "motion_fixed_v6_groundlock_summary.json")

    save_npz(out_npz, fixed)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[DONE]")


if __name__ == "__main__":
    main()