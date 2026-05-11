# npz_auto_fix_v6_1.py

import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"


# ---------------------------------------------------
# utils
# ---------------------------------------------------

def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )

    for d in raw_dirs:
        p = os.path.join(d, "motion_fixed_v5_safe.npz")
        if os.path.exists(p):
            return p

    raise RuntimeError("npz not found")


def load_motion(path):
    data = np.load(path, allow_pickle=True)

    motion = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
        "fps": int(data["fps"][0])
    }

    print("[INFO] loaded:", path)
    print("[INFO] frames:", motion["transl"].shape[0])

    return motion


def save_motion(path, motion):
    np.savez_compressed(
        path,
        global_orient=motion["global_orient"],
        body_pose=motion["body_pose"],
        betas=motion["betas"],
        transl=motion["transl"],
        fps=np.array([motion["fps"]], dtype=np.int32)
    )
    print("[SAVE]", path)


# ---------------------------------------------------
# rotation delta
# ---------------------------------------------------

def rotation_delta(rotmats):
    n = rotmats.shape[0]
    out = np.zeros(n)

    for i in range(1, n):
        rel = rotmats[i - 1].T @ rotmats[i]
        ang = R.from_matrix(rel).magnitude()
        out[i] = np.rad2deg(ang)

    return out


# ---------------------------------------------------
# STEP 1: rotation stabilize (v5 그대로)
# ---------------------------------------------------

def stabilize_rotation(motion):

    g = motion["global_orient"].copy()
    n = g.shape[0]

    fixed = g.copy()

    for i in range(1, n):

        prev = fixed[i - 1, 0]
        curr = fixed[i, 0]

        rel = prev.T @ curr
        angle = R.from_matrix(rel).magnitude()

        if angle > np.pi / 2:  # 90도 이상 튀면

            # 이전 프레임으로 고정
            fixed[i, 0] = prev

    motion["global_orient"] = fixed

    return motion


# ---------------------------------------------------
# STEP 2: transl smoothing
# ---------------------------------------------------

def smooth_transl(motion, window=5):

    t = motion["transl"].copy()

    for i in range(1, len(t) - 1):
        t[i] = (t[i - 1] + t[i] + t[i + 1]) / 3.0

    motion["transl"] = t
    return motion


# ---------------------------------------------------
# STEP 3: foot 기반 ground 보정 (핵심)
# ---------------------------------------------------

def reconstruct_joints_fast(motion):
    """
    SMPL 없이 pelvis만으로 근사
    (안정성 목적)
    """
    return motion["transl"]


def apply_ground_from_transl(motion):

    t = motion["transl"].copy()

    # transl Y만 기준으로 바닥 정의
    y = t[:, 1]

    # 전체 최소값
    floor = np.min(y)

    # offset 계산
    offset = y - floor

    # 전체를 내려줌
    t[:, 1] -= offset

    motion["transl"] = t

    return motion


# ---------------------------------------------------
# main fix
# ---------------------------------------------------

def fix_motion_v6_1(motion):

    fixed = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in motion.items()
    }

    # rotation 안정화
    fixed = stabilize_rotation(fixed)

    # transl smoothing
    fixed = smooth_transl(fixed)

    # ground (핵심)
    fixed = apply_ground_from_transl(fixed)

    # summary
    g = fixed["global_orient"][:, 0]
    rot_delta = rotation_delta(g)

    summary = {
        "rotation": {
            "mean": float(np.mean(rot_delta)),
            "max": float(np.max(rot_delta)),
        },
        "ground": {
            "y_min": float(np.min(fixed["transl"][:, 1])),
            "y_max": float(np.max(fixed["transl"][:, 1])),
        }
    }

    return fixed, summary


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():

    npz_path = find_latest_npz()
    motion = load_motion(npz_path)

    fixed, summary = fix_motion_v6_1(motion)

    out_path = os.path.join(
        os.path.dirname(npz_path),
        "motion_fixed_v6_1_ground.npz"
    )

    save_motion(out_path, fixed)

    print("\n[SUMMARY]")
    print(summary)

    print("\n[DONE]")


if __name__ == "__main__":
    main()