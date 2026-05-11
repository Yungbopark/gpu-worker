import os
import sys
import glob
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------
# paths
# ---------------------------------------------------

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
        p = os.path.join(d, "motion_data.npz")
        if os.path.exists(p):
            return p

    raise RuntimeError("motion_data.npz not found")


def load_motion(path):
    data = np.load(path, allow_pickle=True)

    return {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
        "fps": int(data["fps"][0]) if "fps" in data else 30,
    }


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
# rotation helpers
# ---------------------------------------------------

def rotmat_to_aa(rot):
    flat = rot.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()
    return aa.reshape(*rot.shape[:-2], 3)


def aa_to_rotmat(aa):
    flat = aa.reshape(-1, 3)
    mat = R.from_rotvec(flat).as_matrix()
    return mat.reshape(*aa.shape[:-1], 3, 3)


# ---------------------------------------------------
# FIX 1: rotation flip 제거
# ---------------------------------------------------

def fix_rotation_flip(aa_seq):
    """
    axis-angle sign flip 제거
    """
    out = aa_seq.copy()

    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]

        if np.linalg.norm(cur - prev) > np.linalg.norm(-cur - prev):
            out[i] = -cur

    return out


# ---------------------------------------------------
# FIX 2: transl velocity clamp
# ---------------------------------------------------

def clamp_velocity(transl, max_vel=0.15):
    out = transl.copy()

    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        dist = np.linalg.norm(delta)

        if dist > max_vel:
            out[i] = out[i - 1] + delta / dist * max_vel

    return out


# ---------------------------------------------------
# FIX 3: foot anchoring
# ---------------------------------------------------

def estimate_foot_min_y(transl):
    """
    간단한 근사:
    transl.y 기반 (SMPL reconstruction 없이)
    """
    return transl[:, 1]


def apply_foot_anchoring(transl):
    foot_y = estimate_foot_min_y(transl)

    base = np.percentile(foot_y, 10)

    mask = foot_y < base + 0.02

    out = transl.copy()

    for i in range(len(out)):
        if mask[i]:
            out[i, 1] -= foot_y[i]

    return out


# ---------------------------------------------------
# smoothing (안정화)
# ---------------------------------------------------

def moving_average(x, k=5):
    if k <= 1:
        return x.copy()

    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")

    out = np.zeros_like(x)

    for i in range(len(x)):
        out[i] = xp[i:i + k].mean(axis=0)

    return out


# ---------------------------------------------------
# main fix pipeline
# ---------------------------------------------------

def fix_motion(motion):

    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"],
        "transl": motion["transl"].copy(),
        "fps": motion["fps"]
    }

    print("\n[STEP] rotation flip fix")

    # global orient
    g = fixed["global_orient"][:, 0]
    aa = rotmat_to_aa(g)

    aa = fix_rotation_flip(aa)
    aa = moving_average(aa, 5)

    fixed["global_orient"] = aa_to_rotmat(aa)[:, None]

    # body pose
    print("[STEP] body pose smoothing")

    b = fixed["body_pose"]
    aa = rotmat_to_aa(b)

    aa = fix_rotation_flip(aa.reshape(len(aa), -1, 3)).reshape(aa.shape)
    aa = moving_average(aa.reshape(len(aa), -1), 3).reshape(aa.shape)

    fixed["body_pose"] = aa_to_rotmat(aa)

    # transl
    print("[STEP] transl clamp")

    t = fixed["transl"]

    t = clamp_velocity(t, max_vel=0.15)
    t = moving_average(t, 5)

    print("[STEP] foot anchoring")

    t = apply_foot_anchoring(t)

    fixed["transl"] = t

    return fixed


# ---------------------------------------------------
# debug compare
# ---------------------------------------------------

def compare(before, after):

    diff = np.linalg.norm(after["transl"] - before["transl"], axis=1)

    print("\n[COMPARE]")
    print("transl change mean:", diff.mean())
    print("transl change max :", diff.max())


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()

    motion = load_motion(npz_path)

    fixed = fix_motion(motion)

    out_path = npz_path.replace("motion_data.npz", "motion_fixed_v3.npz")

    save_motion(out_path, fixed)

    compare(motion, fixed)

    print("\n[DONE]")


if __name__ == "__main__":
    main()