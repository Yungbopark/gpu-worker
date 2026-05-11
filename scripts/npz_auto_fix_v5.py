import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"

# ==============================
# CONFIG
# ==============================

SMOOTH_TRANSL = 5
SMOOTH_ROT = 5
SMOOTH_BODY = 3

MAX_ROT_DEG_PER_FRAME = 10.0
VELOCITY_CLAMP = 2.0

# ==============================
# LOAD
# ==============================

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
    raise RuntimeError("npz not found")


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
    print("[INFO] frames:", motion["transl"].shape[0])
    return motion


# ==============================
# UTILS
# ==============================

def moving_average(x, w):
    if w <= 1:
        return x.copy()
    pad = w // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        out[:, i] = np.convolve(x_pad[:, i], np.ones(w)/w, mode="valid")
    return out


def rotation_delta_deg(rotmats):
    n = rotmats.shape[0]
    out = np.zeros(n)
    for i in range(1, n):
        rel = rotmats[i-1].T @ rotmats[i]
        ang = R.from_matrix(rel).magnitude()
        out[i] = np.rad2deg(ang)
    return out


# ==============================
# CORE FIX
# ==============================

def fix_motion_v5_safe(motion):

    g = motion["global_orient"].copy()
    b = motion["body_pose"].copy()
    t = motion["transl"].copy()

    n = t.shape[0]

    # ==========================
    # 1. transl smoothing + clamp
    # ==========================

    vel = np.linalg.norm(t[1:] - t[:-1], axis=1)
    vel = np.concatenate([[0], vel])

    spike = vel > (np.mean(vel) + 3*np.std(vel))

    for i in range(1, n):
        if spike[i]:
            t[i] = t[i-1]

    t = moving_average(t, SMOOTH_TRANSL)

    # ==========================
    # 2. rotation stabilization
    # ==========================

    rot = g[:,0]

    for i in range(1, n):
        rel = rot[i-1].T @ rot[i]
        angle = np.rad2deg(R.from_matrix(rel).magnitude())

        if angle > MAX_ROT_DEG_PER_FRAME:
            rot[i] = rot[i-1]

    g[:,0] = rot

    # ==========================
    # 3. body pose smoothing
    # ==========================

    b_flat = b.reshape(n, -1)
    b_flat = moving_average(b_flat, SMOOTH_BODY)
    b = b_flat.reshape(b.shape)

    # ==========================
    # 4. velocity clamp (global)
    # ==========================

    vel = np.linalg.norm(t[1:] - t[:-1], axis=1)
    for i in range(1, n):
        if vel[i-1] > VELOCITY_CLAMP:
            t[i] = t[i-1]

    # ==========================
    # SUMMARY
    # ==========================

    before_rot = rotation_delta_deg(motion["global_orient"][:,0])
    after_rot = rotation_delta_deg(g[:,0])

    summary = {
        "before_rot_mean": float(np.mean(before_rot)),
        "after_rot_mean": float(np.mean(after_rot)),
        "before_rot_max": float(np.max(before_rot)),
        "after_rot_max": float(np.max(after_rot)),
    }

    fixed = {
        "global_orient": g,
        "body_pose": b,
        "betas": motion["betas"],
        "transl": t,
        "fps": motion["fps"],
    }

    return fixed, summary


# ==============================
# SAVE
# ==============================

def save_npz(path, motion):
    np.savez_compressed(
        path,
        global_orient=motion["global_orient"],
        body_pose=motion["body_pose"],
        betas=motion["betas"],
        transl=motion["transl"],
        fps=np.array([motion["fps"]])
    )


# ==============================
# MAIN
# ==============================

def main():

    npz = find_latest_npz()
    motion = load_motion(npz)

    fixed, summary = fix_motion_v5_safe(motion)

    out = os.path.join(os.path.dirname(npz), "motion_fixed_v5_safe.npz")
    save_npz(out, fixed)

    print("\n[SUMMARY]")
    print(summary)
    print("\n[SAVED]", out)


if __name__ == "__main__":
    main()