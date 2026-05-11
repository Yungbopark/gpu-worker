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
VELOCITY_CLAMP_XZ = 2.0

# foot contact detection
CONTACT_HEIGHT_PERCENTILE = 20.0
CONTACT_HEIGHT_MARGIN = 0.03
CONTACT_VEL_PERCENTILE = 40.0

# contact 기반 Y 보정
Y_OFFSET_SMOOTH_WINDOW = 15
Y_OFFSET_APPLY_RATIO = 0.85

# frame0 기준 상대좌표
USE_RELATIVE_FRAME0 = True


# ---------------------------------------------------
# IO
# ---------------------------------------------------

def find_latest_motion_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )

    preferred = [
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

    print("[INFO] loaded motion:", npz_path)
    print("[INFO] global_orient:", motion["global_orient"].shape)
    print("[INFO] body_pose    :", motion["body_pose"].shape)
    print("[INFO] betas        :", motion["betas"].shape)
    print("[INFO] transl       :", motion["transl"].shape)
    print("[INFO] fps          :", motion["fps"])
    return motion


def find_metrics_npz(motion_npz_path):
    raw_dir = os.path.dirname(motion_npz_path)
    candidates = [
        os.path.join(raw_dir, "debug_metrics.npz"),
        os.path.join(raw_dir, "debug_metrics_v2.npz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_metrics(metrics_npz_path):
    d = np.load(metrics_npz_path, allow_pickle=True)
    metrics = {k: d[k] for k in d.files}
    print("[INFO] loaded metrics:", metrics_npz_path)
    print("[INFO] metric keys:", sorted(metrics.keys()))
    return metrics


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
# Utils
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
# Rotation / pose stabilization
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


def smooth_body_pose(body_pose, window=3):
    b = body_pose.copy().astype(np.float32)
    n = b.shape[0]
    flat = b.reshape(n, -1)
    flat = moving_average_2d(flat, window)
    return flat.reshape(b.shape).astype(np.float32)


# ---------------------------------------------------
# XZ stabilization
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
# Contact-aware Y correction
# ---------------------------------------------------

def detect_contact_mask(metrics, n_frames):
    if "foot_min_y" in metrics:
        foot_y = metrics["foot_min_y"].astype(np.float32)
    elif "left_foot_y" in metrics and "right_foot_y" in metrics:
        foot_y = np.minimum(
            metrics["left_foot_y"].astype(np.float32),
            metrics["right_foot_y"].astype(np.float32),
        )
    else:
        raise RuntimeError("Metrics file does not contain foot height keys")

    if len(foot_y) != n_frames:
        raise RuntimeError(
            f"foot metric length mismatch: {len(foot_y)} vs motion frames {n_frames}"
        )

    if "transl_vel" in metrics and len(metrics["transl_vel"]) == n_frames:
        transl_vel = metrics["transl_vel"].astype(np.float32)
    else:
        transl_vel = np.zeros(n_frames, dtype=np.float32)

    base_height = np.percentile(foot_y, CONTACT_HEIGHT_PERCENTILE)
    height_mask = foot_y <= (base_height + CONTACT_HEIGHT_MARGIN)

    vel_thresh = np.percentile(transl_vel, CONTACT_VEL_PERCENTILE)
    vel_mask = transl_vel <= vel_thresh

    contact_mask = height_mask & vel_mask

    debug = {
        "base_height": float(base_height),
        "vel_thresh": float(vel_thresh),
        "contact_count": int(np.sum(contact_mask)),
        "contact_ratio": float(np.mean(contact_mask.astype(np.float32))),
        "foot_y_min": float(np.min(foot_y)),
        "foot_y_max": float(np.max(foot_y)),
    }

    return foot_y, transl_vel, contact_mask, debug


def interpolate_contact_offsets(foot_y, contact_mask):
    """
    contact frame에서만 foot_y를 샘플링하고,
    non-contact 구간은 보간해서 offset 곡선을 만든다.
    """
    n = len(foot_y)
    idx = np.arange(n)
    contact_idx = np.where(contact_mask)[0]

    if len(contact_idx) == 0:
        # contact를 못 찾으면 전체 최소값 기반
        offset = foot_y - np.min(foot_y)
        return offset.astype(np.float32), False

    sampled = foot_y[contact_idx]
    baseline = np.min(sampled)
    sampled_offset = sampled - baseline

    if len(contact_idx) == 1:
        offset = np.full(n, sampled_offset[0], dtype=np.float32)
    else:
        offset = np.interp(idx, contact_idx, sampled_offset).astype(np.float32)

    return offset.astype(np.float32), True


def apply_contact_aware_y_correction(transl, metrics):
    t = transl.copy().astype(np.float32)
    y_before = t[:, 1].copy()
    n = len(t)

    foot_y, transl_vel, contact_mask, contact_debug = detect_contact_mask(metrics, n)
    raw_offset, has_contact = interpolate_contact_offsets(foot_y, contact_mask)

    offset_smooth = moving_average_1d(raw_offset, Y_OFFSET_SMOOTH_WINDOW).astype(np.float32)
    y_after = y_before - (offset_smooth * Y_OFFSET_APPLY_RATIO)

    t[:, 1] = y_after

    summary = {
        "mode": "contact_aware_y_offset",
        "has_contact": bool(has_contact),
        "offset_apply_ratio": float(Y_OFFSET_APPLY_RATIO),
        "offset_raw_min": float(np.min(raw_offset)),
        "offset_raw_max": float(np.max(raw_offset)),
        "offset_smooth_min": float(np.min(offset_smooth)),
        "offset_smooth_max": float(np.max(offset_smooth)),
        "y_before_min": float(np.min(y_before)),
        "y_before_max": float(np.max(y_before)),
        "y_before_std": float(np.std(y_before)),
        "y_after_min": float(np.min(y_after)),
        "y_after_max": float(np.max(y_after)),
        "y_after_std": float(np.std(y_after)),
        "contact": contact_debug,
    }

    aux = {
        "foot_min_y": foot_y,
        "transl_vel": transl_vel,
        "contact_mask": contact_mask.astype(np.uint8),
        "raw_offset": raw_offset,
        "offset_smooth": offset_smooth,
    }

    return t.astype(np.float32), summary, aux


# ---------------------------------------------------
# Main fix
# ---------------------------------------------------

def fix_motion_v8(motion, metrics):
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

    fixed["transl"], y_summary, aux = apply_contact_aware_y_correction(
        fixed["transl"],
        metrics,
    )

    summary = {
        "rotation": rot_summary,
        "transl_xz": xz_summary,
        "transl_y": y_summary,
    }

    return fixed, summary, aux


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    args = parser.parse_args()

    motion_npz = args.npz if args.npz else find_latest_motion_npz()
    motion = load_motion(motion_npz)

    metrics_npz = args.metrics if args.metrics else find_metrics_npz(motion_npz)
    if metrics_npz is None:
        raise RuntimeError("debug_metrics.npz or debug_metrics_v2.npz not found")

    metrics = load_metrics(metrics_npz)

    fixed, summary, aux = fix_motion_v8(motion, metrics)

    raw_dir = os.path.dirname(motion_npz)
    out_npz = os.path.join(raw_dir, "motion_fixed_v8.npz")
    out_json = os.path.join(raw_dir, "motion_fixed_v8_summary.json")
    out_aux = os.path.join(raw_dir, "motion_fixed_v8_aux.npz")

    save_motion_npz(out_npz, fixed)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez_compressed(out_aux, **aux)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[DONE]")


if __name__ == "__main__":
    main()