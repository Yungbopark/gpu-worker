import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import medfilt
from smplx import SMPL

# ---------------------------------------------------
# path setup
# ---------------------------------------------------

sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"
SMPL_MODEL_DIR = os.path.expanduser("~/.cache/4DHumans/data/smpl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------
# utils
# ---------------------------------------------------

def find_latest_npz() -> str:
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not raw_dirs:
        raise RuntimeError("raw_* folder not found")

    for d in raw_dirs:
        npz_path = os.path.join(d, "motion_data.npz")
        if os.path.exists(npz_path):
            return npz_path

    raise RuntimeError("motion_data.npz not found")


def load_motion(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    motion = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
        "fps": int(data["fps"][0]) if "fps" in data else 30,
    }
    print("[INFO] loaded npz:", npz_path)
    print("[INFO] global_orient:", motion["global_orient"].shape)
    print("[INFO] body_pose    :", motion["body_pose"].shape)
    print("[INFO] betas        :", motion["betas"].shape)
    print("[INFO] transl       :", motion["transl"].shape)
    print("[INFO] fps          :", motion["fps"])
    return motion


def save_motion_npz(out_path: str, motion: Dict[str, np.ndarray]):
    np.savez_compressed(
        out_path,
        global_orient=motion["global_orient"],
        body_pose=motion["body_pose"],
        betas=motion["betas"],
        transl=motion["transl"],
        fps=np.array([motion["fps"]], dtype=np.int32),
    )
    print("[INFO] saved fixed npz:", out_path)


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    flat = rotmat.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()
    return aa.reshape(*rotmat.shape[:-2], 3)


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    flat = axis_angle.reshape(-1, 3)
    mats = R.from_rotvec(flat).as_matrix()
    return mats.reshape(*axis_angle.shape[:-1], 3, 3)


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    flat = rotmat.reshape(-1, 3, 3)
    q = R.from_matrix(flat).as_quat()
    return q.reshape(*rotmat.shape[:-2], 4)


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    flat = quat.reshape(-1, 4)
    mats = R.from_quat(flat).as_matrix()
    return mats.reshape(*quat.shape[:-1], 3, 3)


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def moving_average_nd(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    flat_in = x.reshape(x.shape[0], -1)
    flat_out = out.reshape(x.shape[0], -1)
    for j in range(flat_in.shape[1]):
        flat_out[:, j] = moving_average_1d(flat_in[:, j], window)
    return out


def median_filter_nd(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return x.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    out = np.zeros_like(x, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1)
    out_flat = out.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = medfilt(flat[:, j], kernel_size=kernel_size)
    return out


def detect_spikes_zscore(signal: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x, dtype=bool)
    z = np.abs((x - mu) / std)
    return z > z_thresh


def clamp_spikes_1d(x: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    y = x.copy()
    n = len(y)
    for i in range(n):
        if not spike_mask[i]:
            continue
        if i == 0 and n > 1:
            y[i] = y[i + 1]
        elif i == n - 1 and n > 1:
            y[i] = y[i - 1]
        elif 0 < i < n - 1:
            y[i] = 0.5 * (y[i - 1] + y[i + 1])
    return y


def clamp_spikes_nd(x: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    y = x.copy()
    flat = y.reshape(y.shape[0], -1)
    for j in range(flat.shape[1]):
        flat[:, j] = clamp_spikes_1d(flat[:, j], spike_mask)
    return y


def interpolate_bad_frames_nd(x: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    y = x.copy()
    flat = y.reshape(y.shape[0], -1)
    n = flat.shape[0]
    good_idx = np.where(~bad_mask)[0]
    if len(good_idx) < 2:
        return y

    for j in range(flat.shape[1]):
        series = flat[:, j]
        flat[:, j] = np.interp(np.arange(n), good_idx, series[good_idx]).astype(np.float32)
    return y


def drop_bad_frames_motion(motion: Dict[str, np.ndarray], keep_mask: np.ndarray) -> Dict[str, np.ndarray]:
    fixed = {
        "global_orient": motion["global_orient"][keep_mask].copy(),
        "body_pose": motion["body_pose"][keep_mask].copy(),
        "betas": motion["betas"].copy() if motion["betas"].ndim == 1 else motion["betas"][keep_mask].copy(),
        "transl": motion["transl"][keep_mask].copy(),
        "fps": motion["fps"],
    }
    return fixed


# ---------------------------------------------------
# rotation helpers
# ---------------------------------------------------

def rotation_delta_deg(rotmats: np.ndarray) -> np.ndarray:
    n = rotmats.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        rel = rotmats[i - 1].T @ rotmats[i]
        ang = R.from_matrix(rel).magnitude()
        out[i] = np.rad2deg(ang)
    return out


def fix_quaternion_sign_flips(rotmats: np.ndarray) -> np.ndarray:
    """
    Sign continuity fix only.
    Same rotation, different quaternion sign -> normalize to continuous sign.
    """
    quat = rotmat_to_quat_xyzw(rotmats).reshape(rotmats.shape[0], 4)
    out = quat.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] *= -1.0
    return quat_xyzw_to_rotmat(out)


def clamp_large_rotation_jumps(rotmats: np.ndarray, threshold_deg: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    If rotation delta from previous frame exceeds threshold, replace with previous frame.
    Conservative correction.
    """
    out = rotmats.copy()
    bad = np.zeros(len(out), dtype=bool)
    for i in range(1, len(out)):
        rel = out[i - 1].T @ out[i]
        ang_deg = np.rad2deg(R.from_matrix(rel).magnitude())
        if ang_deg > threshold_deg:
            out[i] = out[i - 1]
            bad[i] = True
    return out, bad


# ---------------------------------------------------
# velocity clamp
# ---------------------------------------------------

def compute_vel_norm(x: np.ndarray) -> np.ndarray:
    n = len(x)
    vel = np.zeros(n, dtype=np.float32)
    if n > 1:
        vel[1:] = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return vel


def velocity_clamp_translation(transl: np.ndarray, max_vel: float) -> Tuple[np.ndarray, np.ndarray]:
    out = transl.copy().astype(np.float32)
    bad = np.zeros(len(out), dtype=bool)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        vel = float(np.linalg.norm(delta))
        if vel > max_vel:
            scale = max_vel / max(vel, 1e-8)
            out[i] = out[i - 1] + delta * scale
            bad[i] = True
    return out, bad


# ---------------------------------------------------
# SMPL reconstruction
# ---------------------------------------------------

def reconstruct_smpl(motion: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        gender="neutral",
        batch_size=1
    ).to(DEVICE)

    faces = smpl.faces.astype(np.uint32)

    g = motion["global_orient"]
    b = motion["body_pose"]
    t = motion["transl"]
    betas = motion["betas"]

    if betas.ndim == 2:
        betas_one = betas[0:1]
    else:
        betas_one = betas.reshape(1, -1)

    betas_t = torch.tensor(betas_one, dtype=torch.float32, device=DEVICE)

    n_frames = t.shape[0]
    verts_seq = []
    joints_seq = []

    is_rotmat = (b.ndim == 4 and b.shape[-2:] == (3, 3))
    print("[INFO] reconstruct frames:", n_frames)
    print("[INFO] pose mode:", "rotmat" if is_rotmat else "axis-angle")

    for i in range(n_frames):
        with torch.no_grad():
            g_i = torch.tensor(g[i:i+1], dtype=torch.float32, device=DEVICE)
            b_i = torch.tensor(b[i:i+1], dtype=torch.float32, device=DEVICE)
            t_i = torch.tensor(t[i:i+1], dtype=torch.float32, device=DEVICE)

            if is_rotmat and g_i.ndim == 3:
                g_i = g_i.unsqueeze(1)

            out = smpl(
                global_orient=g_i,
                body_pose=b_i,
                betas=betas_t,
                transl=t_i,
                pose2rot=not is_rotmat,
            )

        verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        joints = out.joints[0].detach().cpu().numpy().astype(np.float32)

        verts_seq.append(verts)
        joints_seq.append(joints)

    return np.stack(verts_seq, axis=0), np.stack(joints_seq, axis=0), faces


# ---------------------------------------------------
# metrics
# ---------------------------------------------------

def compute_metrics(motion: Dict[str, np.ndarray], joints_seq: np.ndarray) -> Dict[str, np.ndarray]:
    transl = motion["transl"]
    global_orient = motion["global_orient"]
    body_pose = motion["body_pose"]

    n = transl.shape[0]
    pelvis_y = joints_seq[:, 0, 1]

    foot_indices = [j for j in [7, 8, 10, 11] if j < joints_seq.shape[1]]
    if len(foot_indices) == 0:
        raise RuntimeError("No foot joints available")

    left_candidates = [j for j in [7, 10] if j < joints_seq.shape[1]]
    right_candidates = [j for j in [8, 11] if j < joints_seq.shape[1]]

    left_foot_y = joints_seq[:, left_candidates, 1].min(axis=1) if left_candidates else np.zeros(n, dtype=np.float32)
    right_foot_y = joints_seq[:, right_candidates, 1].min(axis=1) if right_candidates else np.zeros(n, dtype=np.float32)
    foot_min_y = joints_seq[:, foot_indices, 1].min(axis=1)

    transl_vel = compute_vel_norm(transl)

    if global_orient.ndim == 4:
        root_rot = global_orient[:, 0]
    else:
        root_rot = axis_angle_to_rotmat(global_orient.reshape(n, 3))
    root_rot_delta_deg = rotation_delta_deg(root_rot)

    if body_pose.ndim == 4:
        body_flat = body_pose.reshape(n, -1, 3, 3)
        body_delta = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            acc = []
            for j in range(body_flat.shape[1]):
                rel = body_flat[i - 1, j].T @ body_flat[i, j]
                ang = R.from_matrix(rel).magnitude()
                acc.append(np.rad2deg(ang))
            body_delta[i] = np.mean(acc)
    else:
        body_flat = body_pose.reshape(n, -1, 3)
        body_delta = np.zeros(n, dtype=np.float32)
        body_delta[1:] = np.linalg.norm(body_flat[1:] - body_flat[:-1], axis=(1, 2))

    return {
        "pelvis_y": pelvis_y.astype(np.float32),
        "left_foot_y": left_foot_y.astype(np.float32),
        "right_foot_y": right_foot_y.astype(np.float32),
        "foot_min_y": foot_min_y.astype(np.float32),
        "transl_vel": transl_vel.astype(np.float32),
        "root_rot_delta_deg": root_rot_delta_deg.astype(np.float32),
        "body_pose_delta": body_delta.astype(np.float32),
    }


def detect_metric_spikes(metrics: Dict[str, np.ndarray], z_thresh: float = 3.0) -> Dict[str, np.ndarray]:
    masks = {}
    for k, v in metrics.items():
        masks[k] = detect_spikes_zscore(v, z_thresh=z_thresh)
    return masks


def merge_spike_masks(masks: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    merged = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for k in keys:
        merged |= masks[k]
    return merged


# ---------------------------------------------------
# auto tuning
# ---------------------------------------------------

def auto_tune_windows(metrics_before: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
    """
    Conservative heuristic.
    """
    root_std = float(np.std(metrics_before["root_rot_delta_deg"]))
    vel_std = float(np.std(metrics_before["transl_vel"]))
    body_std = float(np.std(metrics_before["body_pose_delta"]))

    transl_w = 5
    orient_w = 5
    body_w = 3

    if vel_std > 40:
        transl_w = 7
    if vel_std > 80:
        transl_w = 9

    if root_std > 30:
        orient_w = 7
    if root_std > 50:
        orient_w = 9

    if body_std > 8:
        body_w = 5
    if body_std > 12:
        body_w = 7

    if transl_w % 2 == 0:
        transl_w += 1
    if orient_w % 2 == 0:
        orient_w += 1
    if body_w % 2 == 0:
        body_w += 1

    return transl_w, orient_w, body_w


# ---------------------------------------------------
# correction pipeline
# ---------------------------------------------------

def fix_motion_v2(
    motion: Dict[str, np.ndarray],
    metrics_before: Dict[str, np.ndarray],
    merged_spike_mask: np.ndarray,
    mode: str = "interpolate",   # interpolate | clamp | drop
    z_thresh: float = 3.0,
    transl_vel_limit: float = 80.0,
    rot_jump_threshold_deg: float = 120.0,
    auto_tune: bool = True,
    smooth_window_transl: int = 5,
    smooth_window_orient: int = 5,
    smooth_window_body: int = 3,
    median_kernel: int = 3,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"].copy(),
        "transl": motion["transl"].copy(),
        "fps": motion["fps"],
    }

    if auto_tune:
        smooth_window_transl, smooth_window_orient, smooth_window_body = auto_tune_windows(metrics_before)

    info = {
        "auto_tuned_windows": {
            "transl": smooth_window_transl,
            "orient": smooth_window_orient,
            "body": smooth_window_body,
        },
        "rotation_flip_count": 0,
        "rotation_jump_clamp_count": 0,
        "velocity_clamp_count": 0,
        "input_spike_count": int(np.sum(merged_spike_mask)),
        "mode": mode,
    }

    # ---------- 1) transl ----------
    transl = fixed["transl"].astype(np.float32)

    # velocity clamp first
    transl, vel_bad = velocity_clamp_translation(transl, max_vel=transl_vel_limit)
    info["velocity_clamp_count"] = int(np.sum(vel_bad))

    bad_mask = merged_spike_mask | vel_bad

    if mode == "interpolate":
        transl = interpolate_bad_frames_nd(transl, bad_mask)
    elif mode == "clamp":
        transl = clamp_spikes_nd(transl, bad_mask)
    elif mode == "drop":
        pass
    else:
        raise ValueError("mode must be interpolate | clamp | drop")

    if mode != "drop":
        transl = median_filter_nd(transl, median_kernel)
        transl = moving_average_nd(transl, smooth_window_transl)

    fixed["transl"] = transl.astype(np.float32)

    # ---------- 2) global orient ----------
    g = fixed["global_orient"]
    if g.ndim == 4:
        root_rot = g[:, 0].astype(np.float32)

        # quat sign flip fix
        quat_before = rotmat_to_quat_xyzw(root_rot).reshape(len(root_rot), 4)
        quat_after = quat_before.copy()
        flip_count = 0
        for i in range(1, len(quat_after)):
            if np.dot(quat_after[i - 1], quat_after[i]) < 0:
                quat_after[i] *= -1.0
                flip_count += 1
        info["rotation_flip_count"] = flip_count
        root_rot = quat_xyzw_to_rotmat(quat_after)

        # large jump clamp
        root_rot, rot_bad = clamp_large_rotation_jumps(root_rot, threshold_deg=rot_jump_threshold_deg)
        info["rotation_jump_clamp_count"] = int(np.sum(rot_bad))
        root_bad_mask = bad_mask | rot_bad

        root_aa = rotmat_to_axis_angle(root_rot)
        if mode == "interpolate":
            root_aa = interpolate_bad_frames_nd(root_aa, root_bad_mask)
        elif mode == "clamp":
            root_aa = clamp_spikes_nd(root_aa, root_bad_mask)

        if mode != "drop":
            root_aa = median_filter_nd(root_aa, median_kernel)
            root_aa = moving_average_nd(root_aa, smooth_window_orient)

        fixed["global_orient"] = axis_angle_to_rotmat(root_aa)[:, None, :, :].astype(np.float32)

    else:
        root_aa = g.reshape(g.shape[0], -1).astype(np.float32)
        if mode == "interpolate":
            root_aa = interpolate_bad_frames_nd(root_aa, bad_mask)
        elif mode == "clamp":
            root_aa = clamp_spikes_nd(root_aa, bad_mask)
        if mode != "drop":
            root_aa = median_filter_nd(root_aa, median_kernel)
            root_aa = moving_average_nd(root_aa, smooth_window_orient)
        fixed["global_orient"] = root_aa.reshape(g.shape).astype(np.float32)

    # ---------- 3) body pose ----------
    b = fixed["body_pose"]
    if b.ndim == 4:
        body_aa = rotmat_to_axis_angle(b).reshape(b.shape[0], -1, 3)
        body_flat = body_aa.reshape(body_aa.shape[0], -1)

        if mode == "interpolate":
            body_flat = interpolate_bad_frames_nd(body_flat, bad_mask)
        elif mode == "clamp":
            body_flat = clamp_spikes_nd(body_flat, bad_mask)

        if mode != "drop":
            body_flat = median_filter_nd(body_flat, median_kernel)
            body_flat = moving_average_nd(body_flat, smooth_window_body)

        body_aa = body_flat.reshape(body_aa.shape)
        fixed["body_pose"] = axis_angle_to_rotmat(body_aa).reshape(b.shape[0], b.shape[1], 3, 3).astype(np.float32)
    else:
        body_flat = b.reshape(b.shape[0], -1).astype(np.float32)
        if mode == "interpolate":
            body_flat = interpolate_bad_frames_nd(body_flat, bad_mask)
        elif mode == "clamp":
            body_flat = clamp_spikes_nd(body_flat, bad_mask)
        if mode != "drop":
            body_flat = median_filter_nd(body_flat, median_kernel)
            body_flat = moving_average_nd(body_flat, smooth_window_body)
        fixed["body_pose"] = body_flat.reshape(b.shape).astype(np.float32)

    # ---------- 4) drop mode ----------
    if mode == "drop":
        keep_mask = ~bad_mask
        fixed = drop_bad_frames_motion(fixed, keep_mask)
        info["dropped_frames"] = np.where(bad_mask)[0].astype(int).tolist()
        info["dropped_count"] = int(np.sum(bad_mask))
    else:
        info["dropped_frames"] = []
        info["dropped_count"] = 0

    return fixed, info


# ---------------------------------------------------
# plotting / summary
# ---------------------------------------------------

def save_plots(
    metrics_before: Dict[str, np.ndarray],
    metrics_after: Dict[str, np.ndarray],
    spike_mask: np.ndarray,
    out_path: str
):
    keys = [
        "pelvis_y",
        "left_foot_y",
        "right_foot_y",
        "foot_min_y",
        "transl_vel",
        "root_rot_delta_deg",
        "body_pose_delta",
    ]

    fig, axes = plt.subplots(len(keys), 1, figsize=(16, 3 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    x_before = np.arange(len(spike_mask))

    for ax, k in zip(axes, keys):
        ax.plot(x_before, metrics_before[k], label=f"{k}_before", linewidth=1.1)
        spike_idx = np.where(spike_mask)[0]
        if len(spike_idx) > 0:
            ax.scatter(spike_idx, metrics_before[k][spike_idx], s=10, label="spikes")

        if metrics_after is not None:
            x_after = np.arange(len(metrics_after[k]))
            ax.plot(x_after, metrics_after[k], label=f"{k}_after", linewidth=1.1)

        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[INFO] saved plot:", out_path)


def summarize_metrics(metrics: Dict[str, np.ndarray], spike_mask: np.ndarray) -> Dict:
    summary = {}
    for k, v in metrics.items():
        summary[k] = {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }
    summary["spike_count"] = int(np.sum(spike_mask))
    summary["spike_frames"] = np.where(spike_mask)[0].astype(int).tolist()
    return summary


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--z_thresh", type=float, default=3.0)
    parser.add_argument("--mode", type=str, default="interpolate", choices=["interpolate", "clamp", "drop"])
    parser.add_argument("--save_fixed", action="store_true")
    parser.add_argument("--auto_tune", action="store_true")

    parser.add_argument("--smooth_transl", type=int, default=5)
    parser.add_argument("--smooth_orient", type=int, default=5)
    parser.add_argument("--smooth_body", type=int, default=3)
    parser.add_argument("--median_kernel", type=int, default=3)

    parser.add_argument("--transl_vel_limit", type=float, default=80.0)
    parser.add_argument("--rot_jump_threshold_deg", type=float, default=120.0)

    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()
    raw_dir = os.path.dirname(npz_path)

    motion = load_motion(npz_path)

    # before
    _, joints_before, _ = reconstruct_smpl(motion)
    metrics_before = compute_metrics(motion, joints_before)
    masks = detect_metric_spikes(metrics_before, z_thresh=args.z_thresh)

    merged_spike_mask = merge_spike_masks(
        masks,
        keys=[
            "transl_vel",
            "root_rot_delta_deg",
            "body_pose_delta",
            "foot_min_y",
        ]
    )

    print("[INFO] merged spike count:", int(np.sum(merged_spike_mask)))
    print("[INFO] merged spike frames:", np.where(merged_spike_mask)[0].tolist())

    metrics_after = None
    fix_info = {}

    if args.save_fixed:
        fixed_motion, fix_info = fix_motion_v2(
            motion=motion,
            metrics_before=metrics_before,
            merged_spike_mask=merged_spike_mask,
            mode=args.mode,
            z_thresh=args.z_thresh,
            transl_vel_limit=args.transl_vel_limit,
            rot_jump_threshold_deg=args.rot_jump_threshold_deg,
            auto_tune=args.auto_tune,
            smooth_window_transl=args.smooth_transl,
            smooth_window_orient=args.smooth_orient,
            smooth_window_body=args.smooth_body,
            median_kernel=args.median_kernel,
        )

        fixed_npz_path = os.path.join(raw_dir, "motion_fixed_v2.npz")
        save_motion_npz(fixed_npz_path, fixed_motion)

        _, joints_after, _ = reconstruct_smpl(fixed_motion)
        metrics_after = compute_metrics(fixed_motion, joints_after)

    # save metrics
    metrics_npz_path = os.path.join(raw_dir, "debug_metrics_v2.npz")
    np.savez_compressed(
        metrics_npz_path,
        **metrics_before,
        spike_mask=merged_spike_mask.astype(np.uint8),
    )
    print("[INFO] saved metrics npz:", metrics_npz_path)

    # save plot
    plot_path = os.path.join(raw_dir, "debug_plots_v2.png")
    save_plots(metrics_before, metrics_after, merged_spike_mask, plot_path)

    # summary
    summary = {
        "before": summarize_metrics(metrics_before, merged_spike_mask),
        "fix_info": fix_info,
    }

    if metrics_after is not None:
        summary["after"] = summarize_metrics(
            metrics_after,
            np.zeros(len(next(iter(metrics_after.values()))), dtype=bool)
        )

    summary_path = os.path.join(raw_dir, "debug_summary_v2.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[INFO] saved summary:", summary_path)

    print("\n========== DONE ==========")
    print("NPZ:", npz_path)
    if args.save_fixed:
        print("Fixed NPZ:", os.path.join(raw_dir, "motion_fixed_v2.npz"))
    print("Metrics:", metrics_npz_path)
    print("Plot:", plot_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()