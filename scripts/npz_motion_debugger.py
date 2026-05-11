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

from scipy.spatial.transform import Rotation as R
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
        reverse=True
    )

    if not raw_dirs:
        raise RuntimeError("raw_* folder not found")

    for d in raw_dirs:
        npz_path = os.path.join(d, "motion_fixed_clean.npz")
        if os.path.exists(npz_path):
            return npz_path

    raise RuntimeError("motion_data.npz not found")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def moving_average_nd(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):
        out.reshape(x.shape[0], -1)[:, j] = moving_average_1d(flat[:, j], window)
    return out


def median_filter_nd(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return x.copy()
    out = np.zeros_like(x, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1)
    out_flat = out.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = medfilt(flat[:, j], kernel_size=kernel_size)
    return out


def rotation_delta_deg(rotmats: np.ndarray) -> np.ndarray:
    """
    rotmats: (N, 3, 3)
    returns: (N,) delta from previous frame in degrees
    """
    n = rotmats.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        rel = rotmats[i - 1].T @ rotmats[i]
        ang = R.from_matrix(rel).magnitude()
        out[i] = np.rad2deg(ang)
    return out


def detect_spikes(signal: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """
    signal: (N,)
    returns boolean mask
    """
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
        if i == 0:
            y[i] = y[i + 1]
        elif i == n - 1:
            y[i] = y[i - 1]
        else:
            y[i] = 0.5 * (y[i - 1] + y[i + 1])
    return y


def clamp_spikes_nd(x: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    y = x.copy()
    flat = y.reshape(y.shape[0], -1)
    for j in range(flat.shape[1]):
        flat[:, j] = clamp_spikes_1d(flat[:, j], spike_mask)
    return y


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

    verts_seq = np.stack(verts_seq, axis=0)
    joints_seq = np.stack(joints_seq, axis=0)

    return verts_seq, joints_seq, faces


# ---------------------------------------------------
# metrics
# ---------------------------------------------------

def compute_metrics(motion: Dict[str, np.ndarray], joints_seq: np.ndarray) -> Dict[str, np.ndarray]:
    transl = motion["transl"]
    global_orient = motion["global_orient"]
    body_pose = motion["body_pose"]

    n = transl.shape[0]

    pelvis_y = joints_seq[:, 0, 1]

    # common SMPL foot-related joints
    foot_indices = [j for j in [7, 8, 10, 11] if j < joints_seq.shape[1]]
    if len(foot_indices) == 0:
        raise RuntimeError("No foot joints available")

    left_candidates = [j for j in [7, 10] if j < joints_seq.shape[1]]
    right_candidates = [j for j in [8, 11] if j < joints_seq.shape[1]]

    left_foot_y = joints_seq[:, left_candidates, 1].min(axis=1) if left_candidates else np.zeros(n, dtype=np.float32)
    right_foot_y = joints_seq[:, right_candidates, 1].min(axis=1) if right_candidates else np.zeros(n, dtype=np.float32)
    foot_min_y = joints_seq[:, foot_indices, 1].min(axis=1)

    transl_vel = np.zeros(n, dtype=np.float32)
    transl_vel[1:] = np.linalg.norm(transl[1:] - transl[:-1], axis=1)

    # root rotation delta
    if global_orient.ndim == 4:
        root_rot = global_orient[:, 0]
    else:
        root_rot = axis_angle_to_rotmat(global_orient.reshape(n, 3))
    root_rot_delta_deg = rotation_delta_deg(root_rot)

    # body pose delta
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

    metrics = {
        "pelvis_y": pelvis_y.astype(np.float32),
        "left_foot_y": left_foot_y.astype(np.float32),
        "right_foot_y": right_foot_y.astype(np.float32),
        "foot_min_y": foot_min_y.astype(np.float32),
        "transl_vel": transl_vel.astype(np.float32),
        "root_rot_delta_deg": root_rot_delta_deg.astype(np.float32),
        "body_pose_delta": body_delta.astype(np.float32),
    }

    return metrics


def detect_metric_spikes(metrics: Dict[str, np.ndarray], z_thresh: float = 3.0) -> Dict[str, np.ndarray]:
    masks = {}
    for k, v in metrics.items():
        masks[k] = detect_spikes(v, z_thresh=z_thresh)
    return masks


def merge_spike_masks(masks: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    if not keys:
        raise ValueError("No keys to merge")
    merged = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for k in keys:
        merged |= masks[k]
    return merged


# ---------------------------------------------------
# correction
# ---------------------------------------------------

def fix_motion(
    motion: Dict[str, np.ndarray],
    spike_mask: np.ndarray,
    smooth_window_transl: int = 5,
    smooth_window_orient: int = 5,
    smooth_window_body: int = 3,
    apply_spike_clamp: bool = True,
) -> Dict[str, np.ndarray]:
    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"].copy(),
        "transl": motion["transl"].copy(),
        "fps": motion["fps"],
    }

    # 1) transl
    transl = fixed["transl"].astype(np.float32)
    if apply_spike_clamp:
        transl = clamp_spikes_nd(transl, spike_mask)
    transl = moving_average_nd(transl, smooth_window_transl)
    fixed["transl"] = transl.astype(np.float32)

    # 2) global orient
    g = fixed["global_orient"]
    if g.ndim == 4:
        aa = rotmat_to_axis_angle(g[:, 0])  # (N,3)
        if apply_spike_clamp:
            aa = clamp_spikes_nd(aa, spike_mask)
        aa = moving_average_nd(aa, smooth_window_orient)
        g_fixed = axis_angle_to_rotmat(aa)[:, None, :, :]
        fixed["global_orient"] = g_fixed.astype(np.float32)
    else:
        aa = g.reshape(g.shape[0], -1).astype(np.float32)
        if apply_spike_clamp:
            aa = clamp_spikes_nd(aa, spike_mask)
        aa = moving_average_nd(aa, smooth_window_orient)
        fixed["global_orient"] = aa.reshape(g.shape).astype(np.float32)

    # 3) body pose
    b = fixed["body_pose"]
    if b.ndim == 4:
        aa = rotmat_to_axis_angle(b).reshape(b.shape[0], -1, 3)  # (N,23,3)
        aa_flat = aa.reshape(aa.shape[0], -1)
        if apply_spike_clamp:
            aa_flat = clamp_spikes_nd(aa_flat, spike_mask)
        aa_flat = moving_average_nd(aa_flat, smooth_window_body)
        aa = aa_flat.reshape(aa.shape)
        b_fixed = axis_angle_to_rotmat(aa).reshape(b.shape[0], b.shape[1], 3, 3)
        fixed["body_pose"] = b_fixed.astype(np.float32)
    else:
        aa = b.reshape(b.shape[0], -1).astype(np.float32)
        if apply_spike_clamp:
            aa = clamp_spikes_nd(aa, spike_mask)
        aa = moving_average_nd(aa, smooth_window_body)
        fixed["body_pose"] = aa.reshape(b.shape).astype(np.float32)

    return fixed


# ---------------------------------------------------
# plotting
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

    x = np.arange(len(spike_mask))

    for ax, k in zip(axes, keys):
        ax.plot(x, metrics_before[k], label=f"{k}_before", linewidth=1.2)
        if metrics_after is not None:
            ax.plot(x, metrics_after[k], label=f"{k}_after", linewidth=1.2)
        spike_idx = np.where(spike_mask)[0]
        if len(spike_idx) > 0:
            ax.scatter(spike_idx, metrics_before[k][spike_idx], s=12, label="spikes")
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print("[INFO] saved plot:", out_path)


# ---------------------------------------------------
# summary
# ---------------------------------------------------

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
    parser.add_argument("--save_fixed", action="store_true")
    parser.add_argument("--smooth_transl", type=int, default=5)
    parser.add_argument("--smooth_orient", type=int, default=5)
    parser.add_argument("--smooth_body", type=int, default=3)
    parser.add_argument("--no_spike_clamp", action="store_true")

    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()
    raw_dir = os.path.dirname(npz_path)

    motion = load_motion(npz_path)

    # reconstruct before
    verts_seq, joints_seq, _ = reconstruct_smpl(motion)
    metrics_before = compute_metrics(motion, joints_seq)
    masks = detect_metric_spikes(metrics_before, z_thresh=args.z_thresh)

    # merge masks
    # 여기서는 튀는 문제와 관련이 큰 항목 위주로 합칩니다.
    merged_spike_mask = merge_spike_masks(
        masks,
        keys=[
            "transl_vel",
            "root_rot_delta_deg",
            "body_pose_delta",
            "foot_min_y",
        ]
    )

    print("[INFO] spike count:", int(np.sum(merged_spike_mask)))
    print("[INFO] spike frames:", np.where(merged_spike_mask)[0].tolist())

    fixed_motion = None
    metrics_after = None

    if args.save_fixed:
        fixed_motion = fix_motion(
            motion=motion,
            spike_mask=merged_spike_mask,
            smooth_window_transl=args.smooth_transl,
            smooth_window_orient=args.smooth_orient,
            smooth_window_body=args.smooth_body,
            apply_spike_clamp=not args.no_spike_clamp,
        )

        fixed_npz_path = os.path.join(raw_dir, "motion_fixed.npz")
        save_motion_npz(fixed_npz_path, fixed_motion)

        _, fixed_joints_seq, _ = reconstruct_smpl(fixed_motion)
        metrics_after = compute_metrics(fixed_motion, fixed_joints_seq)

    # save metrics
    metrics_npz_path = os.path.join(raw_dir, "debug_metrics.npz")
    np.savez_compressed(
        metrics_npz_path,
        **metrics_before,
        spike_mask=merged_spike_mask.astype(np.uint8),
    )
    print("[INFO] saved metrics npz:", metrics_npz_path)

    # save plot
    plot_path = os.path.join(raw_dir, "debug_plots.png")
    save_plots(metrics_before, metrics_after, merged_spike_mask, plot_path)

    # save summary
    summary = summarize_metrics(metrics_before, merged_spike_mask)
    summary_path = os.path.join(raw_dir, "debug_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[INFO] saved summary:", summary_path)

    print("\n========== DONE ==========")
    print("NPZ:", npz_path)
    if args.save_fixed:
        print("Fixed NPZ:", os.path.join(raw_dir, "motion_fixed.npz"))
    print("Metrics:", metrics_npz_path)
    print("Plot:", plot_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()