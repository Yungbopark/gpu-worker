import os
import glob
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"


# ---------------------------------------------------
# IO
# ---------------------------------------------------

def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    for d in raw_dirs:
        p = os.path.join(d, "motion_data.npz")
        if os.path.exists(p):
            return p

    raise RuntimeError("motion_data.npz not found")


def load_motion(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    motion = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
        "fps": int(data["fps"][0]) if "fps" in data else 30,
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
# Math
# ---------------------------------------------------

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def rotation_matrix_from_axis_angle(axis, angle_rad):
    axis = normalize(axis.astype(np.float64))
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3, dtype=np.float64)

    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    return np.array([
        [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c    ],
    ], dtype=np.float64)


def signed_angle_on_plane(v1, v2, up_axis):
    """
    signed angle from v1 to v2 around up_axis
    """
    v1n = normalize(v1)
    v2n = normalize(v2)
    upn = normalize(up_axis)

    if np.linalg.norm(v1n) < 1e-8 or np.linalg.norm(v2n) < 1e-8:
        return 0.0

    cross = np.cross(v1n, v2n)
    sin_val = np.dot(cross, upn)
    cos_val = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.arctan2(sin_val, cos_val))


def project_to_plane(v, up_axis):
    upn = normalize(up_axis)
    return v - np.dot(v, upn) * upn


# ---------------------------------------------------
# Root yaw stabilization
# ---------------------------------------------------

def stabilize_root_yaw(
    root_rot_seq,
    up_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    forward_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    mode="lock_ref",
    max_step_deg=10.0,
):
    """
    root_rot_seq: (N, 3, 3)

    mode:
      - lock_ref: 모든 프레임의 heading을 첫 프레임 heading으로 맞춤
      - clamp_step: 프레임 간 yaw 변화량만 제한

    return:
      fixed_root_rot_seq, debug_info
    """
    n = root_rot_seq.shape[0]
    fixed = root_rot_seq.copy().astype(np.float64)

    headings = []
    for i in range(n):
        fwd = fixed[i] @ forward_axis
        fwd_proj = project_to_plane(fwd, up_axis)
        headings.append(normalize(fwd_proj))
    headings = np.stack(headings, axis=0)

    ref_heading = headings[0].copy()
    yaw_corrections_deg = []

    if mode == "lock_ref":
        for i in range(n):
            cur_heading = headings[i]
            angle = signed_angle_on_plane(cur_heading, ref_heading, up_axis)
            corr = rotation_matrix_from_axis_angle(up_axis, angle)
            fixed[i] = corr @ fixed[i]
            yaw_corrections_deg.append(float(np.rad2deg(angle)))

    elif mode == "clamp_step":
        max_step_rad = np.deg2rad(max_step_deg)

        prev_heading = headings[0].copy()
        yaw_corrections_deg.append(0.0)

        for i in range(1, n):
            cur_fwd = fixed[i] @ forward_axis
            cur_heading = normalize(project_to_plane(cur_fwd, up_axis))

            raw_angle = signed_angle_on_plane(prev_heading, cur_heading, up_axis)
            clamped_angle = np.clip(raw_angle, -max_step_rad, max_step_rad)

            # raw_angle이 너무 크면 그 차이만큼 역보정
            correction_needed = clamped_angle - raw_angle
            corr = rotation_matrix_from_axis_angle(up_axis, correction_needed)
            fixed[i] = corr @ fixed[i]

            new_fwd = fixed[i] @ forward_axis
            prev_heading = normalize(project_to_plane(new_fwd, up_axis))

            yaw_corrections_deg.append(float(np.rad2deg(correction_needed)))

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # after headings
    fixed_headings = []
    for i in range(n):
        fwd = fixed[i] @ forward_axis
        fwd_proj = project_to_plane(fwd, up_axis)
        fixed_headings.append(normalize(fwd_proj))
    fixed_headings = np.stack(fixed_headings, axis=0)

    debug = {
        "mode": mode,
        "max_step_deg": float(max_step_deg),
        "yaw_correction_deg_min": float(np.min(yaw_corrections_deg)),
        "yaw_correction_deg_max": float(np.max(yaw_corrections_deg)),
        "yaw_correction_deg_mean": float(np.mean(np.abs(yaw_corrections_deg))),
    }

    return fixed.astype(np.float32), debug


# ---------------------------------------------------
# Diagnostics
# ---------------------------------------------------

def compute_root_rot_delta_deg(root_rot_seq):
    n = root_rot_seq.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        rel = root_rot_seq[i - 1].T @ root_rot_seq[i]
        out[i] = np.rad2deg(R.from_matrix(rel).magnitude())
    return out


def summarize_before_after(before_root, after_root):
    before_delta = compute_root_rot_delta_deg(before_root)
    after_delta = compute_root_rot_delta_deg(after_root)

    summary = {
        "before_root_rot_delta_deg": {
            "min": float(np.min(before_delta)),
            "max": float(np.max(before_delta)),
            "mean": float(np.mean(before_delta)),
            "std": float(np.std(before_delta)),
        },
        "after_root_rot_delta_deg": {
            "min": float(np.min(after_delta)),
            "max": float(np.max(after_delta)),
            "mean": float(np.mean(after_delta)),
            "std": float(np.std(after_delta)),
        },
    }
    return summary


# ---------------------------------------------------
# Main fix
# ---------------------------------------------------

def fix_motion_v4(motion, mode="lock_ref", max_step_deg=10.0):
    fixed = {
        "global_orient": motion["global_orient"].copy(),
        "body_pose": motion["body_pose"].copy(),
        "betas": motion["betas"].copy(),
        "transl": motion["transl"].copy(),
        "fps": motion["fps"],
    }

    g = fixed["global_orient"]

    if not (g.ndim == 4 and g.shape[1:] == (1, 3, 3)):
        raise RuntimeError(
            f"Expected global_orient shape (N,1,3,3), got {g.shape}"
        )

    root_rot = g[:, 0].astype(np.float64)

    # 회전만 보정
    fixed_root_rot, debug = stabilize_root_yaw(
        root_rot_seq=root_rot,
        up_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        forward_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mode=mode,
        max_step_deg=max_step_deg,
    )

    fixed["global_orient"][:, 0] = fixed_root_rot.astype(np.float32)

    summary = summarize_before_after(root_rot.astype(np.float32), fixed_root_rot.astype(np.float32))
    summary["yaw_stabilization"] = debug

    return fixed, summary


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="lock_ref",
        choices=["lock_ref", "clamp_step"],
        help="lock_ref: 첫 프레임 방향으로 고정 / clamp_step: 프레임 간 회전량 제한",
    )
    parser.add_argument(
        "--max_step_deg",
        type=float,
        default=10.0,
        help="mode=clamp_step 일 때 프레임 간 최대 yaw 변화량",
    )
    args = parser.parse_args()

    npz_path = args.npz if args.npz else find_latest_npz()
    raw_dir = os.path.dirname(npz_path)

    motion = load_motion(npz_path)

    fixed, summary = fix_motion_v4(
        motion=motion,
        mode=args.mode,
        max_step_deg=args.max_step_deg,
    )

    out_npz = os.path.join(raw_dir, "motion_fixed_v4.npz")
    out_json = os.path.join(raw_dir, "motion_fixed_v4_summary.json")

    save_motion_npz(out_npz, fixed)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[DONE]")


if __name__ == "__main__":
    main()