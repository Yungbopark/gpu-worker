import argparse
import glob
import os
import sys  # 추가
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")

import numpy as np
import torch
from smplx import SMPL


BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/4DHumans/data/smpl")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_Z_SCALE = 0.1
DEFAULT_SMOOTH_WINDOW = 1
DEBUG_FOCUS_FRAMES = [40, 44, 105, 126, 144]
PELVIS_INDEX = 0


def find_latest_npz() -> str:
    search_roots = [
        os.path.join(OUTPUT_DIR, "raw_*"),
        os.path.join(os.getcwd(), "raw_*"),
    ]
    candidates = []
    for pattern in search_roots:
        for raw_dir in glob.glob(pattern):
            path = os.path.join(raw_dir, "motion_data.npz")
            if os.path.exists(path):
                candidates.append(path)

    if not candidates:
        raise RuntimeError("motion_data.npz not found")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_input_path(input_path: Optional[str]) -> str:
    if not input_path:
        return find_latest_npz()

    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        npz_path = path / "motion_data.npz"
        if not npz_path.exists():
            raise FileNotFoundError("motion_data.npz not found in directory: %s" % str(path))
        return str(npz_path)

    if not path.exists():
        raise FileNotFoundError("input path not found: %s" % str(path))

    return str(path)


def default_output_path(input_npz_path: str) -> str:
    input_path = Path(input_npz_path)
    return str(input_path.with_name("motion_data_25d.npz"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert 4D-Humans motion_data.npz to a stable 2.5D npz.")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to motion_data.npz or raw_* directory. If omitted, the latest motion_data.npz is used.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Path to output .npz. If omitted, motion_data_25d.npz is written next to the input npz.",
    )
    parser.add_argument(
        "--mode",
        choices=["flatten", "scaled_z", "root_relative"],
        default="root_relative",
        help="2.5D transform mode. root_relative is the recommended default.",
    )
    parser.add_argument(
        "--z_scale",
        type=float,
        default=DEFAULT_Z_SCALE,
        help="Scale factor applied to z when mode keeps any depth.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Temporal moving-average window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override. If omitted, uses fps from the input npz when available.",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="SMPL model path used for joint reconstruction.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device, for example cpu or cuda.",
    )
    return parser.parse_args()


def load_motion(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    motion = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
    }
    if "fps" in data.files:
        motion["fps"] = np.asarray(data["fps"])
    if "img_name" in data.files:
        motion["img_name"] = np.asarray(data["img_name"])
    if "person_id" in data.files:
        motion["person_id"] = np.asarray(data["person_id"])
    return motion


def resolve_fps(motion: Dict[str, np.ndarray], fps_override: Optional[float]) -> float:
    if fps_override is not None:
        return float(fps_override)
    if "fps" in motion:
        fps_arr = np.asarray(motion["fps"]).reshape(-1)
        if fps_arr.size > 0:
            return float(fps_arr[0])
    return 30.0


def validate_rotation_matrices(name: str, mats: np.ndarray) -> None:
    flat = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    identity = np.eye(3, dtype=np.float64)
    ortho_err = np.linalg.norm(np.matmul(np.transpose(flat, (0, 2, 1)), flat) - identity, axis=(1, 2))
    det = np.linalg.det(flat)
    invalid_count = int(np.sum((np.abs(det - 1.0) > 1e-2) | (ortho_err > 1e-2)))

    print("[DEBUG] %s det min/max/std: %.8f %.8f %.8f" % (name, float(det.min()), float(det.max()), float(det.std())))
    print("[DEBUG] %s ortho_err max/mean: %.8f %.8f" % (name, float(ortho_err.max()), float(ortho_err.mean())))
    print("[DEBUG] %s invalid_count: %d" % (name, invalid_count))

    if invalid_count > 0:
        raise ValueError("%s contains invalid rotation matrices" % name)


def reconstruct_joints_local(
    motion: Dict[str, np.ndarray],
    model_path: str,
    device: str,
) -> np.ndarray:
    smpl = SMPL(
        model_path=model_path,
        gender="neutral",
        batch_size=1,
    ).to(device)

    global_orient = motion["global_orient"]
    body_pose = motion["body_pose"]
    betas = motion["betas"]
    zero_transl = np.zeros_like(motion["transl"], dtype=np.float32)

    if betas.ndim == 2:
        betas_one = betas[0:1]
    else:
        betas_one = betas.reshape(1, -1)
    betas_t = torch.tensor(betas_one, dtype=torch.float32, device=device)

    joints_seq = []
    for frame_idx in range(len(zero_transl)):
        with torch.no_grad():
            out = smpl(
                global_orient=torch.tensor(global_orient[frame_idx:frame_idx + 1], dtype=torch.float32, device=device),
                body_pose=torch.tensor(body_pose[frame_idx:frame_idx + 1], dtype=torch.float32, device=device),
                transl=torch.tensor(zero_transl[frame_idx:frame_idx + 1], dtype=torch.float32, device=device),
                betas=betas_t,
                pose2rot=False,
            )
        joints_seq.append(out.joints[0].detach().cpu().numpy().astype(np.float32))

    return np.stack(joints_seq)


def canonicalize_points(points_seq: np.ndarray) -> Tuple[np.ndarray, float, float]:
    points = np.asarray(points_seq, dtype=np.float32).copy()
    points[:, :, 1] *= -1.0

    floor_y = float(points[0, :, 1].min())
    points[:, :, 1] -= floor_y

    height = float(points[0, :, 1].max() - points[0, :, 1].min())
    points[:, :, 1] += height * 0.1
    return points, floor_y, height


def moving_average(seq: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(seq, dtype=np.float32).copy()

    if window < 1:
        raise ValueError("smooth_window must be >= 1")

    arr = np.asarray(seq, dtype=np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, ((pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)

    for frame_idx in range(arr.shape[0]):
        out[frame_idx] = padded[frame_idx:frame_idx + window].mean(axis=0)

    return out


def transform_joints_to_25d(joints3d: np.ndarray, mode: str, z_scale: float) -> np.ndarray:
    joints = np.asarray(joints3d, dtype=np.float32).copy()

    if mode == "flatten":
        joints -= joints[:, PELVIS_INDEX:PELVIS_INDEX + 1, :]
        joints[:, :, 2] = 0.0
        return joints

    if mode == "scaled_z":
        joints[:, :, 2] *= float(z_scale)
        return joints

    if mode == "root_relative":
        joints -= joints[:, PELVIS_INDEX:PELVIS_INDEX + 1, :]
        joints[:, :, 2] *= float(z_scale)
        return joints

    raise ValueError("Unsupported mode: %s" % mode)


def compute_bbox(points_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bbox_min = points_seq.min(axis=1)
    bbox_max = points_seq.max(axis=1)
    bbox_size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) * 0.5
    return bbox_min, bbox_max, bbox_size, center


def compute_frame_deltas(points_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta_from_frame0 = np.linalg.norm((points_seq - points_seq[0]).reshape(points_seq.shape[0], -1), axis=1)
    if points_seq.shape[0] <= 1:
        delta_from_prev = np.zeros(1, dtype=np.float32)
    else:
        prev = np.linalg.norm(np.diff(points_seq, axis=0).reshape(points_seq.shape[0] - 1, -1), axis=1)
        delta_from_prev = np.concatenate([np.zeros(1, dtype=np.float32), prev.astype(np.float32)], axis=0)
    return delta_from_frame0.astype(np.float32), delta_from_prev.astype(np.float32)


def format_vec(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join("%.6f" % value for value in flat.tolist()) + "]"


def log_basic_stats(name: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64)
    print(
        "[DEBUG] %s min/max/std: %.6f %.6f %.6f"
        % (name, float(arr.min()), float(arr.max()), float(arr.std()))
    )


def log_focus_frames(
    focus_frames: List[int],
    pelvis25d: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    bbox_size: np.ndarray,
    center25d: np.ndarray,
    delta_from_frame0: np.ndarray,
    delta_from_prev: np.ndarray,
) -> None:
    print("[DEBUG] ===== FOCUS FRAMES =====")
    total_frames = pelvis25d.shape[0]
    for frame_idx in focus_frames:
        if frame_idx < 0 or frame_idx >= total_frames:
            print("[DEBUG] frame %d: out of range for total frames=%d" % (frame_idx, total_frames))
            continue

        print("[DEBUG] frame %d pelvis25d: %s" % (frame_idx, format_vec(pelvis25d[frame_idx])))
        print("[DEBUG] frame %d bbox25d_min: %s" % (frame_idx, format_vec(bbox_min[frame_idx])))
        print("[DEBUG] frame %d bbox25d_max: %s" % (frame_idx, format_vec(bbox_max[frame_idx])))
        print("[DEBUG] frame %d bbox25d_size: %s" % (frame_idx, format_vec(bbox_size[frame_idx])))
        print("[DEBUG] frame %d center25d: %s" % (frame_idx, format_vec(center25d[frame_idx])))
        print("[DEBUG] frame %d joints delta-from-frame0: %.6f" % (frame_idx, float(delta_from_frame0[frame_idx])))
        print("[DEBUG] frame %d joints delta-from-prev: %.6f" % (frame_idx, float(delta_from_prev[frame_idx])))
    print("[DEBUG] ===== END FOCUS FRAMES =====")


def save_25d_npz(
    out_path: str,
    frames: np.ndarray,
    fps: float,
    joints25d: np.ndarray,
    pelvis25d: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    bbox_size: np.ndarray,
    center25d: np.ndarray,
    mode: str,
    z_scale: float,
    smooth_window: int,
    original_joints3d: np.ndarray,
    delta_from_frame0: np.ndarray,
    delta_from_prev: np.ndarray,
    focus_frames: List[int],
    source_input: str,
    floor_y: float,
    base_height: float,
) -> None:
    original_bbox_min, original_bbox_max, original_bbox_size, original_center = compute_bbox(original_joints3d)

    np.savez_compressed(
        out_path,
        frames=frames.astype(np.int32),
        fps=np.asarray([fps], dtype=np.float32),
        joints25d=joints25d.astype(np.float32),
        pelvis25d=pelvis25d.astype(np.float32),
        bbox25d_min=bbox_min.astype(np.float32),
        bbox25d_max=bbox_max.astype(np.float32),
        bbox25d_size=bbox_size.astype(np.float32),
        center25d=center25d.astype(np.float32),
        transform_mode=np.asarray(mode),
        z_scale=np.asarray([z_scale], dtype=np.float32),
        smoothing_window=np.asarray([smooth_window], dtype=np.int32),
        original_joints3d_min=original_joints3d.min(axis=(0, 1)).astype(np.float32),
        original_joints3d_max=original_joints3d.max(axis=(0, 1)).astype(np.float32),
        original_joints3d_std=original_joints3d.std(axis=(0, 1)).astype(np.float32),
        original_bbox3d_min=original_bbox_min.astype(np.float32),
        original_bbox3d_max=original_bbox_max.astype(np.float32),
        original_bbox3d_size=original_bbox_size.astype(np.float32),
        original_center3d=original_center.astype(np.float32),
        joints25d_delta_from_frame0=delta_from_frame0.astype(np.float32),
        joints25d_delta_from_prev=delta_from_prev.astype(np.float32),
        focus_frames_requested=np.asarray(focus_frames, dtype=np.int32),
        source_input=np.asarray(source_input),
        pelvis_index=np.asarray([PELVIS_INDEX], dtype=np.int32),
        canonical_floor_y=np.asarray([floor_y], dtype=np.float32),
        canonical_base_height=np.asarray([base_height], dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    input_npz_path = resolve_input_path(args.input_path)
    output_npz_path = args.output_path or default_output_path(input_npz_path)

    print("Input:", input_npz_path)
    print("Output:", output_npz_path)
    print("Mode:", args.mode)
    print("z_scale:", float(args.z_scale))
    print("smooth_window:", int(args.smooth_window))
    print("device:", args.device)
    print("model_path:", args.model_path)

    motion = load_motion(input_npz_path)
    fps = resolve_fps(motion, args.fps)

    validate_rotation_matrices("global_orient", motion["global_orient"])
    validate_rotation_matrices("body_pose", motion["body_pose"])

    joints3d_local = reconstruct_joints_local(
        motion=motion,
        model_path=args.model_path,
        device=args.device,
    )
    joints3d_local, floor_y, base_height = canonicalize_points(joints3d_local)

    joints25d = transform_joints_to_25d(
        joints3d=joints3d_local,
        mode=args.mode,
        z_scale=float(args.z_scale),
    )
    joints25d = moving_average(joints25d, int(args.smooth_window))

    frames = np.arange(joints25d.shape[0], dtype=np.int32)
    pelvis25d = joints25d[:, PELVIS_INDEX, :]
    bbox_min, bbox_max, bbox_size, center25d = compute_bbox(joints25d)
    delta_from_frame0, delta_from_prev = compute_frame_deltas(joints25d)

    print("[DEBUG] frames:", int(joints25d.shape[0]))
    print("[DEBUG] fps:", float(fps))
    print("[DEBUG] joints25d shape:", tuple(joints25d.shape))
    print("[DEBUG] z compression active:", "yes" if args.mode != "flatten" else "no")
    log_basic_stats("joints25d", joints25d)
    log_basic_stats("bbox25d_y_range", bbox_size[:, 1])
    log_basic_stats("joints25d_delta_from_prev", delta_from_prev)
    log_basic_stats("joints25d_delta_from_frame0", delta_from_frame0)

    log_focus_frames(
        focus_frames=DEBUG_FOCUS_FRAMES,
        pelvis25d=pelvis25d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size=bbox_size,
        center25d=center25d,
        delta_from_frame0=delta_from_frame0,
        delta_from_prev=delta_from_prev,
    )

    Path(output_npz_path).parent.mkdir(parents=True, exist_ok=True)
    save_25d_npz(
        out_path=output_npz_path,
        frames=frames,
        fps=fps,
        joints25d=joints25d,
        pelvis25d=pelvis25d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size=bbox_size,
        center25d=center25d,
        mode=args.mode,
        z_scale=float(args.z_scale),
        smooth_window=int(args.smooth_window),
        original_joints3d=joints3d_local,
        delta_from_frame0=delta_from_frame0,
        delta_from_prev=delta_from_prev,
        focus_frames=DEBUG_FOCUS_FRAMES,
        source_input=input_npz_path,
        floor_y=floor_y,
        base_height=base_height,
    )

    print("[DONE]", output_npz_path)


if __name__ == "__main__":
    main()
