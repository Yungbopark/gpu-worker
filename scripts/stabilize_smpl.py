import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np


BASE_DIR = "/home/yungbopark/gpu-worker"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(SCRIPT_DIR, "4d-Humans"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "4D-Humans"))
sys.path.insert(0, f"{BASE_DIR}/chumpy")
sys.path.insert(0, f"{BASE_DIR}/4D-Humans")
sys.path.insert(0, f"{BASE_DIR}/detectron2")

DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/4DHumans/data/smpl")
DEFAULT_SMOOTH_WINDOW = 3
DEFAULT_BATCH_SIZE = 64
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal smoothing for 4DHumans SMPL params and stable mesh PNG generation."
    )
    parser.add_argument("--input_npz", required=True, help="Path to raw motion_data.npz.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to <input_dir>/stable.")
    parser.add_argument("--smooth_window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="SMPL model path for smplx.SMPL.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--render_res", type=int, default=512, help="Square PNG render resolution.")
    parser.add_argument("--focal_length", type=float, default=5000.0)
    parser.add_argument("--camera_z", type=float, default=3.0)
    parser.add_argument(
        "--framing",
        default="raw",
        choices=["raw", "sequence"],
        help="PNG framing mode. raw keeps original camera translation; sequence uses fixed sequence-level framing.",
    )
    parser.add_argument(
        "--target_extent",
        type=float,
        default=1.6,
        help="Target max 3D extent for sequence framing after fixed scaling.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_output_dir(input_npz, output_dir):
    if output_dir:
        return output_dir
    return os.path.join(os.path.dirname(os.path.abspath(input_npz)), "stable")


def load_motion(input_npz):
    if not os.path.exists(input_npz):
        raise FileNotFoundError("input_npz not found: %s" % input_npz)

    data = np.load(input_npz, allow_pickle=True)
    required = [
        "global_orient",
        "body_pose",
        "betas",
        "transl",
        "joints3d",
        "img_name",
        "person_id",
    ]
    for key in required:
        if key not in data.files:
            raise RuntimeError("motion_data.npz missing required field: %s" % key)

    motion = {}
    for key in data.files:
        motion[key] = data[key]

    return motion


def filter_single_person(motion):
    person_id = np.asarray(motion["person_id"]).reshape(-1)
    keep_mask = person_id == 0

    if not np.any(keep_mask):
        raise RuntimeError("No person_id=0 found in motion_data.npz")

    filtered = {}
    for key, value in motion.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == keep_mask.shape[0]:
            filtered[key] = arr[keep_mask]
        else:
            filtered[key] = value

    print("[INFO] Single-person filter: kept %d / %d rows" % (int(keep_mask.sum()), len(keep_mask)))
    return filtered


def sort_by_img_name(motion):
    img_name = np.asarray(motion["img_name"]).astype(str)
    order = np.argsort(img_name, kind="stable")

    sorted_motion = {}
    for key, value in motion.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == order.shape[0]:
            sorted_motion[key] = arr[order]
        else:
            sorted_motion[key] = value

    return sorted_motion


def validate_motion_shapes(motion):
    global_orient = np.asarray(motion["global_orient"])
    body_pose = np.asarray(motion["body_pose"])
    betas = np.asarray(motion["betas"])
    transl = np.asarray(motion["transl"])
    joints3d = np.asarray(motion["joints3d"])
    img_name = np.asarray(motion["img_name"])
    person_id = np.asarray(motion["person_id"])

    frame_count = global_orient.shape[0]
    expected = {
        "global_orient": (frame_count, 1, 3, 3),
        "body_pose": (frame_count, 23, 3, 3),
        "transl": (frame_count, 3),
    }

    for key, shape in expected.items():
        if np.asarray(motion[key]).shape != shape:
            raise RuntimeError("%s must have shape %s, got %s" % (key, shape, np.asarray(motion[key]).shape))

    if betas.ndim != 2 or betas.shape[0] != frame_count:
        raise RuntimeError("betas must have shape (F, B), got %s" % (betas.shape,))
    if joints3d.ndim != 3 or joints3d.shape[0] != frame_count or joints3d.shape[2] != 3:
        raise RuntimeError("joints3d must have shape (F, J, 3), got %s" % (joints3d.shape,))
    if img_name.shape[0] != frame_count:
        raise RuntimeError("img_name length mismatch: %d != %d" % (img_name.shape[0], frame_count))
    if person_id.shape[0] != frame_count:
        raise RuntimeError("person_id length mismatch: %d != %d" % (person_id.shape[0], frame_count))

    return frame_count


def moving_average_array(values, window):
    arr = np.asarray(values, dtype=np.float32)

    if window <= 1:
        return arr.copy()
    if window < 1:
        raise RuntimeError("smooth_window must be >= 1")

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    pad_width = [(pad_left, pad_right)]
    for _ in range(arr.ndim - 1):
        pad_width.append((0, 0))

    padded = np.pad(arr, pad_width, mode="edge")
    smoothed = np.zeros_like(arr, dtype=np.float32)
    for frame_idx in range(arr.shape[0]):
        smoothed[frame_idx] = padded[frame_idx:frame_idx + window].mean(axis=0)

    return smoothed


def smooth_rotation_matrices(rot_mats, window):
    if rot_mats.shape[-2:] != (3, 3):
        raise RuntimeError("rotation input must end with (3, 3), got %s" % (rot_mats.shape,))

    if window <= 1:
        return np.asarray(rot_mats, dtype=np.float32).copy()

    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError as exc:
        raise RuntimeError("scipy is required for rotation-aware smoothing") from exc

    arr = np.asarray(rot_mats, dtype=np.float32)
    original_shape = arr.shape
    frame_count = original_shape[0]
    joint_count = int(np.prod(original_shape[1:-2]))

    flat = arr.reshape(frame_count * joint_count, 3, 3)
    rotvec = R.from_matrix(flat).as_rotvec().astype(np.float32)
    rotvec = rotvec.reshape(frame_count, joint_count, 3)

    smoothed_rotvec = moving_average_array(rotvec, window)
    smoothed_flat = R.from_rotvec(smoothed_rotvec.reshape(-1, 3)).as_matrix().astype(np.float32)

    return smoothed_flat.reshape(original_shape)


def stabilize_params(motion, smooth_window):
    global_orient = np.asarray(motion["global_orient"], dtype=np.float32)
    body_pose = np.asarray(motion["body_pose"], dtype=np.float32)
    betas = np.asarray(motion["betas"], dtype=np.float32)
    transl = np.asarray(motion["transl"], dtype=np.float32)

    betas_fixed_one = np.median(betas, axis=0, keepdims=True).astype(np.float32)
    betas_fixed = np.repeat(betas_fixed_one, betas.shape[0], axis=0).astype(np.float32)

    transl_smooth = moving_average_array(transl, smooth_window)
    global_orient_smooth = smooth_rotation_matrices(global_orient, smooth_window)
    body_pose_smooth = smooth_rotation_matrices(body_pose, smooth_window)

    return {
        "global_orient": global_orient_smooth,
        "body_pose": body_pose_smooth,
        "betas": betas_fixed,
        "transl": transl_smooth,
    }


def resolve_device(device_arg):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for SMPL forward") from exc

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to cpu.")
        return "cpu"
    return device_arg


def run_smpl_forward(params, model_path, batch_size, device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for SMPL forward") from exc

    try:
        from smplx import SMPL
    except ImportError as exc:
        raise RuntimeError("smplx is required for SMPL forward") from exc

    if not os.path.exists(model_path):
        raise FileNotFoundError("SMPL model path not found: %s" % model_path)

    frame_count = params["transl"].shape[0]
    smpl = SMPL(
        model_path=model_path,
        gender="neutral",
        batch_size=min(batch_size, frame_count),
    ).to(device)
    smpl.eval()

    vertices_batches = []
    joints_batches = []

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)

            output = smpl(
                global_orient=torch.from_numpy(params["global_orient"][start:end]).to(device),
                body_pose=torch.from_numpy(params["body_pose"][start:end]).to(device),
                betas=torch.from_numpy(params["betas"][start:end]).to(device),
                transl=torch.from_numpy(params["transl"][start:end]).to(device),
                pose2rot=False,
            )

            vertices_batches.append(output.vertices.detach().cpu().numpy().astype(np.float32))
            joints_batches.append(output.joints.detach().cpu().numpy().astype(np.float32))

    vertices = np.concatenate(vertices_batches, axis=0)
    joints3d = np.concatenate(joints_batches, axis=0)
    faces = smpl.faces

    print("[INFO] Stable vertices shape:", vertices.shape)
    print("[INFO] Stable joints3d shape:", joints3d.shape)

    return vertices, joints3d, faces


def create_renderer(faces, render_res, focal_length):
    try:
        from hmr2.utils.renderer import Renderer
    except ImportError as exc:
        raise RuntimeError("Could not import 4DHumans Renderer from hmr2.utils.renderer") from exc

    cfg = SimpleNamespace(
        EXTRA=SimpleNamespace(FOCAL_LENGTH=float(focal_length)),
        MODEL=SimpleNamespace(IMAGE_SIZE=int(render_res)),
    )
    return Renderer(cfg, faces=faces)


def compute_sequence_center(local_joints, pelvis_index=0):
    if local_joints.ndim != 3 or local_joints.shape[2] != 3:
        raise RuntimeError("local_joints must have shape (F, J, 3), got %s" % (local_joints.shape,))
    if pelvis_index < 0 or pelvis_index >= local_joints.shape[1]:
        raise RuntimeError("pelvis_index out of range: %d" % pelvis_index)

    return np.median(local_joints[:, pelvis_index, :], axis=0).astype(np.float32)


def compute_sequence_scale(local_vertices, target_extent):
    if local_vertices.ndim != 3 or local_vertices.shape[2] != 3:
        raise RuntimeError("local_vertices must have shape (F, V, 3), got %s" % (local_vertices.shape,))
    if target_extent <= 0:
        raise RuntimeError("target_extent must be > 0, got %s" % target_extent)

    flat_vertices = local_vertices.reshape(-1, 3)
    mins = np.percentile(flat_vertices, 5, axis=0)
    maxs = np.percentile(flat_vertices, 95, axis=0)
    extent = float(np.max(maxs - mins))
    if not np.isfinite(extent) or extent <= 0:
        raise RuntimeError("Invalid sequence extent: %s" % extent)

    scale = float(target_extent) / extent
    print("[DEBUG] extent:", extent)
    print("[DEBUG] scale:", scale)
    print("[DEBUG] mins:", mins.tolist())
    print("[DEBUG] maxs:", maxs.tolist())

    max_scale = 3.0
    if scale > max_scale:
        print("[WARN] scale too large, clamped:", scale, "->", max_scale)
        scale = max_scale

    return scale, extent, mins.astype(np.float32), maxs.astype(np.float32)


def compute_sequence_framing(vertices, joints3d, transl, target_extent):
    local_vertices = np.asarray(vertices, dtype=np.float32) - transl[:, None, :]
    local_joints = np.asarray(joints3d, dtype=np.float32) - transl[:, None, :]

    center = compute_sequence_center(local_joints)
    scale, extent, mins, maxs = compute_sequence_scale(local_vertices, target_extent)

    print("[INFO] Sequence framing center:", center.tolist())
    print("[INFO] Sequence framing percentile mins:", mins.tolist())
    print("[INFO] Sequence framing percentile maxs:", maxs.tolist())
    print("[INFO] Sequence framing source extent:", extent)
    print("[INFO] Sequence framing scale:", scale)

    return local_vertices, center, scale


def render_png_sequence(
    vertices,
    joints3d,
    transl,
    faces,
    png_dir,
    render_res,
    focal_length,
    camera_z,
    framing,
    target_extent,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to write PNG renders") from exc

    ensure_dir(png_dir)
    renderer = create_renderer(faces, render_res, focal_length)
    zeros_cam = np.zeros(3, dtype=np.float32)
    sequence_local_vertices = None
    sequence_center = None
    sequence_scale = None

    if framing == "sequence":
        sequence_local_vertices, sequence_center, sequence_scale = compute_sequence_framing(
            vertices=vertices,
            joints3d=joints3d,
            transl=transl,
            target_extent=target_extent,
        )

    for frame_idx in range(vertices.shape[0]):
        if framing == "sequence":
            render_vertices = (sequence_local_vertices[frame_idx] - sequence_center[None, :]) * sequence_scale
            fixed_depth = camera_z * focal_length / render_res
            cam_t = np.array([0.0, 0.0, fixed_depth], dtype=np.float32)
        else:
            # SMPL output already includes transl. Renderer.render_rgba expects local vertices
            # plus camera translation, so undo transl for rendering and pass transl separately.
            render_vertices = vertices[frame_idx] - transl[frame_idx][None, :]
            cam_t = transl[frame_idx].copy() if transl is not None else zeros_cam.copy()

        rgba = renderer.render_rgba(
            render_vertices,
            cam_t=cam_t,
            camera_z=camera_z,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            render_res=[render_res, render_res],
        )
        rgb = np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        out_path = os.path.join(png_dir, "frame_%06d.png" % frame_idx)
        cv2.imwrite(out_path, rgb[:, :, ::-1])

        if frame_idx % 25 == 0:
            print("[INFO] Rendered PNG frame:", frame_idx)


def write_outputs(output_dir, params, stable_joints3d, motion, input_npz, fps, smooth_window, framing, target_extent):
    ensure_dir(output_dir)
    output_npz = os.path.join(output_dir, "motion_stable.npz")

    np.savez_compressed(
        output_npz,
        global_orient=params["global_orient"].astype(np.float32),
        body_pose=params["body_pose"].astype(np.float32),
        transl=params["transl"].astype(np.float32),
        betas=params["betas"].astype(np.float32),
        joints3d=stable_joints3d.astype(np.float32),
        img_name=np.asarray(motion["img_name"]),
        person_id=np.asarray(motion["person_id"]),
        fps=np.asarray([fps], dtype=np.float32),
    )
    print("[OK] Wrote:", output_npz)

    meta = {
        "input_npz": os.path.abspath(input_npz),
        "fps": float(fps),
        "frame_count": int(params["transl"].shape[0]),
        "smoothing_window": int(smooth_window),
        "render_framing": {
            "mode": framing,
            "target_extent": float(target_extent),
            "sequence_mode": (
                "render-only transform using fixed median pelvis center and fixed percentile bbox scale"
            ),
        },
        "smoothing": {
            "betas": "median fixed across frames",
            "transl": "moving average",
            "global_orient": "rotation matrix -> rotvec -> moving average -> rotation matrix",
            "body_pose": "rotation matrix -> rotvec -> moving average -> rotation matrix",
        },
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[OK] Wrote:", meta_path)


def resolve_fps(motion):
    if "fps" not in motion:
        return 30.0
    fps_arr = np.asarray(motion["fps"]).reshape(-1)
    if fps_arr.size == 0:
        return 30.0
    return float(fps_arr[0])


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.input_npz, args.output_dir)
    png_dir = os.path.join(output_dir, "png")

    print("[INFO] Input npz:", args.input_npz)
    print("[INFO] Output dir:", output_dir)
    print("[INFO] Smoothing window:", args.smooth_window)

    motion = load_motion(args.input_npz)
    motion = filter_single_person(motion)
    motion = sort_by_img_name(motion)
    frame_count = validate_motion_shapes(motion)
    fps = resolve_fps(motion)

    print("[INFO] Frame count:", frame_count)
    print("[INFO] FPS:", fps)

    params = stabilize_params(motion, args.smooth_window)
    device = resolve_device(args.device)
    print("[INFO] SMPL device:", device)

    stable_vertices, stable_joints3d, faces = run_smpl_forward(
        params=params,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=device,
    )

    render_png_sequence(
        vertices=stable_vertices,
        joints3d=stable_joints3d,
        transl=params["transl"],
        faces=faces,
        png_dir=png_dir,
        render_res=args.render_res,
        focal_length=args.focal_length,
        camera_z=args.camera_z,
        framing=args.framing,
        target_extent=args.target_extent,
    )

    write_outputs(
        output_dir=output_dir,
        params=params,
        stable_joints3d=stable_joints3d,
        motion=motion,
        input_npz=args.input_npz,
        fps=fps,
        smooth_window=args.smooth_window,
        framing=args.framing,
        target_extent=args.target_extent,
    )

    print("[OK] Stabilized SMPL sequence generated")
    print("[OK] PNG dir:", png_dir)


if __name__ == "__main__":
    main()
