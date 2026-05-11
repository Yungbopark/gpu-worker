import argparse
import os

import numpy as np


cv2 = None


SMPL_24 = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}


ANGLE_DEFS = {
    "LK": ("left_hip", "left_knee", "left_ankle", "left_knee"),
    "RK": ("right_hip", "right_knee", "right_ankle", "right_knee"),
    "LE": ("left_shoulder", "left_elbow", "left_wrist", "left_elbow"),
    "RE": ("right_shoulder", "right_elbow", "right_wrist", "right_elbow"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay SMPL joint angles on rendered PNG frames.")
    parser.add_argument("--input_npz", required=True, help="Path to motion_stable.npz.")
    parser.add_argument("--png_dir", required=True, help="Directory containing frame_*.png files.")
    parser.add_argument("--output_dir", required=True, help="Directory for overlay_old PNG output.")
    parser.add_argument("--knee_hyperextension_threshold", type=float, default=175.0)
    parser.add_argument("--margin_ratio", type=float, default=0.08)
    parser.add_argument("--standing_smooth_window", type=int, default=5)
    parser.add_argument("--standing_min_consecutive", type=int, default=3)
    parser.add_argument("--standing_mean_flexion_threshold", type=float, default=15.0)
    parser.add_argument("--standing_max_flexion_threshold", type=float, default=25.0)
    parser.add_argument("--standing_hip_height_threshold", type=float, default=0.85)
    parser.add_argument(
        "--show_lr_average",
        action="store_true",
        help="Also draw left/right average values for knees and elbows.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cv2():
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read and write PNG overlays") from exc
    return cv2_module


def load_joints(input_npz):
    if not os.path.exists(input_npz):
        raise FileNotFoundError("input_npz not found: %s" % input_npz)

    data = np.load(input_npz, allow_pickle=True)
    if "joints3d" not in data.files:
        raise RuntimeError("input_npz missing joints3d")

    joints3d = np.asarray(data["joints3d"], dtype=np.float32)
    if joints3d.ndim != 3 or joints3d.shape[2] != 3:
        raise RuntimeError("joints3d must have shape (F, J, 3), got %s" % (joints3d.shape,))
    if joints3d.shape[1] <= max(SMPL_24.values()):
        raise RuntimeError("joints3d must contain at least 24 SMPL joints, got %d" % joints3d.shape[1])

    return joints3d


def list_png_frames(png_dir):
    if not os.path.isdir(png_dir):
        raise NotADirectoryError("png_dir not found: %s" % png_dir)

    names = [
        name for name in os.listdir(png_dir)
        if name.lower().endswith(".png") and name.startswith("frame_")
    ]
    names.sort()
    if not names:
        raise RuntimeError("No frame_*.png files found in %s" % png_dir)

    return [os.path.join(png_dir, name) for name in names]


def normalize_vector(v, eps=1e-8):
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return None
    return np.asarray(v, dtype=np.float32) / norm


def compute_body_axes(joints3d_frame, eps=1e-8):
    pelvis = joints3d_frame[SMPL_24["pelvis"]]
    left_hip = joints3d_frame[SMPL_24["left_hip"]]
    right_hip = joints3d_frame[SMPL_24["right_hip"]]
    neck = joints3d_frame[SMPL_24["neck"]]

    x_axis = normalize_vector(right_hip - left_hip, eps)
    if x_axis is None:
        return None

    y_raw = normalize_vector(neck - pelvis, eps)
    if y_raw is None:
        return None

    z_axis = normalize_vector(np.cross(x_axis, y_raw), eps)
    if z_axis is None:
        return None

    y_axis = normalize_vector(np.cross(z_axis, x_axis), eps)
    if y_axis is None:
        return None

    return {
        "origin": pelvis.astype(np.float32),
        "x": x_axis.astype(np.float32),
        "y": y_axis.astype(np.float32),
        "z": z_axis.astype(np.float32),
    }


def project_to_sagittal(v, axes):
    x_axis = axes["x"]
    return np.asarray(v, dtype=np.float32) - float(np.dot(v, x_axis)) * x_axis


def compute_sagittal_angle(a, b, c, axes, eps=1e-8):
    if axes is None:
        return np.nan

    v1 = project_to_sagittal(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32), axes)
    v2 = project_to_sagittal(np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32), axes)

    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < eps:
        return np.nan

    cross_norm = float(np.linalg.norm(np.cross(v1, v2)))
    dot_value = float(np.dot(v1, v2))
    return float(np.degrees(np.arctan2(cross_norm, dot_value)))


def compute_frame_angles(frame_joints):
    axes = compute_body_axes(frame_joints)
    angles = {}
    for label, (a_name, b_name, c_name, anchor_name) in ANGLE_DEFS.items():
        a = frame_joints[SMPL_24[a_name]]
        b = frame_joints[SMPL_24[b_name]]
        c = frame_joints[SMPL_24[c_name]]
        angles[label] = {
            "angle": compute_sagittal_angle(a, b, c, axes),
            "anchor": SMPL_24[anchor_name],
            "points": (SMPL_24[a_name], SMPL_24[b_name], SMPL_24[c_name]),
        }
    return angles


def moving_average_nan(values, window):
    values = np.asarray(values, dtype=np.float32)
    if window <= 1:
        return values.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    smoothed = np.zeros_like(values, dtype=np.float32)

    for idx in range(values.shape[0]):
        chunk = padded[idx:idx + window]
        finite = chunk[np.isfinite(chunk)]
        smoothed[idx] = float(np.mean(finite)) if finite.size else np.nan

    return smoothed


def normalize_01(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=np.float32)

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    return (values - min_value) / max(max_value - min_value, eps)


def require_consecutive(mask, min_frames):
    mask = np.asarray(mask, dtype=bool)
    if min_frames <= 1:
        return mask.copy()

    stable = np.zeros_like(mask, dtype=bool)
    run_start = None

    for idx, value in enumerate(mask):
        if value and run_start is None:
            run_start = idx

        is_last = idx == len(mask) - 1
        if run_start is not None and ((not value) or is_last):
            run_end = idx if not value else idx + 1
            if run_end - run_start >= min_frames:
                stable[run_start:run_end] = True
            run_start = None

    return stable


def finite_pair_mean(left_values, right_values):
    pair = np.stack([left_values, right_values], axis=1)
    output = np.full(pair.shape[0], np.nan, dtype=np.float32)
    for idx in range(pair.shape[0]):
        finite = pair[idx][np.isfinite(pair[idx])]
        if finite.size:
            output[idx] = float(np.mean(finite))
    return output


def finite_pair_max(left_values, right_values):
    pair = np.stack([left_values, right_values], axis=1)
    output = np.full(pair.shape[0], np.nan, dtype=np.float32)
    for idx in range(pair.shape[0]):
        finite = pair[idx][np.isfinite(pair[idx])]
        if finite.size:
            output[idx] = float(np.max(finite))
    return output


def compute_standing_detection(
    joints3d,
    angles_by_frame,
    smooth_window,
    min_consecutive,
    mean_flexion_threshold,
    max_flexion_threshold,
    hip_height_threshold,
):
    left_knee_angle = np.asarray([angles["LK"]["angle"] for angles in angles_by_frame], dtype=np.float32)
    right_knee_angle = np.asarray([angles["RK"]["angle"] for angles in angles_by_frame], dtype=np.float32)

    left_flexion = 180.0 - left_knee_angle
    right_flexion = 180.0 - right_knee_angle
    mean_flexion = finite_pair_mean(left_flexion, right_flexion)
    max_flexion = finite_pair_max(left_flexion, right_flexion)

    left_hip = joints3d[:, SMPL_24["left_hip"], :]
    right_hip = joints3d[:, SMPL_24["right_hip"], :]
    hip_center = 0.5 * (left_hip + right_hip)
    hip_height = hip_center[:, 1]
    hip_height_norm = normalize_01(hip_height)

    mean_flexion_s = moving_average_nan(mean_flexion, smooth_window)
    max_flexion_s = moving_average_nan(max_flexion, smooth_window)
    hip_height_norm_s = moving_average_nan(hip_height_norm, smooth_window)

    raw_standing = (
        (mean_flexion_s < mean_flexion_threshold) &
        (max_flexion_s < max_flexion_threshold) &
        (hip_height_norm_s > hip_height_threshold)
    )
    raw_standing &= np.isfinite(mean_flexion_s)
    raw_standing &= np.isfinite(max_flexion_s)
    raw_standing &= np.isfinite(hip_height_norm_s)

    standing = require_consecutive(raw_standing, min_consecutive)

    print("[INFO] Standing frames:", int(np.sum(standing)), "/", int(len(standing)))
    print("[INFO] Standing thresholds mean_flexion<%.1f max_flexion<%.1f hip_height>%.2f" % (
        mean_flexion_threshold,
        max_flexion_threshold,
        hip_height_threshold,
    ))

    return {
        "standing": standing,
        "raw_standing": raw_standing,
        "mean_knee_flexion": mean_flexion_s,
        "max_knee_flexion": max_flexion_s,
        "hip_height_norm": hip_height_norm_s,
    }


def make_sequence_projector(joints3d, image_width, image_height, margin_ratio):
    joints2d = joints3d[:, :, :2].astype(np.float32).copy()
    joints2d[:, :, 1] *= -1.0

    flat = joints2d.reshape(-1, 2)
    finite_mask = np.isfinite(flat).all(axis=1)
    flat = flat[finite_mask]
    if flat.size == 0:
        raise RuntimeError("No finite joint coordinates for projection")

    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = maxs - mins

    margin_px = min(image_width, image_height) * float(margin_ratio)
    usable_width = max(float(image_width) - 2.0 * margin_px, 1.0)
    usable_height = max(float(image_height) - 2.0 * margin_px, 1.0)
    scale_x = usable_width / max(float(extent[0]), 1e-6)
    scale_y = usable_height / max(float(extent[1]), 1e-6)
    scale = min(scale_x, scale_y)

    print("[INFO] 2D projection mins:", mins.tolist())
    print("[INFO] 2D projection maxs:", maxs.tolist())
    print("[INFO] 2D projection scale:", scale)

    def project(frame_joints):
        points = frame_joints[:, :2].astype(np.float32).copy()
        points[:, 1] *= -1.0
        points = (points - center[None, :]) * scale
        points[:, 0] += image_width * 0.5
        points[:, 1] += image_height * 0.5
        return np.rint(points).astype(np.int32)

    return project


def draw_lr_average(img, frame_angles):
    normal_color = (40, 180, 60)
    text_bg = (255, 255, 255)
    pairs = [
        ("K avg", "LK", "RK"),
        ("E avg", "LE", "RE"),
    ]

    y = 28
    for label, left_key, right_key in pairs:
        values = [
            frame_angles[left_key]["angle"],
            frame_angles[right_key]["angle"],
        ]
        values = [value for value in values if np.isfinite(value)]
        if not values:
            text = "%s nan" % label
        else:
            text = "%s %.1f" % (label, float(np.mean(values)))

        text_pos = (12, y)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            img,
            (text_pos[0] - 4, text_pos[1] - th - 4),
            (text_pos[0] + tw + 4, text_pos[1] + baseline + 4),
            text_bg,
            -1,
        )
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, normal_color, 2, cv2.LINE_AA)
        y += 30


def draw_joint_overlay(img, joints2d, frame_angles, knee_threshold, show_lr_average):
    normal_color = (40, 180, 60)
    risk_color = (30, 30, 230)
    line_color = (80, 80, 80)
    text_bg = (255, 255, 255)

    for label, info in frame_angles.items():
        angle = info["angle"]
        anchor = info["anchor"]
        points = info["points"]
        color = normal_color
        if label.endswith("K") and np.isfinite(angle) and angle > knee_threshold:
            color = risk_color

        p0 = tuple(joints2d[points[0]])
        p1 = tuple(joints2d[points[1]])
        p2 = tuple(joints2d[points[2]])
        cv2.line(img, p0, p1, line_color, 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, line_color, 2, cv2.LINE_AA)
        cv2.circle(img, p1, 5, color, -1, cv2.LINE_AA)

        anchor_xy = joints2d[anchor]
        text = "%s %.1f" % (label, angle) if np.isfinite(angle) else "%s nan" % label
        text_pos = (int(anchor_xy[0] + 8), int(anchor_xy[1] - 8))
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            img,
            (text_pos[0] - 3, text_pos[1] - th - 3),
            (text_pos[0] + tw + 3, text_pos[1] + baseline + 3),
            text_bg,
            -1,
        )
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if show_lr_average:
        draw_lr_average(img, frame_angles)

    return img


def main():
    global cv2
    args = parse_args()
    cv2 = load_cv2()
    joints3d = load_joints(args.input_npz)
    png_paths = list_png_frames(args.png_dir)
    ensure_dir(args.output_dir)

    first_img = cv2.imread(png_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError("Could not read PNG: %s" % png_paths[0])
    image_height, image_width = first_img.shape[:2]
    project = make_sequence_projector(joints3d, image_width, image_height, args.margin_ratio)

    frame_count = min(joints3d.shape[0], len(png_paths))
    if frame_count < joints3d.shape[0] or frame_count < len(png_paths):
        print("[WARN] Frame count mismatch. joints=%d png=%d using=%d" % (
            joints3d.shape[0],
            len(png_paths),
            frame_count,
        ))

    for frame_idx in range(frame_count):
        img = cv2.imread(png_paths[frame_idx], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Could not read PNG: %s" % png_paths[frame_idx])

        joints2d = project(joints3d[frame_idx])
        angles = compute_frame_angles(joints3d[frame_idx])
        img = draw_joint_overlay(
            img,
            joints2d,
            angles,
            args.knee_hyperextension_threshold,
            args.show_lr_average,
        )

        out_path = os.path.join(args.output_dir, "frame_%06d.png" % frame_idx)
        cv2.imwrite(out_path, img)
        if frame_idx % 25 == 0:
            print("[INFO] Wrote overlay_old frame:", frame_idx)

    print("[OK] Overlay PNG dir:", args.output_dir)


if __name__ == "__main__":
    main()
