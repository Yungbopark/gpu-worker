import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 설정
# =========================
npz_path = "/home/yungbopark/gpu-worker/output/raw_20260406_052254/motion_data_25d_root_relative.npz"
if not os.path.exists(npz_path):
    npz_path = "./motion_data_25d_root_relative.npz"
out_dir = "./frames_25d"
os.makedirs(out_dir, exist_ok=True)


# =========================
# SMPL body-24 subset 정의
# =========================
# npz_to_25d.py는 SMPL 출력 joints(45개)를 그대로 저장한다.
# 이 중 앞쪽 24개는 body joint이며, 나머지 21개는 얼굴/발끝/손가락 보조 joint다.
# 현재 렌더링 문제는 45개 joint에 대해 임의의 edge를 적용한 것이 주원인으로 보인다.
#
# body-24 source index:
#  0 pelvis        1 left_hip      2 right_hip     3 spine1
#  4 left_knee     5 right_knee    6 spine2        7 left_ankle
#  8 right_ankle   9 spine3       10 left_foot    11 right_foot
# 12 neck         13 left_collar  14 right_collar 15 head
# 16 left_shoulder 17 right_shoulder 18 left_elbow 19 right_elbow
# 20 left_wrist   21 right_wrist  22 left_hand    23 right_hand
BODY24_INDICES = list(range(24))
BODY24_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

# SMPL body-24 kinematic tree 기반 연결.
BODY24_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def select_joint_subset(all_joints: np.ndarray) -> Tuple[np.ndarray, Sequence[int], Sequence[str]]:
    if all_joints.ndim != 3 or all_joints.shape[2] != 3:
        raise ValueError("joints25d must have shape (frames, joints, 3), got %s" % (all_joints.shape,))

    if all_joints.shape[1] < len(BODY24_INDICES):
        raise ValueError(
            "Expected at least %d joints for SMPL body-24 rendering, got %d"
            % (len(BODY24_INDICES), all_joints.shape[1])
        )

    subset = all_joints[:, BODY24_INDICES, :]
    return subset, BODY24_INDICES, BODY24_NAMES


def draw_frame(
    pts: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    title: str,
    save_path: str,
    joint_names: Sequence[str],
    annotate: bool,
) -> None:
    plt.figure(figsize=(4, 6))
    plt.title(title)
    plt.scatter(pts[:, 0], pts[:, 1], c="blue", s=20, zorder=3)

    for start_idx, end_idx in edges:
        x = [pts[start_idx, 0], pts[end_idx, 0]]
        y = [pts[start_idx, 1], pts[end_idx, 1]]
        plt.plot(x, y, c="black", linewidth=2.0, zorder=2)

    if annotate:
        for joint_idx, joint_name in enumerate(joint_names):
            plt.text(
                pts[joint_idx, 0],
                pts[joint_idx, 1],
                "%d:%s" % (joint_idx, joint_name),
                fontsize=6,
                color="crimson",
                zorder=4,
            )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# 데이터 로드
# =========================
data = np.load(npz_path)
all_joints = data["joints25d"]  # (N, J, 3)

joints, used_source_indices, used_joint_names = select_joint_subset(all_joints)
edges = BODY24_EDGES

print("frames:", len(joints))
print("joints25d original shape:", all_joints.shape)
print("render joint subset: SMPL body-24")
print("used joint count:", joints.shape[1])
print("edge count:", len(edges))
print("used source joint indices:", list(used_source_indices))
print("first frame pelvis xyz:", joints[0, 0].tolist())
print("first frame bbox xy min/max:", joints[0, :, :2].min(axis=0).tolist(), joints[0, :, :2].max(axis=0).tolist())
print("first frame edges:")
for start_idx, end_idx in edges:
    print("  %02d:%s -> %02d:%s" % (
        start_idx, used_joint_names[start_idx], end_idx, used_joint_names[end_idx]
    ))


# =========================
# 렌더링
# =========================
for i in range(len(joints)):
    pts = joints[i]
    save_path = os.path.join(out_dir, "frame_%04d.png" % i)
    draw_frame(
        pts=pts,
        edges=edges,
        title="Frame %d" % i,
        save_path=save_path,
        joint_names=used_joint_names,
        annotate=(i == 0),
    )

debug_path = os.path.join(out_dir, "frame_0000_debug.png")
draw_frame(
    pts=joints[0],
    edges=edges,
    title="Frame 0 Debug",
    save_path=debug_path,
    joint_names=used_joint_names,
    annotate=True,
)
print("first-frame debug image:", debug_path)
print("Done: PNG frames generated")
