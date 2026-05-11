import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import torch
import trimesh
from smplx import SMPL
from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Primitive,
    Attributes,
    Asset,
    Buffer,
    BufferView,
    Accessor,
    Animation,
    AnimationSampler,
    AnimationChannel,
    AnimationChannelTarget,
)

# ---------------------------------------------------
# path
# ---------------------------------------------------
sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"
SMPL_MODEL_DIR = os.path.expanduser("~/.cache/4DHumans/data/smpl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# options
# ---------------------------------------------------
FPS_OVERRIDE = 8.0
DEFAULT_FRAME_STRIDE = 3


# ---------------------------------------------------
# npz 찾기
# ---------------------------------------------------
def find_latest_npz() -> str:
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    for d in raw_dirs:
        path = os.path.join(d, "motion_data.npz")
        if os.path.exists(path):
            return path

    raise RuntimeError("npz not found")


# ---------------------------------------------------
# load
# ---------------------------------------------------
def load_motion(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
    }


def default_output_path(input_path: str) -> str:
    viewer_dir = Path(__file__).resolve().parent.parent / "viewer"
    if viewer_dir.exists():
        return str(viewer_dir / "motion_vertex_anim.glb")
    return os.path.join(os.path.dirname(input_path), "motion_vertex_anim.glb")


def parse_args():
    parser = argparse.ArgumentParser(description="Export SMPL motion npz to vertex-animated GLB.")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to motion_data.npz. If omitted, the latest raw_* output is used.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Path to output .glb. If omitted, motion_vertex_anim.glb is written next to the input npz.",
    )
    parser.add_argument(
        "--transl_mode",
        choices=["keep", "relative", "zero"],
        default="relative",
        help="How to apply transl for node translation animation.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Stride for selecting morph target frames during GLB export.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS_OVERRIDE,
        help="Playback FPS used for exported animation timing.",
    )
    return parser.parse_args()


# ---------------------------------------------------
# transl 처리
# ---------------------------------------------------
def process_transl(t: np.ndarray, mode: str = "relative") -> np.ndarray:
    t = np.asarray(t, dtype=np.float32).copy()

    if mode == "keep":
        return t
    if mode == "relative":
        return t - t[0:1]
    if mode == "zero":
        return np.zeros_like(t, dtype=np.float32)

    raise ValueError(f"Unsupported transl_mode: {mode}")


# ---------------------------------------------------
# SMPL reconstruct
# ---------------------------------------------------
def reconstruct_smpl(motion: dict, transl_input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        gender="neutral",
        batch_size=1,
    ).to(DEVICE)

    faces = smpl.faces.astype(np.uint32)

    global_orient = motion["global_orient"]
    body_pose = motion["body_pose"]
    betas = motion["betas"]

    betas = betas[0:1] if betas.ndim == 2 else betas.reshape(1, -1)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=DEVICE)

    verts_seq = []
    joints_seq = []

    for i in range(len(transl_input)):
        with torch.no_grad():
            out = smpl(
                global_orient=torch.tensor(global_orient[i:i + 1], dtype=torch.float32, device=DEVICE),
                body_pose=torch.tensor(body_pose[i:i + 1], dtype=torch.float32, device=DEVICE),
                transl=torch.tensor(transl_input[i:i + 1], dtype=torch.float32, device=DEVICE),
                betas=betas_t,
                pose2rot=False,
            )

        verts_seq.append(out.vertices[0].detach().cpu().numpy().astype(np.float32))
        joints_seq.append(out.joints[0].detach().cpu().numpy().astype(np.float32))

    return np.stack(verts_seq), np.stack(joints_seq), faces


# ---------------------------------------------------
# canonicalize
# ---------------------------------------------------
def canonicalize_local_mesh(verts_seq: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    local mesh용 canonicalize.
    global translation animation과 섞이지 않도록
    'mesh 자체'에만 적용되는 고정 보정만 수행한다.
    """
    verts = verts_seq.copy()

    # 1. 좌표계 뒤집기
    verts[:, :, 1] *= -1.0

    # 2. 첫 프레임 바닥을 기준으로 맞춤
    floor_y = float(verts[0, :, 1].min())
    verts[:, :, 1] -= floor_y

    # 3. 첫 프레임 기준 약간 띄우기
    height = float(verts[0, :, 1].max() - verts[0, :, 1].min())
    verts[:, :, 1] += height * 0.1

    return verts, floor_y, height


def canonicalize_translation(transl_seq: np.ndarray) -> np.ndarray:
    """
    verts canonicalize와 좌표계를 맞추기 위한 translation 변환.
    local mesh에서 y를 뒤집었으므로 translation도 같은 축 변환을 적용한다.
    z 값은 camera-depth 성격이 강해 viewer에서 모델을 far plane 밖으로 밀어내므로 제거한다.
    """
    t = transl_seq.copy().astype(np.float32)
    t[:, 1] *= -1.0
    t[:, 2] = 0.0
    return t


# ---------------------------------------------------
# 디버그
# ---------------------------------------------------
def find_spike_frames(values: np.ndarray) -> list[int]:
    diff = np.abs(np.diff(values))
    if diff.size == 0:
        return []
    threshold = float(diff.mean() + 2.0 * diff.std())
    if threshold <= 0:
        return []
    return (np.where(diff > threshold)[0] + 1).tolist()


def debug_transl_stats(raw_transl: np.ndarray, processed_transl: np.ndarray, transl_mode: str):
    transl_delta = (
        np.linalg.norm(np.diff(processed_transl, axis=0), axis=1)
        if len(processed_transl) > 1
        else np.zeros(1, dtype=np.float32)
    )

    print("[DEBUG] ===== TRANSL STATS =====")
    print("[DEBUG] transl_mode:", transl_mode)
    print("[DEBUG] raw transl frame0     :", raw_transl[0].tolist())
    print("[DEBUG] processed transl frame0:", processed_transl[0].tolist())
    print("[DEBUG] transl_x min/max/std:", float(processed_transl[:, 0].min()), float(processed_transl[:, 0].max()), float(processed_transl[:, 0].std()))
    print("[DEBUG] transl_y min/max/std:", float(processed_transl[:, 1].min()), float(processed_transl[:, 1].max()), float(processed_transl[:, 1].std()))
    print("[DEBUG] transl_z min/max/std:", float(processed_transl[:, 2].min()), float(processed_transl[:, 2].max()), float(processed_transl[:, 2].std()))
    print("[DEBUG] transl_delta norm min/max/std:", float(transl_delta.min()), float(transl_delta.max()), float(transl_delta.std()))
    print("[DEBUG] ===== END TRANSL STATS =====")


def debug_reconstruct_stats(verts_seq: np.ndarray, joints_seq: np.ndarray):
    verts_delta = np.linalg.norm((verts_seq - verts_seq[0]).reshape(len(verts_seq), -1), axis=1)
    joints_delta = np.linalg.norm((joints_seq - joints_seq[0]).reshape(len(joints_seq), -1), axis=1)
    pelvis_y = joints_seq[:, 0, 1]
    bbox_height = verts_seq[:, :, 1].max(axis=1) - verts_seq[:, :, 1].min(axis=1)

    print("[DEBUG] ===== RECONSTRUCT STATS =====")
    print("[DEBUG] verts delta-from-frame0 min/max/std:", float(verts_delta.min()), float(verts_delta.max()), float(verts_delta.std()))
    print("[DEBUG] joints delta-from-frame0 min/max/std:", float(joints_delta.min()), float(joints_delta.max()), float(joints_delta.std()))
    print("[DEBUG] bbox_height spike frames:", find_spike_frames(bbox_height))
    print("[DEBUG] pelvis_y spike frames:", find_spike_frames(pelvis_y))
    print("[DEBUG] ===== END RECONSTRUCT STATS =====")


def debug_motion_stats(verts_seq: np.ndarray, joints_seq: np.ndarray):
    pelvis_y = joints_seq[:, 0, 1]

    foot_indices = [i for i in [7, 8, 10, 11] if i < joints_seq.shape[1]]
    if foot_indices:
        foot_min_y = joints_seq[:, foot_indices, 1].min(axis=1)
    else:
        foot_min_y = np.zeros(len(joints_seq), dtype=np.float32)

    bbox_height = verts_seq[:, :, 1].max(axis=1) - verts_seq[:, :, 1].min(axis=1)

    print("[DEBUG] ===== MOTION STATS =====")
    print("[DEBUG] pelvis_y  min/max/std :", float(pelvis_y.min()), float(pelvis_y.max()), float(pelvis_y.std()))
    print("[DEBUG] foot_y    min/max/std :", float(foot_min_y.min()), float(foot_min_y.max()), float(foot_min_y.std()))
    print("[DEBUG] height    min/max/std :", float(bbox_height.min()), float(bbox_height.max()), float(bbox_height.std()))
    print("[DEBUG] first 10 pelvis_y     :", pelvis_y[:10].tolist())
    print("[DEBUG] first 10 foot_min_y   :", foot_min_y[:10].tolist())
    print("[DEBUG] first 10 bbox_height  :", bbox_height[:10].tolist())
    print("[DEBUG] ===== END MOTION STATS =====")


def debug_rotation_matrix_stats(name: str, mats: np.ndarray):
    mats = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    eye = np.eye(3, dtype=np.float64)
    ortho_err = np.linalg.norm(np.matmul(np.transpose(mats, (0, 2, 1)), mats) - eye, axis=(1, 2))
    det = np.linalg.det(mats)
    bad_mask = (np.abs(det - 1.0) > 1e-2) | (ortho_err > 1e-2)

    print(f"[DEBUG] ===== {name} ROTMAT STATS =====")
    print(f"[DEBUG] {name} det min/max/std:", float(det.min()), float(det.max()), float(det.std()))
    print(f"[DEBUG] {name} abs(det-1) max/mean:", float(np.abs(det - 1.0).max()), float(np.abs(det - 1.0).mean()))
    print(f"[DEBUG] {name} ortho_err max/mean:", float(ortho_err.max()), float(ortho_err.mean()))
    print(f"[DEBUG] {name} invalid_count:", int(bad_mask.sum()))
    print(f"[DEBUG] ===== END {name} ROTMAT STATS =====")
    if np.any(bad_mask):
        raise ValueError(f"{name} contains invalid rotation matrices")


# ---------------------------------------------------
# GLTF buffer builder
# ---------------------------------------------------
class GLTFBufferBuilder:
    def __init__(self):
        self.binary_blob = bytearray()
        self.buffer_views: list[BufferView] = []
        self.accessors: list[Accessor] = []

    def _pad(self):
        while len(self.binary_blob) % 4:
            self.binary_blob += b"\x00"

    def add_data(self, arr: np.ndarray, target=None) -> int:
        arr = np.ascontiguousarray(arr)
        self._pad()
        offset = len(self.binary_blob)
        raw = arr.tobytes()
        self.binary_blob += raw

        idx = len(self.buffer_views)
        self.buffer_views.append(
            BufferView(
                buffer=0,
                byteOffset=offset,
                byteLength=len(raw),
                target=target,
            )
        )
        return idx

    from typing import Optional
    def add_accessor(
        self,
        arr: np.ndarray,
        component_type: int,
        gltf_type: str,
        target=None,
        include_minmax: bool = True,
        count_override: Optional[int] = None
    ) -> int:
        arr = np.ascontiguousarray(arr)
        bv = self.add_data(arr, target=target)

        count = int(count_override) if count_override is not None else int(arr.shape[0])

        kwargs = dict(
            bufferView=bv,
            componentType=component_type,
            count=count,
            type=gltf_type,
        )

        if include_minmax and arr.size > 0 and gltf_type != "SCALAR":
            flat = arr.reshape(count, -1)
            kwargs["min"] = flat.min(axis=0).tolist()
            kwargs["max"] = flat.max(axis=0).tolist()
        elif include_minmax and arr.size > 0 and gltf_type == "SCALAR":
            scalar = arr.reshape(-1)
            kwargs["min"] = [float(scalar.min())]
            kwargs["max"] = [float(scalar.max())]

        idx = len(self.accessors)
        self.accessors.append(Accessor(**kwargs))
        return idx

    def finalize(self) -> bytes:
        self._pad()
        return bytes(self.binary_blob)


# ---------------------------------------------------
# GLB export
# ---------------------------------------------------
def compute_base_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = mesh.vertex_normals.astype(np.float32)
        if normals.shape == vertices.shape:
            return normals
    except Exception:
        pass
    return np.zeros_like(vertices, dtype=np.float32)


def export_glb(
    local_verts_seq: np.ndarray,
    node_transl_seq: np.ndarray,
    faces: np.ndarray,
    out_path: str,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    fps: float = FPS_OVERRIDE,
):
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")

    base = local_verts_seq[0].astype(np.float32)
    normals = compute_base_normals(base, faces)

    idxs = list(range(1, len(local_verts_seq), frame_stride))
    if not idxs:
        idxs = [len(local_verts_seq) - 1]
    elif idxs[-1] != len(local_verts_seq) - 1:
        idxs.append(len(local_verts_seq) - 1)

    morph_target_count = len(idxs)
    print("[DEBUG] frame_stride:", frame_stride)
    print("[DEBUG] morph target count:", morph_target_count)
    if morph_target_count > 64:
        print("[WARN] too many morph targets for viewer stability:", morph_target_count)

    # local deformation만 morph target에 넣는다.
    deltas = [(local_verts_seq[i] - base).astype(np.float32) for i in idxs]

    # keyframe 시간축
    key_times = (np.array([0] + idxs, dtype=np.float32) / float(fps)).astype(np.float32)
    keyframe_count = len(key_times)

    # node translation 애니메이션
    translation_keys = node_transl_seq[[0] + idxs].astype(np.float32)

    builder = GLTFBufferBuilder()

    # base mesh
    pos_acc = builder.add_accessor(base, 5126, "VEC3", target=34962)
    norm_acc = builder.add_accessor(normals, 5126, "VEC3", target=34962)
    idx_acc = builder.add_accessor(faces.reshape(-1).astype(np.uint32), 5125, "SCALAR", target=34963)

    # morph targets
    targets = []
    for d in deltas:
        target_pos_acc = builder.add_accessor(d, 5126, "VEC3", target=34962)
        targets.append(Attributes(POSITION=target_pos_acc))

    # animation accessors
    time_acc = builder.add_accessor(key_times, 5126, "SCALAR", include_minmax=True)

    # translation accessor
    translation_acc = builder.add_accessor(
        translation_keys,
        5126,
        "VEC3",
        include_minmax=True,
    )

    primitive = Primitive(
        attributes=Attributes(POSITION=pos_acc, NORMAL=norm_acc),
        indices=idx_acc,
        targets=targets,
    )

    mesh = Mesh(
        primitives=[primitive],
        weights=[0.0] * morph_target_count,
    )

    node = Node(mesh=0)
    scene = Scene(nodes=[0])

    # node translation animation only
    sampler_translation = AnimationSampler(
        input=time_acc,
        output=translation_acc,
        interpolation="LINEAR",
    )
    channel_translation = AnimationChannel(
        sampler=0,
        target=AnimationChannelTarget(node=0, path="translation"),
    )

    animation = Animation(
        samplers=[sampler_translation],
        channels=[channel_translation],
    )

    binary_blob = builder.finalize()

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="OpenAI-vertex-glb-export"),
        scenes=[scene],
        scene=0,
        nodes=[node],
        meshes=[mesh],
        animations=[animation],
        accessors=builder.accessors,
        bufferViews=builder.buffer_views,
        buffers=[Buffer(byteLength=len(binary_blob))],
    )

    gltf.set_binary_blob(binary_blob)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(out_path)


# ---------------------------------------------------
# main
# ---------------------------------------------------
def main():
    args = parse_args()
    path = args.input_path or find_latest_npz()
    print("NPZ:", path)
    print("transl_mode:", args.transl_mode)
    print("frame_stride:", args.frame_stride)
    print("device:", DEVICE)

    motion = load_motion(path)
    debug_rotation_matrix_stats("global_orient", motion["global_orient"])
    debug_rotation_matrix_stats("body_pose", motion["body_pose"])

    # 원본 transl 디버그
    processed_transl = process_transl(motion["transl"], mode=args.transl_mode)
    debug_transl_stats(motion["transl"], processed_transl, args.transl_mode)

    # 1) local morph용 reconstruct: transl = 0
    zero_transl = np.zeros_like(motion["transl"], dtype=np.float32)
    local_verts_seq, local_joints_seq, faces = reconstruct_smpl(motion, zero_transl)

    # 2) debug: local reconstruct 상태 확인
    debug_reconstruct_stats(local_verts_seq, local_joints_seq)
    debug_motion_stats(local_verts_seq, local_joints_seq)

    # 3) canonicalize local mesh
    canonical_local_verts, floor_y, height = canonicalize_local_mesh(local_verts_seq)

    # 4) node translation sequence 준비
    # morph에는 넣지 않고 node translation으로만 사용
    node_transl_seq = canonicalize_translation(processed_transl)

    print("[DEBUG] canonical floor_y:", floor_y)
    print("[DEBUG] canonical base height:", height)
    print("[DEBUG] node translation first 5:", node_transl_seq[:5].tolist())

    # 5) export
    out = args.output_path or default_output_path(path)
    export_glb(
        local_verts_seq=canonical_local_verts,
        node_transl_seq=node_transl_seq,
        faces=faces,
        out_path=out,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )

    print("[DONE]", out)


if __name__ == "__main__":
    main()
