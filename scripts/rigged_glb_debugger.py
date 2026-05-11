import os
import sys
import glob
import json
import numpy as np
import torch
import trimesh
from smplx import SMPL
from scipy.spatial.transform import Rotation as R

try:
    from pygltflib import GLTF2
    HAS_PYGLTFLIB = True
except Exception:
    HAS_PYGLTFLIB = False

# chumpy path
sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"
SMPL_MODEL_DIR = os.path.expanduser("~/.cache/4DHumans/data/smpl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------
# helpers
# ---------------------------------------------------

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(to_serializable(data), f, indent=2)


def print_section(title):
    print("\n" + "=" * 10, title, "=" * 10)


# ---------------------------------------------------
# latest paths
# ---------------------------------------------------

def find_latest_raw_dir():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )
    if not raw_dirs:
        raise RuntimeError("raw_* folder not found")
    return raw_dirs[0]


def find_latest_npz():
    raw_dir = find_latest_raw_dir()
    npz_path = os.path.join(raw_dir, "motion_data.npz")
    if not os.path.exists(npz_path):
        raise RuntimeError(f"motion_data.npz not found in {raw_dir}")
    return npz_path


def find_latest_glb(raw_dir):
    candidates = [
        os.path.join(raw_dir, "motion_rigged.glb"),
        os.path.join(raw_dir, "motion.glb"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------
# load npz
# ---------------------------------------------------

def load_motion(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    motion = {
        "global_orient": data["global_orient"],
        "body_pose": data["body_pose"],
        "betas": data["betas"],
        "transl": data["transl"],
        "fps": int(data["fps"][0]) if "fps" in data else 30,
    }
    return motion


def inspect_motion_npz(motion):
    print_section("NPZ INSPECTION")

    info = {}
    for k in ["global_orient", "body_pose", "betas", "transl"]:
        arr = motion[k]
        info[k] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}, min={info[k]['min']:.6f}, max={info[k]['max']:.6f}")

    info["fps"] = motion["fps"]
    print("fps:", motion["fps"])
    return info


# ---------------------------------------------------
# rotation utils
# ---------------------------------------------------

def rotmat_to_axis_angle(rotmat):
    rotmat = np.asarray(rotmat)
    flat = rotmat.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()
    return aa.reshape(*rotmat.shape[:-2], 3)


def rotmat_to_quat_xyzw(rotmat):
    rotmat = np.asarray(rotmat)
    flat = rotmat.reshape(-1, 3, 3)
    quat = R.from_matrix(flat).as_quat()
    return quat.reshape(*rotmat.shape[:-2], 4)


def convert_rotation_basis(rotmat):
    C = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)
    return C @ rotmat @ C.T


# ---------------------------------------------------
# coordinate correction (same as exporter)
# ---------------------------------------------------

def correct_coordinate_system(verts, transl):
    verts = verts.copy()
    transl = transl.copy()

    transl = transl * 0.01

    verts[:, [1, 2]] = verts[:, [2, 1]]
    verts[:, 2] *= -1

    transl[[1, 2]] = transl[[2, 1]]
    transl[2] *= -1

    return verts, transl


def correct_joints_coordinate_system(joints):
    joints = joints.copy()
    joints[:, [1, 2]] = joints[:, [2, 1]]
    joints[:, 2] *= -1
    return joints


def normalize_mesh_center(verts_seq):
    center = verts_seq[0].mean(axis=0)
    verts_seq = verts_seq - center[None, None, :]
    return verts_seq, center


# ---------------------------------------------------
# build rest pose (same as exporter)
# ---------------------------------------------------

def build_smpl_and_rest(motion):
    print_section("REST POSE")

    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        gender="neutral",
        batch_size=1
    ).to(DEVICE)

    betas_mean = motion["betas"].mean(axis=0, keepdims=True)

    betas_t = torch.tensor(betas_mean, dtype=torch.float32, device=DEVICE)
    zero_global = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)
    zero_body = torch.zeros((1, 69), dtype=torch.float32, device=DEVICE)
    zero_transl = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        out = smpl(
            betas=betas_t,
            global_orient=zero_global,
            body_pose=zero_body,
            transl=zero_transl
        )

    rest_vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    rest_joints = out.joints[0].detach().cpu().numpy().astype(np.float32)

    rest_vertices, _ = correct_coordinate_system(rest_vertices, np.zeros(3, dtype=np.float32))
    rest_joints = correct_joints_coordinate_system(rest_joints)

    faces = smpl.faces.astype(np.uint32)
    parents = smpl.parents.detach().cpu().numpy().astype(np.int32)
    lbs_weights = smpl.lbs_weights.detach().cpu().numpy().astype(np.float32)

    print("rest_vertices:", rest_vertices.shape)
    print("rest_joints:", rest_joints.shape)
    print("faces:", faces.shape)
    print("parents:", parents.shape)
    print("lbs_weights:", lbs_weights.shape)
    print("rest bbox min:", rest_vertices.min(axis=0))
    print("rest bbox max:", rest_vertices.max(axis=0))

    return smpl, rest_vertices, rest_joints, faces, parents, lbs_weights, betas_mean


# ---------------------------------------------------
# reconstruct debug mesh seq (same as exporter)
# ---------------------------------------------------

def generate_debug_mesh_sequence(motion, betas_mean):
    print_section("SMPL DEBUG SEQUENCE")

    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        gender="neutral",
        batch_size=1
    ).to(DEVICE)

    n_frames = motion["transl"].shape[0]
    verts_seq = []

    for i in range(n_frames):
        gR = motion["global_orient"][i]
        bR = motion["body_pose"][i]
        transl = motion["transl"][i].copy()

        global_orient_aa = rotmat_to_axis_angle(gR)[0]
        body_pose_aa = rotmat_to_axis_angle(bR).reshape(-1)

        global_orient = torch.tensor(global_orient_aa[None], dtype=torch.float32, device=DEVICE)
        body_pose = torch.tensor(body_pose_aa[None], dtype=torch.float32, device=DEVICE)
        betas_t = torch.tensor(betas_mean, dtype=torch.float32, device=DEVICE)
        transl_t = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            out = smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_t,
                transl=transl_t
            )

        verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        verts, transl_corr = correct_coordinate_system(verts, transl)
        verts += transl_corr

        verts_seq.append(verts)

    verts_seq = np.stack(verts_seq, axis=0)
    verts_seq, center = normalize_mesh_center(verts_seq)

    print("verts_seq:", verts_seq.shape)
    print("center:", center)
    print("frame0 bbox min:", verts_seq[0].min(axis=0))
    print("frame0 bbox max:", verts_seq[0].max(axis=0))

    return verts_seq, center


def analyze_mesh_sequence(verts_seq):
    print_section("MESH ANALYSIS")

    heights_y = []
    bbox_min = []
    bbox_max = []

    for verts in verts_seq:
        heights_y.append(float(verts[:, 1].max() - verts[:, 1].min()))
        bbox_min.append(verts.min(axis=0))
        bbox_max.append(verts.max(axis=0))

    heights_y = np.array(heights_y)
    bbox_min = np.stack(bbox_min)
    bbox_max = np.stack(bbox_max)

    result = {
        "height_y_mean": float(heights_y.mean()),
        "height_y_std": float(heights_y.std()),
        "height_y_min": float(heights_y.min()),
        "height_y_max": float(heights_y.max()),
        "frame0_bbox_min": bbox_min[0],
        "frame0_bbox_max": bbox_max[0],
        "all_bbox_min": bbox_min.min(axis=0),
        "all_bbox_max": bbox_max.max(axis=0),
    }

    for k, v in result.items():
        print(k, ":", v)

    return result


# ---------------------------------------------------
# skeleton hierarchy
# ---------------------------------------------------

def debug_skeleton_hierarchy(parents, rest_joints):
    print_section("SKELETON HIERARCHY")

    parents_24 = parents[:24]
    rest_joints_24 = rest_joints[:24]

    root_indices = np.where(parents_24 == -1)[0]
    hierarchy_ok = True
    cycle_detected = False

    print("root_indices:", root_indices.tolist())
    if len(root_indices) != 1:
        hierarchy_ok = False

    for i, p in enumerate(parents_24):
        if p == -1:
            print(f"joint {i}: ROOT")
        else:
            print(f"joint {i}: parent {p}")

    for i in range(len(parents_24)):
        seen = set()
        cur = i
        while cur != -1:
            if cur in seen:
                cycle_detected = True
                hierarchy_ok = False
                break
            seen.add(cur)
            cur = parents_24[cur]

    result = {
        "root_indices": root_indices,
        "hierarchy_ok": hierarchy_ok,
        "cycle_detected": cycle_detected,
        "parents_24": parents_24,
        "rest_joints_24": rest_joints_24,
    }

    print("hierarchy_ok:", hierarchy_ok)
    print("cycle_detected:", cycle_detected)
    return result


# ---------------------------------------------------
# skin weights
# ---------------------------------------------------

def prepare_skin_weights(lbs_weights_24):
    top4_idx = np.argsort(lbs_weights_24, axis=1)[:, -4:]
    top4_w = np.take_along_axis(lbs_weights_24, top4_idx, axis=1)

    order = np.argsort(top4_w, axis=1)[:, ::-1]
    top4_idx = np.take_along_axis(top4_idx, order, axis=1)
    top4_w = np.take_along_axis(top4_w, order, axis=1)

    sums = top4_w.sum(axis=1, keepdims=True)
    top4_w = np.divide(top4_w, np.maximum(sums, 1e-8))

    return top4_idx.astype(np.uint16), top4_w.astype(np.float32)


def debug_skin_weights(lbs_weights):
    print_section("SKIN WEIGHTS")

    lbs_weights_24 = lbs_weights[:, :24]
    joints_u16, weights_f32 = prepare_skin_weights(lbs_weights_24)

    sums = weights_f32.sum(axis=1)
    result = {
        "shape_joints": joints_u16.shape,
        "shape_weights": weights_f32.shape,
        "weight_sum_min": float(sums.min()),
        "weight_sum_max": float(sums.max()),
        "weight_sum_mean": float(sums.mean()),
        "zero_weight_vertices": int(np.sum(sums < 1e-6)),
        "joint_index_min": int(joints_u16.min()),
        "joint_index_max": int(joints_u16.max()),
        "bad_joint_index": bool(np.any(joints_u16 >= 24)),
        "sample_vertex0_joints": joints_u16[0],
        "sample_vertex0_weights": weights_f32[0],
    }

    for k, v in result.items():
        print(k, ":", v)

    return joints_u16, weights_f32, result


# ---------------------------------------------------
# bind matrices
# ---------------------------------------------------

def make_inverse_bind_matrices(rest_joints, parents):
    n_joints = len(parents)
    global_mats = []

    for j in range(n_joints):
        T = np.eye(4, dtype=np.float32)

        if parents[j] == -1:
            T[:3, 3] = rest_joints[j]
        else:
            T[:3, 3] = rest_joints[j] - rest_joints[parents[j]]

        if parents[j] == -1:
            G = T
        else:
            G = global_mats[parents[j]] @ T

        global_mats.append(G)

    global_mats = np.stack(global_mats, axis=0)
    inv_bind = np.linalg.inv(global_mats).astype(np.float32)
    return global_mats, inv_bind


def debug_bind_matrices(rest_joints, parents):
    print_section("BIND MATRICES")

    bind, inv_bind = make_inverse_bind_matrices(rest_joints[:24], parents[:24])

    errors = []
    for i in range(len(bind)):
        err = float(np.abs(bind[i] @ inv_bind[i] - np.eye(4, dtype=np.float32)).max())
        errors.append(err)
        print(f"joint {i} bind_check_error: {err:.8f}")

    result = {
        "bind_shape": bind.shape,
        "inv_bind_shape": inv_bind.shape,
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
    }

    print("bind_shape:", bind.shape)
    print("inv_bind_shape:", inv_bind.shape)
    print("max_error:", result["max_error"])
    print("mean_error:", result["mean_error"])

    return bind, inv_bind, result


# ---------------------------------------------------
# animation tracks (same as exporter)
# ---------------------------------------------------

def build_animation_tracks(motion):
    n_frames = motion["transl"].shape[0]
    n_joints = 24

    times = np.arange(n_frames, dtype=np.float32) / float(motion["fps"])

    joint_rotations = [np.zeros((n_frames, 4), dtype=np.float32) for _ in range(n_joints)]
    root_translations = np.zeros((n_frames, 3), dtype=np.float32)

    for f in range(n_frames):
        gR = motion["global_orient"][f][0]
        bR = motion["body_pose"][f]
        transl = motion["transl"][f].copy()

        gR = convert_rotation_basis(gR)
        bR = np.stack([convert_rotation_basis(x) for x in bR], axis=0)

        joint_rotations[0][f] = rotmat_to_quat_xyzw(gR)[None][0]

        body_quats = rotmat_to_quat_xyzw(bR)
        for j in range(23):
            joint_rotations[j + 1][f] = body_quats[j]

        _, transl_corr = correct_coordinate_system(
            np.zeros((1, 3), dtype=np.float32),
            transl
        )
        root_translations[f] = transl_corr

    root_origin = root_translations[0].copy()
    root_translations = root_translations - root_origin[None, :]

    return times, root_translations, joint_rotations


def debug_animation_tracks(motion):
    print_section("ANIMATION TRACKS")

    times, root_translations, joint_rotations = build_animation_tracks(motion)

    quat_norms = []
    for j in range(24):
        quat_norms.append(np.linalg.norm(joint_rotations[j], axis=1))
    quat_norms = np.concatenate(quat_norms)

    result = {
        "frames": len(times),
        "fps": motion["fps"],
        "joint_count": 24,
        "translation_track_count": 1,   # exporter 기준
        "rotation_track_count": 24,
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "quat_norm_min": float(quat_norms.min()),
        "quat_norm_max": float(quat_norms.max()),
        "root_translation_min": root_translations.min(axis=0),
        "root_translation_max": root_translations.max(axis=0),
    }

    for k, v in result.items():
        print(k, ":", v)

    return times, root_translations, joint_rotations, result


# ---------------------------------------------------
# obj export
# ---------------------------------------------------

def export_debug_obj_sequence(verts_seq, faces, out_dir, limit=30):
    print_section("OBJ EXPORT")

    obj_dir = os.path.join(out_dir, "debug_obj")
    os.makedirs(obj_dir, exist_ok=True)

    for i in range(min(limit, len(verts_seq))):
        mesh = trimesh.Trimesh(vertices=verts_seq[i], faces=faces, process=False)
        mesh.export(os.path.join(obj_dir, f"{i:04d}.obj"))

    print("exported:", obj_dir)
    return obj_dir


# ---------------------------------------------------
# inspect actual GLB
# ---------------------------------------------------

def inspect_glb(glb_path):
    print_section("GLB INSPECTION")

    if glb_path is None:
        print("No GLB found")
        return {"found": False}

    if not HAS_PYGLTFLIB:
        print("pygltflib not installed; skip actual inspection")
        return {"found": True, "path": glb_path, "inspected": False}

    gltf = GLTF2().load(glb_path)

    translation_channels = 0
    rotation_channels = 0
    scale_channels = 0

    if gltf.animations:
        for anim in gltf.animations:
            for ch in anim.channels:
                path = ch.target.path
                if path == "translation":
                    translation_channels += 1
                elif path == "rotation":
                    rotation_channels += 1
                elif path == "scale":
                    scale_channels += 1

    result = {
        "found": True,
        "path": glb_path,
        "inspected": True,
        "meshes": len(gltf.meshes or []),
        "nodes": len(gltf.nodes or []),
        "skins": len(gltf.skins or []),
        "animations": len(gltf.animations or []),
        "accessors": len(gltf.accessors or []),
        "bufferViews": len(gltf.bufferViews or []),
        "translation_channels_actual": translation_channels,
        "rotation_channels_actual": rotation_channels,
        "scale_channels_actual": scale_channels,
    }

    for k, v in result.items():
        print(k, ":", v)

    return result


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():
    raw_dir = find_latest_raw_dir()
    npz_path = find_latest_npz()
    glb_path = find_latest_glb(raw_dir)

    print("RAW_DIR:", raw_dir)
    print("NPZ:", npz_path)
    print("GLB:", glb_path)

    motion = load_motion(npz_path)
    npz_info = inspect_motion_npz(motion)

    smpl, rest_vertices, rest_joints, faces, parents, lbs_weights, betas_mean = build_smpl_and_rest(motion)

    verts_seq_debug, center = generate_debug_mesh_sequence(motion, betas_mean)
    mesh_info = analyze_mesh_sequence(verts_seq_debug)

    # exporter와 동일하게 center normalize
    rest_vertices = rest_vertices - center[None, :]
    rest_joints = rest_joints - center[None, :]

    skeleton_info = debug_skeleton_hierarchy(parents, rest_joints)

    joints_u16, weights_f32, skin_info = debug_skin_weights(lbs_weights)

    bind, inv_bind, bind_info = debug_bind_matrices(rest_joints, parents)

    times, root_translations, joint_rotations, anim_info = debug_animation_tracks(motion)

    obj_dir = export_debug_obj_sequence(verts_seq_debug, faces, raw_dir, limit=30)

    glb_info = inspect_glb(glb_path)

    summary = {
        "raw_dir": raw_dir,
        "npz_path": npz_path,
        "glb_path": glb_path,
        "npz_info": npz_info,
        "mesh_info": mesh_info,
        "skeleton_info": skeleton_info,
        "skin_info": skin_info,
        "bind_info": bind_info,
        "anim_info": anim_info,
        "glb_info": glb_info,
    }

    save_json(os.path.join(raw_dir, "debug_summary.json"), summary)

    save_json(os.path.join(raw_dir, "debug_skeleton.json"), {
        "parents_24": parents[:24],
        "rest_joints_24": rest_joints[:24],
        "center": center,
    })

    np.savez_compressed(
        os.path.join(raw_dir, "debug_weights.npz"),
        joints_u16=joints_u16,
        weights_f32=weights_f32,
        lbs_weights_24=lbs_weights[:, :24]
    )

    np.savez_compressed(
        os.path.join(raw_dir, "debug_bind_matrices.npz"),
        bind=bind,
        inv_bind=inv_bind
    )

    np.savez_compressed(
        os.path.join(raw_dir, "debug_animation.npz"),
        times=times,
        root_translations=root_translations,
        joint_rotations=np.stack(joint_rotations, axis=0)
    )

    print_section("DONE")
    print("Saved:")
    print("-", os.path.join(raw_dir, "debug_summary.json"))
    print("-", os.path.join(raw_dir, "debug_skeleton.json"))
    print("-", os.path.join(raw_dir, "debug_weights.npz"))
    print("-", os.path.join(raw_dir, "debug_bind_matrices.npz"))
    print("-", os.path.join(raw_dir, "debug_animation.npz"))
    print("-", obj_dir)


if __name__ == "__main__":
    main()