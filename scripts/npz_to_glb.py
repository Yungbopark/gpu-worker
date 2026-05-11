import os
import sys
import glob
import numpy as np
import torch
import trimesh
from smplx import SMPL
from scipy.spatial.transform import Rotation as R

# chumpy import가 필요할 수 있으므로 경로 추가
sys.path.insert(0, "/home/yungbopark/gpu-worker/chumpy")

BASE_DIR = "/home/yungbopark/gpu-worker"
OUTPUT_DIR = f"{BASE_DIR}/output"
SMPL_MODEL_DIR = os.path.expanduser("~/.cache/4DHumans/data/smpl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------
# 최신 npz 찾기
# ---------------------------------------------------

def find_latest_npz():
    raw_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "raw_*")),
        key=os.path.getmtime,
        reverse=True
    )

    if not raw_dirs:
        raise RuntimeError("raw_* folder not found")

    for d in raw_dirs:
        npz = os.path.join(d, "motion_data.npz")
        if os.path.exists(npz):
            return npz

    raise RuntimeError("motion_data.npz not found in raw_* folders")


# ---------------------------------------------------
# npz 로드
# ---------------------------------------------------

def load_motion(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    motion = {
        "global_orient": data["global_orient"],  # (N,1,3,3)
        "body_pose": data["body_pose"],          # (N,23,3,3)
        "betas": data["betas"],                  # (N,10)
        "transl": data["transl"],                # (N,3)
    }

    if "fps" in data:
        motion["fps"] = int(data["fps"][0])
    else:
        motion["fps"] = 30

    return motion


# ---------------------------------------------------
# rotmat -> axis-angle
# ---------------------------------------------------

def rotmat_to_axis_angle(rotmat):
    from scipy.spatial.transform import Rotation as R

    rotmat = np.asarray(rotmat)
    flat = rotmat.reshape(-1,3,3)
    aa = R.from_matrix(flat).as_rotvec()

    return aa.reshape(*rotmat.shape[:-2],3)


# ---------------------------------------------------
# SMPL mesh sequence 생성
# ---------------------------------------------------

def generate_smpl_sequence(motion):
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        gender="neutral",
        batch_size=1
    ).to(DEVICE)

    faces = smpl.faces

    n_frames = motion["transl"].shape[0]
    verts_seq = []
    joints_seq = []

    print(f"[INFO] frames: {n_frames}")
    print(f"[INFO] device: {DEVICE}")

    for i in range(n_frames):

        gR = motion["global_orient"][i]
        bR = motion["body_pose"][i]

        betas = motion["betas"][i]
        transl = motion["transl"][i]

        # rotmat -> axis angle
        global_orient_aa = rotmat_to_axis_angle(gR)[0]  # (3,)
        body_pose_aa = rotmat_to_axis_angle(bR).reshape(-1)  # (69,)

        global_orient = torch.tensor(
            global_orient_aa[None],
            dtype=torch.float32,
            device=DEVICE
        )

        body_pose = torch.tensor(
            body_pose_aa[None],
            dtype=torch.float32,
            device=DEVICE
        )

        betas_t = torch.tensor(
            betas[None],
            dtype=torch.float32,
            device=DEVICE
        )

        transl_t = torch.tensor(
            transl[None],
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():
            out = smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_t,
                transl=transl_t
            )

        verts = out.vertices[0].detach().cpu().numpy()
        joints = out.joints[0].detach().cpu().numpy()

        verts_seq.append(verts)
        joints_seq.append(joints)

        if i == 0:
            print(f"[INFO] first frame verts shape: {verts.shape}")
            print(f"[INFO] first frame joints shape: {joints.shape}")

    verts_seq = np.stack(verts_seq, axis=0)   # (N,6890,3)
    joints_seq = np.stack(joints_seq, axis=0) # (N,J,3)

    return verts_seq, joints_seq, faces


# ---------------------------------------------------
# GLB 저장 (검증용: 첫 프레임 메쉬)
# ---------------------------------------------------

def save_glb_first_frame(verts_seq, faces, out_path):
    vertices = verts_seq[0].astype(np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(out_path)


# ---------------------------------------------------
# 참고용 npz 저장 (복원 결과 확인)
# ---------------------------------------------------

def save_debug_npz(verts_seq, joints_seq, npz_path):
    out_path = os.path.join(os.path.dirname(npz_path), "reconstructed_debug.npz")
    np.savez_compressed(
        out_path,
        verts=verts_seq,
        joints=joints_seq
    )
    print("[INFO] debug npz saved:", out_path)


# ---------------------------------------------------
# main
# ---------------------------------------------------

def main():
    npz_path = find_latest_npz()
    print("NPZ:", npz_path)

    motion = load_motion(npz_path)

    verts_seq, joints_seq, faces = generate_smpl_sequence(motion)

    out_glb = os.path.join(os.path.dirname(npz_path), "motion.glb")
    save_glb_first_frame(verts_seq, faces, out_glb)

    save_debug_npz(verts_seq, joints_seq, npz_path)

    print("GLB saved:", out_glb)


if __name__ == "__main__":
    main()