from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import subprocess
import os
import numpy as np

app = FastAPI()

API_TOKEN = os.environ.get("GPU_WORKER_TOKEN", "your-secret-token")

BASE_DIR = "/home/yungbopark/gpu-worker"
BASE_OUTPUT = f"{BASE_DIR}/output"
UPLOAD_DIR = f"{BASE_DIR}/upload_inbox"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# 인증
# =========================================================

def check_token(token):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


# =========================================================
# health
# =========================================================

@app.get("/health")
def health(x_api_token: str = Header(None)):
    check_token(x_api_token)
    return {"status": "ok"}


# =========================================================
# 🔥 파일 업로드 (추가됨)
# =========================================================

@app.post("/upload-video")
def upload_video(file: UploadFile = File(...), x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        f.write(file.file.read())

    print("파일 저장됨:", path)

    return {"path": path}


# =========================================================
# command 실행
# =========================================================

@app.post("/run")
def run(cmd: dict, x_api_token: str = Header(None)):
    check_token(x_api_token)

    try:
        print("실행 명령:", cmd["cmd"])

        result = subprocess.run(
            cmd["cmd"],
            shell=True,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# 폴더 목록
# =========================================================

@app.get("/list-folders")
def list_folders(x_api_token: str = Header(None)):
    check_token(x_api_token)

    folders = [
        f for f in os.listdir(BASE_OUTPUT)
        if f.startswith("raw_") and os.path.isdir(os.path.join(BASE_OUTPUT, f))
    ]

    folders.sort(reverse=True)
    return folders[:20]


# =========================================================
# npz 목록
# =========================================================

@app.get("/list-npz")
def list_npz(folder: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder)

    files = [
        f for f in os.listdir(path)
        if f.endswith(".npz")
    ]

    files.sort()
    return files


# =========================================================
# npz timeseries (안전 처리 포함)
# =========================================================

@app.get("/npz-timeseries")
def npz_timeseries(folder: str, file: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder, file)

    data = np.load(path)

    if "transl" not in data:
        return {
            "error": "invalid npz",
            "keys": list(data.keys())
        }

    transl = data["transl"]
    body_pose = data["body_pose"]

    N = transl.shape[0]

    # -----------------------------
    # metric 계산
    # -----------------------------

    # 1. pelvis 높이
    pelvis_y = transl[:, 1]

    # 2. velocity (이동량)
    velocity = np.linalg.norm(np.diff(transl, axis=0), axis=1)
    velocity = np.insert(velocity, 0, 0)

    # 3. acceleration
    acceleration = np.diff(velocity)
    acceleration = np.insert(acceleration, 0, 0)

    # 4. body_pose 변화량
    pose_delta = np.linalg.norm(np.diff(body_pose, axis=0), axis=1)
    pose_delta = np.insert(pose_delta, 0, 0)

    # -----------------------------
    # 반환
    # -----------------------------

    return {
        "frames": list(range(N)),
        "pelvis_y": pelvis_y.tolist(),
        "velocity": velocity.tolist(),
        "acceleration": acceleration.tolist(),
        "body_pose_delta": pose_delta.tolist(),
    }

from fastapi.responses import FileResponse

@app.get("/list-glb")
def list_glb(folder: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder)
    files = [f for f in os.listdir(path) if f.endswith(".glb")]
    files.sort()
    return files

@app.get("/glb-file")
def glb_file(folder: str, file: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder, file)
    return FileResponse(path, media_type="model/gltf-binary")
