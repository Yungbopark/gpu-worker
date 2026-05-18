from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import os
import numpy as np
from pathlib import Path

app = FastAPI()

API_TOKEN = os.environ.get("GPU_WORKER_TOKEN", "your-secret-token")

BASE_DIR = "/home/yungbopark/gpu-worker"
BASE_OUTPUT = f"{BASE_DIR}/output"
UPLOAD_DIR = f"{BASE_DIR}/upload_inbox"

ALLOWED_RESULT_FILES = {
    "thumbnail.jpg": "image/jpeg",
    "output.mp4": "video/mp4",
    "analysis.json": "application/json",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# 인증
# =========================================================

def check_token(token):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


class ResultFileError(Exception):
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body


def is_safe_name(name):
    if not name:
        return False
    if name in {".", ".."}:
        return False
    if "/" in name or "\\" in name:
        return False
    if ".." in name:
        return False
    return True


def get_result_media_type(filename):
    return ALLOWED_RESULT_FILES[filename]


def get_result_file_path(folder, filename):
    if not is_safe_name(folder):
        raise ResultFileError(400, {"error": "invalid folder"})

    if not is_safe_name(filename):
        raise ResultFileError(400, {"error": "invalid file"})

    if filename not in ALLOWED_RESULT_FILES:
        raise ResultFileError(400, {"error": "file not allowed"})

    base_path = Path(BASE_OUTPUT).resolve()
    target_path = (base_path / folder / filename).resolve()

    if target_path != base_path and base_path not in target_path.parents:
        raise ResultFileError(400, {"error": "invalid result path"})

    if not target_path.is_file():
        raise ResultFileError(
            404,
            {
                "error": "file not found",
                "folder": folder,
                "file": filename,
            },
        )

    return target_path


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

@app.get("/list-glb")
def list_glb(folder: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder)
    files = [f for f in os.listdir(path) if f.endswith(".glb")]
    files.sort()
    return files

@app.get("/result-file")
def result_file(folder: str, file: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    try:
        path = get_result_file_path(folder, file)
    except ResultFileError as e:
        return JSONResponse(status_code=e.status_code, content=e.body)

    return FileResponse(path, media_type=get_result_media_type(file))

@app.get("/glb-file")
def glb_file(folder: str, file: str, x_api_token: str = Header(None)):
    check_token(x_api_token)

    path = os.path.join(BASE_OUTPUT, folder, file)
    return FileResponse(path, media_type="model/gltf-binary")
