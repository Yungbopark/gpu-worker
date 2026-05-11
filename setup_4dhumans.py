#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4D-Humans / detectron2 / chumpy / smpl 모델 초기 설치 스크립트
GPU 서버(Ubuntu)에서 1회 실행하면 환경이 모두 준비됨.
Colab용 스크립트를 서버 환경에 맞도록 변환한 버전.
"""

import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

BASE_DIR = Path("/home/yungbopark/gpu-worker")
REPO_4DH = BASE_DIR / "4D-Humans"
REPO_D2 = BASE_DIR / "detectron2"
REPO_CH = BASE_DIR / "chumpy"
DATA_DIR = BASE_DIR / "data"
SMPL_DIR = DATA_DIR / "smpl"

PY = sys.executable  # 가상환경 python

def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True)

def mark(msg):
    global _t
    now = time.time()
    print(f"[TIME] {msg}: {now - _t:.2f}s")
    _t = now

_t = time.time()

# -----------------------------------------
# 1) 4D-Humans repo clone
# -----------------------------------------
if not REPO_4DH.exists():
    run(f"git clone https://github.com/shubham-goel/4D-Humans.git {REPO_4DH}")
else:
    print("[SKIP] 4D-Humans repo exists")
mark("4D-Humans clone")

# -----------------------------------------
# 2) 기본 deps 설치
# -----------------------------------------
deps = (
    "setuptools wheel ninja pybind11 fvcore iopath yacs hydra-core omegaconf "
    "smplx==0.1.28 pytorch-lightning webdataset pyrender "
    "pygltflib scipy"
)
run(f"{PY} -m pip install {deps}")
mark("basic pip deps")

# -----------------------------------------
# 3) chumpy 설치 + 패치
# -----------------------------------------
if not REPO_CH.exists():
    run(f"git clone https://github.com/mattloper/chumpy.git {REPO_CH}")
else:
    print("[SKIP] chumpy repo exists")

# py3.11+ getargspec 패치
target_ch = REPO_CH / "chumpy/ch.py"
run(f"sed -i \"s/from inspect import getargspec/from inspect import getfullargspec as getargspec/\" {target_ch} || true")

# import 테스트
sys.path.insert(0, str(REPO_CH))
try:
    import chumpy, chumpy.ch
    print("[OK] chumpy import:", chumpy.__file__)
except Exception as e:
    raise RuntimeError(f"chumpy import failed: {e}")

mark("chumpy setup")

# -----------------------------------------
# 4) detectron2 설치
# -----------------------------------------
need_d2 = False
try:
    import detectron2
    print("[SKIP] detectron2 already installed:", detectron2.__file__)
except Exception:
    need_d2 = True

if need_d2:
    if not REPO_D2.exists():
        run(f"git clone https://github.com/facebookresearch/detectron2.git {REPO_D2}")
    run(f"cd {REPO_D2} && MAX_JOBS=1 {PY} -m pip install -v . --no-build-isolation --no-deps")

mark("detectron2 setup")

# -----------------------------------------
# 5) 4D-Humans 설치
# -----------------------------------------
need_4dh = False
try:
    import hmr2
    print("[SKIP] hmr2 already importable:", hmr2.__file__)
except Exception:
    need_4dh = True

if need_4dh:
    run(f"cd {REPO_4DH} && {PY} -m pip install -e . --no-deps")

mark("4D-Humans install")

# -----------------------------------------
# 6) ffmpeg 설치
# -----------------------------------------
run("which ffmpeg || (sudo apt-get update -y && sudo apt-get install -y ffmpeg)")
mark("ffmpeg check")

# -----------------------------------------
# 7) SMPL 모델 파일 다운로드 & 구조 정리
# -----------------------------------------
SMPL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

remote_smpl = "https://github.com/classner/up/raw/821a390fbf87a522fb327fc46736eda0326e2a06/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
local_smpl = DATA_DIR / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

if not local_smpl.exists():
    run(f"wget -nc {remote_smpl} -O {local_smpl}")
else:
    print("[SKIP] SMPL base pkl exists")

neutral_pkl = SMPL_DIR / "SMPL_NEUTRAL.pkl"
if not neutral_pkl.exists():
    shutil.copy2(local_smpl, neutral_pkl)
    print("[OK] Created neutral SMPL file:", neutral_pkl)
else:
    print("[SKIP] neutral file exists")

mark("SMPL setup")

print("\n===============================")
print("🎉 4D-Humans Setup Completed!")
print(f"Python: {PY}")
print(f"4D-Humans repo: {REPO_4DH}")
print(f"Detectron2 repo: {REPO_D2}")
print(f"Chumpy repo: {REPO_CH}")
print(f"SMPL model dir: {DATA_DIR}")
print("===============================")