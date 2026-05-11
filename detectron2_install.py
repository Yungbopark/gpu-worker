#!/usr/bin/env python3
import os
import subprocess
import sys

PY = sys.executable
BASE = "/home/yungbopark/gpu-worker"
D2_REPO = f"{BASE}/detectron2"

def run(cmd):
    print("[RUN]", cmd)
    subprocess.check_call(cmd, shell=True)

print("\n==============================")
print(" DETECTRON2 v0.6 AUTO INSTALL ")
print("==============================\n")

# 0) basic tools
run(f"{PY} -m pip install setuptools wheel")

# 1) 기본 deps 설치
print("[1] Installing core dependencies...")
run(f"{PY} -m pip install -U pip setuptools wheel ninja pybind11")

# 2) torch + torchvision 설치 (CUDA 11.8 + torch 2.0.1 안정 조합)
print("[2] Installing PyTorch CUDA118 (2.0.1)...")
run(f"""{PY} -m pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html
""")

# 3) detectron2 repo 클론
print("[3] Cloning detectron2 v0.6...")
run(f"rm -rf {D2_REPO}")
run(f"git clone https://github.com/facebookresearch/detectron2.git {D2_REPO}")
run(f"cd {D2_REPO} && git checkout v0.6")

# 4) detectron2 의존성 설치
print("[4] Installing required dependencies...")
run(
    f'{PY} -m pip install '
    'fvcore==0.1.5.post20221221 '
    'iopath==0.1.9 '
    'cloudpickle '
    'pycocotools '
    'opencv-python-headless '
    'hydra-core '
    '"omegaconf<2.4" '
    'timm '
)

# 5) detectron2 editable 설치
print("[5] Installing detectron2 in editable mode...")
run(f"cd {D2_REPO} && {PY} -m pip install -e . --no-build-isolation --no-deps")

print("\n===================================")
print(" 🎉 Detectron2 v0.6 installation OK!")
print("   Test with: python3 - << 'EOF'")
print(" import torch, detectron2")
print(" print(torch.__version__)")
print(" print(detectron2.__version__)")
print(" print(torch.cuda.is_available())")
print("EOF")
print("===================================\n")