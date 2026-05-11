import numpy as np
import os

# ---------------------------------------------------
# 설정
# ---------------------------------------------------
BASE_DIR = "/home/yungbopark/gpu-worker"
TARGET_DIR = f"{BASE_DIR}/output/raw_20260316_052713"

INPUT_FILE = "motion_fixed_v5_safe.npz"
OUTPUT_FILE = "motion_fixed_clean.npz"


# ---------------------------------------------------
# load
# ---------------------------------------------------
def load_npz():
    path = os.path.join(TARGET_DIR, INPUT_FILE)
    print("[INFO] loading:", path)
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------
# spike detection (transl 기반)
# ---------------------------------------------------
def build_spike_mask(motion):

    transl = motion["transl"]  # (N, 3)

    y = transl[:, 1]

    # 속도 (프레임 차이)
    vel = np.abs(np.diff(y, prepend=y[0]))

    # 기준 (경험값)
    threshold = 0.3

    mask = vel > threshold

    print("[INFO] spike threshold:", threshold)
    print("[INFO] spike count:", np.sum(mask))

    return mask


# ---------------------------------------------------
# segment 찾기
# ---------------------------------------------------
def find_bad_segments(mask, min_len=5):

    segments = []
    start = None

    for i, m in enumerate(mask):
        if m and start is None:
            start = i

        elif not m and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None

    return segments


# ---------------------------------------------------
# mask 확장 (중요)
# ---------------------------------------------------
def expand_mask(mask, expand=5):

    new_mask = mask.copy()

    for i in range(len(mask)):
        if mask[i]:
            s = max(0, i - expand)
            e = min(len(mask), i + expand)
            new_mask[s:e] = True

    return new_mask


# ---------------------------------------------------
# 데이터 제거
# ---------------------------------------------------
def remove_segments(data, mask):

    keep_mask = ~mask
    return data[keep_mask]


# ---------------------------------------------------
# main 처리
# ---------------------------------------------------
def clean_npz():

    motion = load_npz()

    print("[INFO] original frames:", len(motion["transl"]))

    # ---------------------------------------------------
    # 1. spike 탐지
    # ---------------------------------------------------
    mask = build_spike_mask(motion)

    # ---------------------------------------------------
    # 2. 확장 (중요)
    # ---------------------------------------------------
    mask = expand_mask(mask, expand=5)

    print("[INFO] expanded spike count:", np.sum(mask))

    # ---------------------------------------------------
    # 3. segment 확인
    # ---------------------------------------------------
    segments = find_bad_segments(mask)

    print("[INFO] bad segments:", segments[:10])

    # ---------------------------------------------------
    # 4. 제거
    # ---------------------------------------------------
    cleaned = {}

    for key in motion.keys():

        if key == "fps":
            cleaned[key] = motion[key]
            continue

        data = motion[key]

        cleaned[key] = remove_segments(data, mask)

        print(f"[INFO] {key}: {data.shape} -> {cleaned[key].shape}")

    # ---------------------------------------------------
    # 5. 저장
    # ---------------------------------------------------
    out_path = os.path.join(TARGET_DIR, OUTPUT_FILE)

    np.savez(out_path, **cleaned)

    print("\n[SAVED]", out_path)


# ---------------------------------------------------
if __name__ == "__main__":
    clean_npz()