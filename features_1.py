"""
=============================================================================
PHONE CAMERA FINGERPRINT SYSTEM  —  RTX 3050 Optimized
FILE 1: features.py
=============================================================================
PRNU + Frequency Domain Feature Extraction.
Run this once to pre-extract all handcrafted CV features to disk.
=============================================================================
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path
from scipy import ndimage
from scipy.fft import fft2, fftshift
import warnings
from scipy.stats import kurtosis
warnings.filterwarnings("ignore")

DATA_DIR   = "./data"
OUTPUT_DIR = "./cv_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHONE_CLASSES = [
    "Motog64_5G", "Motog85_5G", "Nothing_A001",
    "Realme8_Pro", "Redmi14C_5G", "Xiaomi_M2101K6P"
]


# -------------------------------------------------------
# WIENER FILTER — gold standard noise residual extraction
# -------------------------------------------------------
def wiener2(img: np.ndarray, k: int = 5) -> np.ndarray:
    img      = img.astype(np.float64)
    lmean    = ndimage.uniform_filter(img, k)
    lsq      = ndimage.uniform_filter(img**2, k)
    lvar     = np.maximum(lsq - lmean**2, 0)
    noise_v  = np.mean(lvar)
    denom    = np.maximum(lvar, noise_v)
    return (lmean + (lvar - noise_v) / denom * (img - lmean)).astype(np.float32)


def noise_residual(img_gray: np.ndarray) -> np.ndarray:
    """img_gray: float32 [0,1]"""
    return img_gray - wiener2(img_gray)


# -------------------------------------------------------
# FREQUENCY FEATURES  (FFT of noise residual)
# -------------------------------------------------------
def freq_features(noise: np.ndarray, n_bins: int = 32) -> np.ndarray:
    mag       = np.abs(fftshift(fft2(noise.astype(np.float64))))
    h, w      = mag.shape
    cy, cx    = h//2, w//2
    y, x      = np.mgrid[-cy:h-cy, -cx:w-cx]
    radius    = np.sqrt(x**2 + y**2).astype(int)
    max_r     = min(cy, cx)
    edges     = np.linspace(0, max_r, n_bins+1)
    rbins     = np.array([mag[(radius >= int(edges[i])) & (radius < int(edges[i+1]))].mean()
                          if ((radius >= int(edges[i])) & (radius < int(edges[i+1]))).any() else 0.0
                          for i in range(n_bins)])
    if rbins.max() > 0: rbins /= rbins.max()
    log_mag   = np.log1p(mag[:cy, :])
    stats     = np.array([log_mag.mean(), log_mag.std(),
                           log_mag.max(), np.percentile(log_mag, 95)])
    return np.concatenate([rbins, stats])   # 36-dim


# -------------------------------------------------------
# DCT FEATURES  (JPEG compression fingerprint)
# -------------------------------------------------------
def dct_features(img_gray: np.ndarray, n: int = 16) -> np.ndarray:
    img   = (img_gray * 255).astype(np.float32)
    h, w  = img.shape
    hc, wc = (h//8)*8, (w//8)*8
    img   = img[:hc, :wc]
    acc   = np.zeros((8, 8))
    cnt   = 0
    for i in range(0, hc, 8):
        for j in range(0, wc, 8):
            acc += np.abs(cv2.dct(img[i:i+8, j:j+8]))
            cnt += 1
    if cnt: acc /= cnt
    # zigzag first n coefficients
    flat = []   
    for d in range(15):
        if d % 2 == 0:
            for i in range(min(d,7), max(-1,d-8), -1): flat.append(acc[i,d-i])
        else:
            for i in range(max(0,d-7), min(d+1,8)): flat.append(acc[i,d-i])
    arr = np.array(flat[:n])
    if arr.max() > 0: arr /= arr.max()
    return arr   # 16-dim


# -------------------------------------------------------
# COLOR FEATURES
# -------------------------------------------------------
def color_features(img_bgr: np.ndarray) -> np.ndarray:
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = []
    for img_s, scale in [(rgb,[1,1,1]), (hsv/[360,1,255],[1,1,1]),
                          (lab/[100,128,128],[1,1,1])]:
        for c in range(3):
            ch = img_s[:,:,c].ravel()
            feats.extend([ch.mean(), ch.std(),
                           np.percentile(ch,25), np.percentile(ch,75)])
    r = rgb[:,:,0].mean()+1e-8; g = rgb[:,:,1].mean()+1e-8; b = rgb[:,:,2].mean()+1e-8
    feats.extend([r/g, b/g, r/b])
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    feats.extend([ycrcb[:,:,1].std(), ycrcb[:,:,2].std()])
    return np.array(feats, dtype=np.float32)   # 38-dim


# -------------------------------------------------------
# NOISE STATS
# -------------------------------------------------------
def noise_stats(noise: np.ndarray) -> np.ndarray:
    return np.array([
        noise.mean(), noise.std(), noise.var(),
        np.percentile(np.abs(noise), 95),
        float(kurtosis(noise.ravel())),
        float(np.corrcoef(noise[:-1,:].ravel(), noise[1:,:].ravel())[0,1]),
    ], dtype=np.float32)   # 6-dim


# -------------------------------------------------------
# FULL FEATURE VECTOR  (96-dim total)
# -------------------------------------------------------
def extract(img_path: str, size: int = 256) -> np.ndarray | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return None
    # center crop
    h, w = img_bgr.shape[:2]
    if h < size or w < size:
        img_bgr = cv2.resize(img_bgr, (max(w,size), max(h,size)))
        h, w = img_bgr.shape[:2]
    img_bgr = img_bgr[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    noise    = noise_residual(img_gray)
    return np.concatenate([
        freq_features(noise),    # 36
        dct_features(img_gray),  # 16
        color_features(img_bgr), # 38
        noise_stats(noise),      # 6
    ])   # 96-dim total


# -------------------------------------------------------
# BUILD DATASET
# -------------------------------------------------------
def build_dataset():
    print("Extracting PRNU + CV features...")
    X, y      = [], []
    label_map = {n: i for i, n in enumerate(PHONE_CLASSES)}

    for phone in PHONE_CLASSES:
        folder = Path(DATA_DIR) / phone
        if not folder.exists():
            print(f"  [skip] {folder}"); continue
        imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) +
                      list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        print(f"  {phone}: {len(imgs)} images", end="", flush=True)
        ok = 0
        for p in imgs:
            v = extract(str(p))
            if v is not None:
                X.append(v); y.append(label_map[phone]); ok += 1
        print(f"  → {ok} ok")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.save(f"{OUTPUT_DIR}/X_features.npy", X)
    np.save(f"{OUTPUT_DIR}/y_labels.npy",   y)
    with open(f"{OUTPUT_DIR}/label_map.json","w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nSaved: X={X.shape}  y={y.shape}  → {OUTPUT_DIR}/")
    return X, y, label_map


if __name__ == "__main__":
    build_dataset()
    print("✅  Done.  Next: python 2_train.py")