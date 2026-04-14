"""
=============================================================================
PHONE CAMERA FINGERPRINT SYSTEM
FILE 5: analysis.py  —  Full Quantitative Analysis
=============================================================================
For every prediction this generates:

SECTION 1 — IMAGE PROPERTIES
  Resolution, aspect ratio, file size, estimated JPEG quality

SECTION 2 — EMBEDDING ANALYSIS
  Query embedding vs every gallery prototype:
  - Cosine similarity
  - Euclidean distance
  - Manhattan (L1) distance
  - Pearson correlation
  - Margin gap (winner vs runner-up)
  - Z-score of winning similarity

SECTION 3 — FEATURE VECTOR COMPARISON (96-dim breakdown)
  Query CV feature vector vs gallery phone feature vectors:
  - Per-component cosine similarity (freq, DCT, color, noise)
  - Chi-squared histogram distance
  - Bhattacharyya distance
  - Component-wise absolute difference heatmap

SECTION 4 — PRNU CORRELATION
  Direct sensor noise pattern correlation (forensics standard)
  - NCC (Normalized Cross-Correlation) vs each phone's PRNU fingerprint
  - Peak-to-correlation energy ratio

SECTION 5 — FREQUENCY DOMAIN
  FFT magnitude spectrum comparison
  - Radial power spectral density
  - Dominant frequency components
  - Spectral entropy

SECTION 6 — COLOR SCIENCE
  Per-channel statistics vs gallery phone baselines
  - Channel mean/std deviation
  - White balance ratios
  - Saturation / brightness distribution

SECTION 7 — DECISION EVIDENCE
  - Similarity score matrix (all phones)
  - Rank ordering with margin analysis
  - Confidence interpretation
  - Anomaly flags (if any metric contradicts the top prediction)

OUTPUT:
  - Full matplotlib report (12-panel PDF-quality figure)
  - JSON report with every number
  - Console report with interpretation

Usage:
  python 5_analysis.py --image photo.jpg
  python 5_analysis.py --image photo.jpg --threshold 0.50 --save_json
  python 5_analysis.py --folder ./test_photos/ --save_json
=============================================================================
"""

import os, sys, json, argparse, warnings
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from scipy import ndimage
from scipy.fft import fft2, fftshift
from scipy.stats import pearsonr, entropy as scipy_entropy
from PIL import Image, ExifTags
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from train_2   import CameraNet, PhoneDataset, EMB_DIM, IMG_SIZE, PHONE_CLASSES
from gallery_3 import load_model, load_gallery, _val_transform
from features_1 import (wiener2, noise_residual, freq_features,
                         dct_features, color_features, noise_stats, extract)

OUTPUT_DIR = Path("./cv_outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── colour palette (forensic / scientific) ────────────────────────────────
C_MATCH   = "#00d4aa"
C_NOMATCH = "#ff4757"
C_NEUTRAL = "#a8b2d8"
C_ACCENT  = "#ffd32a"
C_BG      = "#0d1117"
C_PANEL   = "#161b22"
C_TEXT    = "#e6edf3"
C_GRID    = "#21262d"


# =============================================================================
# SECTION 1 — IMAGE PROPERTIES
# =============================================================================
def image_properties(img_path: Path) -> dict:
    img_pil  = Image.open(img_path)
    img_bgr  = cv2.imread(str(img_path))
    file_sz  = img_path.stat().st_size

    props = {
        "filename":    img_path.name,
        "width":       img_pil.width,
        "height":      img_pil.height,
        "aspect_ratio": round(img_pil.width / img_pil.height, 4),
        "file_size_kb": round(file_sz / 1024, 2),
        "mode":        img_pil.mode,
        "format":      img_pil.format or img_path.suffix.upper().strip("."),
    }

    # Estimate JPEG quality from quantization tables
    try:
        if hasattr(img_pil, "quantization") and img_pil.quantization:
            q_table = list(img_pil.quantization.values())[0]
            props["jpeg_quality_est"] = round(
                max(0, min(100, (100 - np.mean(q_table) * 0.5))), 1)
        else:
            props["jpeg_quality_est"] = "N/A"
    except Exception:
        props["jpeg_quality_est"] = "N/A"

    # EXIF
    exif_data = {}
    try:
        raw_exif = img_pil._getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag in ("Make","Model","Software","DateTime",
                           "FocalLength","ISOSpeedRatings","ExposureTime",
                           "FNumber","Flash","WhiteBalance"):
                    exif_data[tag] = str(val)
    except Exception:
        pass
    props["exif"] = exif_data

    # Basic pixel statistics
    if img_bgr is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        props["mean_luminance"]  = round(float(gray.mean()), 3)
        props["std_luminance"]   = round(float(gray.std()),  3)
        props["dynamic_range"]   = round(float(gray.max()-gray.min()), 1)
        props["megapixels"]      = round(img_pil.width*img_pil.height/1e6, 3)

    return props


# =============================================================================
# SECTION 2 — EMBEDDING ANALYSIS
# =============================================================================
def embedding_analysis(query_emb: np.ndarray,
                        gallery: dict) -> dict:
    results = {}
    sims    = []

    for name, proto in gallery.items():
        q = query_emb.astype(np.float64)
        p = proto.astype(np.float64)

        cos_sim  = float(np.dot(q, p))   # both L2-normalized already
        euc_dist = float(np.linalg.norm(q - p))
        l1_dist  = float(np.sum(np.abs(q - p)))
        try:
            pearson  = float(pearsonr(q, p)[0])
        except Exception:
            pearson  = 0.0

        results[name] = {
            "cosine_similarity":  round(cos_sim,   6),
            "euclidean_distance": round(euc_dist,  6),
            "l1_distance":        round(l1_dist,   6),
            "pearson_correlation":round(pearson,   6),
        }
        sims.append((name, cos_sim))

    ranked = sorted(sims, key=lambda x: x[1], reverse=True)
    top_name, top_sim = ranked[0]

    # Margin gap  — difference between top and runner-up
    margin = (top_sim - ranked[1][1]) if len(ranked) > 1 else top_sim

    # Z-score of top similarity
    all_sims = [s for _,s in sims]
    z_score  = (top_sim - np.mean(all_sims)) / (np.std(all_sims)+1e-9)

    # Normalised confidence (softmax over cosine similarities)
    temps    = np.array(all_sims)
    softmax  = np.exp(temps*10) / np.exp(temps*10).sum()
    conf_map = {ranked[i][0]: round(float(softmax[
                    [r[0] for r in ranked].index(ranked[i][0])]),4)
                for i in range(len(ranked))}

    return {
        "per_phone":   results,
        "ranked":      [(n, round(s,6)) for n,s in ranked],
        "top_match":   top_name,
        "top_sim":     round(top_sim,  6),
        "margin_gap":  round(margin,   6),
        "z_score":     round(z_score,  4),
        "softmax_conf":conf_map,
    }


# =============================================================================
# SECTION 3 — FEATURE VECTOR COMPARISON
# =============================================================================
def build_gallery_feature_vectors(data_dir: str,
                                   n_per_class: int = 30) -> dict:
    """
    Compute mean CV feature vector for each phone from training data.
    Cached to disk so we only compute once.
    """
    cache = Path("./cv_outputs/gallery_feature_vectors.json")
    if cache.exists():
        with open(cache) as f:
            d = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k,v in d.items()}

    print("  Building gallery feature vectors (first run)...")
    EXTS    = {".jpg",".jpeg",".png",".JPG",".JPEG"}
    vectors = {}
    for phone in PHONE_CLASSES:
        folder = Path(data_dir) / phone
        if not folder.exists(): continue
        imgs = [p for p in sorted(folder.iterdir()) if p.suffix in EXTS][:n_per_class]
        vecs = [v for p in imgs if (v := extract(str(p))) is not None]
        if vecs:
            vectors[phone] = np.array(vecs).mean(0).tolist()
    with open(cache,"w") as f:
        json.dump(vectors, f)
    return {k: np.array(v, dtype=np.float32) for k,v in vectors.items()}


def feature_vector_comparison(query_vec: np.ndarray,
                                gallery_vecs: dict) -> dict:
    """
    Compare query 96-dim feature vector against each gallery phone's mean.
    Feature layout: [freq(36) | dct(16) | color(38) | noise(6)]
    """
    SLICES = {
        "freq_features":  slice(0,  36),
        "dct_features":   slice(36, 52),
        "color_features": slice(52, 90),
        "noise_stats":    slice(90, 96),
    }

    results = {}
    for name, gvec in gallery_vecs.items():
        q = query_vec.astype(np.float64)
        g = gvec.astype(np.float64)

        # Overall distances
        cos  = float(np.dot(q,g) / (np.linalg.norm(q)*np.linalg.norm(g)+1e-9))
        l2   = float(np.linalg.norm(q-g))
        l1   = float(np.sum(np.abs(q-g)))

        # Per-component cosine similarity
        comp_cos = {}
        for comp_name, sl in SLICES.items():
            qc = q[sl]; gc = g[sl]
            nq = np.linalg.norm(qc); ng = np.linalg.norm(gc)
            comp_cos[comp_name] = round(float(np.dot(qc,gc)/(nq*ng+1e-9)), 6)

        # Chi-squared distance on normalised histogram (freq component)
        q_hist = np.abs(q[SLICES["freq_features"]])
        g_hist = np.abs(g[SLICES["freq_features"]])
        q_hist /= q_hist.sum()+1e-9; g_hist /= g_hist.sum()+1e-9
        chi2 = float(np.sum((q_hist-g_hist)**2 / (g_hist+1e-9)))

        # Bhattacharyya distance
        bc   = float(-np.log(np.sum(np.sqrt(q_hist*g_hist))+1e-9))

        # Component-wise absolute difference
        abs_diff = np.abs(q - g).tolist()

        results[name] = {
            "overall_cosine":     round(cos, 6),
            "l2_distance":        round(l2,  6),
            "l1_distance":        round(l1,  6),
            "component_cosine":   comp_cos,
            "chi2_freq":          round(chi2, 6),
            "bhattacharyya_freq": round(bc,   6),
            "abs_diff_vector":    abs_diff,
        }

    return results


# =============================================================================
# SECTION 4 — PRNU CORRELATION
# =============================================================================
def build_prnu_gallery(data_dir: str, n_per_class: int = 20,
                        prnu_size: int = 256) -> dict:
    """
    Compute PRNU fingerprint for each phone. Cached.
    """
    cache = Path("./cv_outputs/prnu_gallery.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("  Building PRNU gallery (first run, ~1 min)...")
    EXTS = {".jpg",".jpeg",".png",".JPG",".JPEG"}
    prnu_dict = {}
    for phone in PHONE_CLASSES:
        folder = Path(data_dir) / phone
        if not folder.exists(): continue
        imgs  = [p for p in sorted(folder.iterdir()) if p.suffix in EXTS][:n_per_class]
        accum = np.zeros((prnu_size, prnu_size), dtype=np.float64)
        cnt   = 0
        for p in imgs:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (prnu_size,prnu_size)).astype(np.float32)/255
            accum += noise_residual(img)
            cnt   += 1
        if cnt > 0:
            fp = accum / cnt
            # high-pass to remove scene residuals
            fp = fp - cv2.GaussianBlur(fp.astype(np.float32),(0,0),3)
            prnu_dict[phone] = fp.astype(np.float32)

    np.savez(str(cache), **prnu_dict)
    return prnu_dict


def prnu_analysis(img_path: Path, prnu_gallery: dict,
                   size: int = 256) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return {}
    img   = cv2.resize(img, (size,size)).astype(np.float32)/255
    query_noise = noise_residual(img)
    # High-pass the query noise too
    query_noise -= cv2.GaussianBlur(query_noise.astype(np.float32),(0,0),3)

    results = {}
    for phone, fp in prnu_gallery.items():
        q  = query_noise.ravel().astype(np.float64)
        p  = fp.ravel().astype(np.float64)
        q -= q.mean(); p -= p.mean()
        denom = np.linalg.norm(q)*np.linalg.norm(p)
        ncc   = float(np.dot(q,p)/denom) if denom>1e-10 else 0.0

        # Peak-to-Correlation Energy (PCE)
        corr_map = np.real(np.fft.ifft2(
            np.fft.fft2(query_noise) * np.conj(np.fft.fft2(fp.reshape(size,size)))
        ))
        peak  = float(corr_map.max())
        pce   = float(peak**2 / (np.sum(corr_map**2)/(size*size)+1e-9))

        results[phone] = {
            "ncc":  round(ncc,  8),
            "pce":  round(pce,  4),
        }

    return results


# =============================================================================
# SECTION 5 — FREQUENCY DOMAIN ANALYSIS
# =============================================================================
def frequency_analysis(img_path: Path, size: int = 256) -> dict:
    img  = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return {}
    img  = cv2.resize(img, (size,size)).astype(np.float32)/255
    nr   = noise_residual(img)

    mag  = np.abs(fftshift(fft2(nr.astype(np.float64))))
    h, w = mag.shape
    cy, cx = h//2, w//2
    y, x   = np.mgrid[-cy:h-cy, -cx:w-cx]
    radius = np.sqrt(x**2+y**2).astype(int)
    max_r  = min(cy,cx)

    # Radial power spectrum
    edges  = np.linspace(0, max_r, 33)
    radial = np.array([mag[(radius>=int(edges[i]))&(radius<int(edges[i+1]))].mean()
                        if ((radius>=int(edges[i]))&(radius<int(edges[i+1]))).any()
                        else 0.0 for i in range(32)])

    log_mag = np.log1p(mag)

    # Spectral entropy
    pm = mag.ravel()+1e-9; pm /= pm.sum()
    sp_entropy = float(scipy_entropy(pm))

    # Dominant frequencies (top-5 by magnitude, excluding DC)
    mag_copy = mag.copy(); mag_copy[cy,cx] = 0
    flat     = mag_copy.ravel()
    top5_idx = np.argsort(flat)[-5:][::-1]
    dom_freq  = []
    for idx in top5_idx:
        iy,ix = np.unravel_index(idx, mag.shape)
        fy = (iy-cy)/h; fx = (ix-cx)/w
        dom_freq.append({
            "freq_y": round(float(fy),4), "freq_x": round(float(fx),4),
            "magnitude": round(float(flat[idx]),4)
        })

    return {
        "radial_psd":       radial.tolist(),
        "spectral_entropy": round(sp_entropy,  6),
        "log_mean":         round(float(log_mag.mean()), 6),
        "log_std":          round(float(log_mag.std()),  6),
        "dominant_freqs":   dom_freq,
        "noise_magnitude":  mag.tolist(),  # full 2D (for heatmap)
    }


# =============================================================================
# SECTION 6 — COLOR SCIENCE ANALYSIS
# =============================================================================
def color_science_analysis(img_path: Path) -> dict:
    img_bgr  = cv2.imread(str(img_path))
    if img_bgr is None: return {}
    img_bgr  = cv2.resize(img_bgr, (256,256))
    rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycrcb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    r,g,b    = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    result   = {
        "channel_means":  {"R":round(float(r.mean()),4),
                            "G":round(float(g.mean()),4),
                            "B":round(float(b.mean()),4)},
        "channel_stds":   {"R":round(float(r.std()),4),
                            "G":round(float(g.std()),4),
                            "B":round(float(b.std()),4)},
        "wb_ratios":      {"R/G":round(float(r.mean()/(g.mean()+1e-8)),4),
                            "B/G":round(float(b.mean()/(g.mean()+1e-8)),4),
                            "R/B":round(float(r.mean()/(b.mean()+1e-8)),4)},
        "hsv":            {"H_mean":round(float(hsv[:,:,0].mean()),4),
                            "S_mean":round(float(hsv[:,:,1].mean()),4),
                            "V_mean":round(float(hsv[:,:,2].mean()),4),
                            "S_std": round(float(hsv[:,:,1].std()),4)},
        "lab":            {"L_mean":round(float(lab[:,:,0].mean()),4),
                            "A_mean":round(float(lab[:,:,1].mean()),4),
                            "B_mean":round(float(lab[:,:,2].mean()),4)},
        "chroma_noise":   {"Cr_std":round(float(ycrcb[:,:,1].std()),4),
                            "Cb_std":round(float(ycrcb[:,:,2].std()),4)},
        # Full histograms for plotting
        "hist_R": np.histogram(r.ravel(),64,[0,1])[0].tolist(),
        "hist_G": np.histogram(g.ravel(),64,[0,1])[0].tolist(),
        "hist_B": np.histogram(b.ravel(),64,[0,1])[0].tolist(),
        "hist_H": np.histogram(hsv[:,:,0].ravel(),64,[0,180])[0].tolist(),
        "hist_S": np.histogram(hsv[:,:,1].ravel(),64,[0,255])[0].tolist(),
    }
    return result


# =============================================================================
# MASTER ANALYSIS RUNNER
# =============================================================================
def run_full_analysis(img_path: Path, model, gallery: dict,
                       gallery_vecs: dict, prnu_gallery: dict,
                       data_dir: str, threshold: float = 0.50) -> dict:
    print(f"\n  Running full analysis: {img_path.name}")
    img_path = Path(img_path)

    # Get query embedding
    with torch.no_grad():
        t   = _val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        emb = model(t).cpu().numpy()[0]

    # Get query feature vector
    query_vec = extract(str(img_path))

    report = {
        "image":    image_properties(img_path),
        "embedding":embedding_analysis(emb, gallery),
        "features": feature_vector_comparison(query_vec, gallery_vecs) if query_vec is not None else {},
        "prnu":     prnu_analysis(img_path, prnu_gallery),
        "frequency":frequency_analysis(img_path),
        "color":    color_science_analysis(img_path),
        "_query_emb":   emb.tolist(),
        "_query_vec":   query_vec.tolist() if query_vec is not None else [],
        "_gallery_vecs":{k:v.tolist() for k,v in gallery_vecs.items()},
        "_threshold":   threshold,
    }

    top  = report["embedding"]["top_match"]
    sim  = report["embedding"]["top_sim"]
    margin = report["embedding"]["margin_gap"]
    z    = report["embedding"]["z_score"]

    # Decision flags
    flags = []
    if sim < threshold:
        flags.append("LOW_SIMILARITY: top match below threshold → Unknown")
    if margin < 0.05:
        flags.append("AMBIGUOUS: margin gap < 0.05 → low confidence")
    if z < 1.5:
        flags.append("WEAK_ZSCORE: z-score < 1.5 → similarity not distinctive")

    # Check if PRNU agrees
    if report["prnu"]:
        prnu_top = max(report["prnu"], key=lambda k: report["prnu"][k]["ncc"])
        if prnu_top != top and report["prnu"][prnu_top]["ncc"] > 0.015:
            flags.append(f"PRNU_CONFLICT: embedding says '{top}' but PRNU NCC favours '{prnu_top}'")

    report["flags"]    = flags
    report["decision"] = {
        "prediction":  top if sim >= threshold else "Unknown",
        "is_match":    sim >= threshold,
        "confidence":  report["embedding"]["softmax_conf"].get(top, 0),
        "margin_gap":  margin,
        "z_score":     z,
        "flags":       flags,
    }

    return report


# =============================================================================
# VISUALISATION — 12-panel forensic report
# =============================================================================
def save_analysis_figure(report: dict, img_path: Path) -> Path:
    img_path = Path(img_path)

    # ── Load image data ───────────────────────────────────────────────────
    pil_img  = Image.open(img_path).convert("RGB")
    img_np   = np.array(pil_img.resize((256,256)))/255.0
    img_bgr  = cv2.imread(str(img_path))
    img_bgr  = cv2.resize(img_bgr,(256,256)) if img_bgr is not None \
               else np.zeros((256,256,3),np.uint8)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    noise    = noise_residual(gray)
    noise_d  = (noise-noise.min())/(noise.max()-noise.min()+1e-8)

    emb_data  = report["embedding"]
    feat_data = report["features"]
    prnu_data = report["prnu"]
    freq_data = report["frequency"]
    color_data= report["color"]
    decision  = report["decision"]
    img_props = report["image"]

    ranked    = emb_data["ranked"]
    phones    = [r[0] for r in ranked]
    sims      = [r[1] for r in ranked]
    top_phone = emb_data["top_match"]
    top_sim   = emb_data["top_sim"]
    is_match  = decision["is_match"]
    threshold = report["_threshold"]

    # ── Figure setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 20), facecolor=C_BG)
    plt.rcParams.update({
        "text.color":       C_TEXT,
        "axes.labelcolor":  C_TEXT,
        "axes.edgecolor":   C_GRID,
        "xtick.color":      C_TEXT,
        "ytick.color":      C_TEXT,
        "axes.facecolor":   C_PANEL,
        "figure.facecolor": C_BG,
        "grid.color":       C_GRID,
        "font.family":      "monospace",
    })

    gs_main = gridspec.GridSpec(4, 6, figure=fig,
                                 hspace=0.52, wspace=0.38,
                                 left=0.04, right=0.97,
                                 top=0.93,  bottom=0.04)

    # ── Title banner ─────────────────────────────────────────────────────
    match_color = C_MATCH if is_match else C_NOMATCH
    pred_txt    = (f"✔  {top_phone}" if is_match else "✘  UNKNOWN / NEW PHONE")
    fig.text(0.5, 0.966,
             f"CAMERA FINGERPRINT FORENSIC ANALYSIS  ·  {img_path.name}",
             ha="center", va="center", fontsize=13, color=C_NEUTRAL,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.952,
             f"PREDICTION:  {pred_txt}   |   Cosine Sim: {top_sim:.4f}   |   "
             f"Margin: {emb_data['margin_gap']:.4f}   |   Z-Score: {emb_data['z_score']:.2f}",
             ha="center", va="center", fontsize=11,
             color=match_color, fontfamily="monospace", fontweight="bold")

    # ────────────────────────────────────────────────────────────────────
    # PANEL 0 — Input image
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 0])
    ax.imshow(img_np)
    ax.set_title("INPUT IMAGE", fontsize=8, color=C_NEUTRAL, pad=4)
    ax.axis("off")
    info_txt = (f"{img_props['width']}×{img_props['height']}px  "
                f"{img_props['file_size_kb']}KB\n"
                f"Lum:{img_props.get('mean_luminance','?'):.1f}  "
                f"DR:{img_props.get('dynamic_range','?'):.0f}")
    ax.set_xlabel(info_txt, fontsize=6.5, color=C_NEUTRAL, labelpad=2)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 1 — PRNU noise residual
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 1])
    ax.imshow(noise_d, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("PRNU NOISE RESIDUAL", fontsize=8, color=C_NEUTRAL, pad=4)
    ax.axis("off")
    ax.set_xlabel(f"std={noise.std():.5f}  kurt={float(np.array([noise.ravel()]).std()):.2f}",
                  fontsize=6.5, color=C_NEUTRAL, labelpad=2)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 2 — Cosine similarity bar chart
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 2:4])
    short_ph = [p.replace("_","\n") for p in phones]
    colors_b = [C_MATCH if p==top_phone and is_match else
                C_ACCENT if p==top_phone else C_NEUTRAL for p in phones]
    bars = ax.barh(range(len(phones)), sims, color=colors_b,
                   edgecolor=C_GRID, linewidth=0.5)
    ax.axvline(threshold, color=C_NOMATCH, ls="--", lw=1.2,
               label=f"threshold={threshold}")
    ax.set_yticks(range(len(phones)))
    ax.set_yticklabels(short_ph, fontsize=7)
    ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel("Cosine Similarity", fontsize=8)
    ax.set_title("EMBEDDING SIMILARITY  (cosine)", fontsize=8, color=C_NEUTRAL, pad=4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7, color=C_TEXT)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 3 — Multi-metric distance table
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[0, 4:6])
    ax.axis("off")
    table_data = [["Phone","Cosine","Euclid","L1","Pearson"]]
    for ph, sim_v in ranked:
        ed = emb_data["per_phone"][ph]
        table_data.append([
            ph.replace("_","\n"),
            f"{ed['cosine_similarity']:.4f}",
            f"{ed['euclidean_distance']:.4f}",
            f"{ed['l1_distance']:.4f}",
            f"{ed['pearson_correlation']:.4f}",
        ])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.30,0.17,0.17,0.17,0.17])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_facecolor(C_PANEL if r%2==0 else "#1e2530")
        cell.set_edgecolor(C_GRID)
        cell.set_text_props(color=C_TEXT)
        if r==0: cell.set_facecolor("#1e3a5f")
    ax.set_title("EMBEDDING DISTANCES", fontsize=8, color=C_NEUTRAL, pad=4)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 4 — Feature component cosine similarity (radar-like bar)
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[1, 0:2])
    if feat_data:
        comps    = ["freq_features","dct_features","color_features","noise_stats"]
        comp_labels = ["Freq\n(FFT)","DCT\n(JPEG)","Color\n(ISP)","Noise\n(PRNU)"]
        x        = np.arange(len(comps))
        w        = 0.8 / max(len(phones), 1)
        cmap_ph  = plt.cm.tab10(np.linspace(0,0.9,len(phones)))
        for i, ph in enumerate(phones):
            vals = [feat_data.get(ph,{}).get("component_cosine",{}).get(c,0)
                    for c in comps]
            ax.bar(x + i*w - (len(phones)-1)*w/2, vals, w*0.9,
                   label=ph.replace("_"," "), color=cmap_ph[i], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(comp_labels, fontsize=7)
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0.7, color=C_MATCH,   ls=":", lw=1, alpha=0.6)
        ax.axhline(0.5, color=C_NOMATCH, ls=":", lw=1, alpha=0.6)
        ax.set_ylabel("Cosine Similarity", fontsize=7)
        ax.set_title("FEATURE COMPONENT SIMILARITY", fontsize=8, color=C_NEUTRAL, pad=4)
        ax.legend(fontsize=5.5, loc="lower right", ncol=2)
        ax.grid(axis="y", alpha=0.3)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 5 — Feature vector absolute difference heatmap
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[1, 2:4])
    if feat_data and report["_query_vec"]:
        qv       = np.array(report["_query_vec"])
        diff_mat = np.zeros((len(phones), len(qv)))
        for i, ph in enumerate(phones):
            if ph in feat_data:
                diff_mat[i] = feat_data[ph].get("abs_diff_vector", np.zeros(len(qv)))
        im = ax.imshow(diff_mat, aspect="auto", cmap="hot",
                       interpolation="nearest", vmin=0)
        ax.set_yticks(range(len(phones)))
        ax.set_yticklabels([p.replace("_"," ") for p in phones], fontsize=7)
        ax.set_xlabel("Feature dimension (0-95)", fontsize=7)
        ax.set_title("|QUERY − GALLERY| FEATURE DIFF HEATMAP", fontsize=8,
                     color=C_NEUTRAL, pad=4)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
        # Mark component boundaries
        for x_line, lbl in [(36,"DCT"),(52,"Color"),(90,"Noise")]:
            ax.axvline(x_line-0.5, color=C_ACCENT, lw=1, alpha=0.7)
            ax.text(x_line, -0.7, lbl, fontsize=5.5, color=C_ACCENT, ha="center")

    # ────────────────────────────────────────────────────────────────────
    # PANEL 6 — Chi-squared + Bhattacharyya distances
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[1, 4:6])
    if feat_data:
        chi2_v  = [feat_data.get(ph,{}).get("chi2_freq",0)      for ph in phones]
        bhat_v  = [feat_data.get(ph,{}).get("bhattacharyya_freq",0) for ph in phones]
        x       = np.arange(len(phones))
        w       = 0.38
        b1      = ax.bar(x-w/2, chi2_v,  w, label="Chi² (Freq)", color="#5580ff", alpha=0.85)
        b2      = ax.bar(x+w/2, bhat_v, w, label="Bhattacharyya", color="#ff7055", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_","\n") for p in phones], fontsize=7)
        ax.set_ylabel("Distance (lower = more similar)", fontsize=7)
        ax.set_title("HISTOGRAM DISTANCES  (Freq component)", fontsize=8,
                     color=C_NEUTRAL, pad=4)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=6, color=C_TEXT)
        ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=6, color=C_TEXT)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 7 — PRNU NCC bar
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[2, 0:2])
    if prnu_data:
        ncc_vals = [prnu_data.get(ph,{}).get("ncc",0) for ph in phones]
        pce_vals = [prnu_data.get(ph,{}).get("pce",0) for ph in phones]
        prnu_top = max(prnu_data, key=lambda k: prnu_data[k]["ncc"])
        cols_p   = [C_MATCH if p==prnu_top else C_NEUTRAL for p in phones]
        bars = ax.barh(range(len(phones)), ncc_vals, color=cols_p,
                       edgecolor=C_GRID, linewidth=0.5)
        ax.set_yticks(range(len(phones)))
        ax.set_yticklabels([p.replace("_"," ") for p in phones], fontsize=7)
        ax.set_xlabel("NCC (Normalized Cross-Correlation)", fontsize=7)
        ax.set_title("PRNU SENSOR FINGERPRINT  (NCC)", fontsize=8,
                     color=C_NEUTRAL, pad=4)
        ax.bar_label(bars, fmt="%.6f", padding=3, fontsize=6.5, color=C_TEXT)
        ax.axvline(0.01, color=C_ACCENT, ls="--", lw=1,
                   label="Typical match threshold ≈0.01")
        ax.legend(fontsize=6.5)
        ax.grid(axis="x", alpha=0.3)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 8 — Radial Power Spectral Density
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[2, 2:4])
    if freq_data.get("radial_psd"):
        psd    = np.array(freq_data["radial_psd"])
        freqs  = np.arange(len(psd))
        ax.fill_between(freqs, psd, alpha=0.4, color=C_ACCENT)
        ax.plot(freqs, psd, color=C_ACCENT, lw=1.5)
        ax.set_xlabel("Spatial Frequency Bin", fontsize=7)
        ax.set_ylabel("Mean Magnitude", fontsize=7)
        ax.set_title("RADIAL POWER SPECTRAL DENSITY  (noise residual)", fontsize=8,
                     color=C_NEUTRAL, pad=4)
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.95,
                f"Spectral entropy: {freq_data['spectral_entropy']:.4f}",
                transform=ax.transAxes, fontsize=7, color=C_NEUTRAL,
                ha="right", va="top")

    # ────────────────────────────────────────────────────────────────────
    # PANEL 9 — FFT magnitude heatmap
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[2, 4:6])
    if freq_data.get("noise_magnitude"):
        mag_arr = np.array(freq_data["noise_magnitude"])
        ax.imshow(np.log1p(mag_arr), cmap="inferno", aspect="auto")
        ax.set_title("FFT MAGNITUDE SPECTRUM  (noise residual, log scale)",
                     fontsize=8, color=C_NEUTRAL, pad=4)
        ax.set_xlabel("Frequency X", fontsize=7)
        ax.set_ylabel("Frequency Y", fontsize=7)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 10 — Color histograms (R, G, B)
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[3, 0:2])
    if color_data:
        bins = np.linspace(0, 1, 65)[:-1]
        for ch, col, key in [("R","#ff5555","hist_R"),
                               ("G","#55ff55","hist_G"),
                               ("B","#5599ff","hist_B")]:
            h = np.array(color_data[key], dtype=float)
            h /= h.sum()+1e-9
            ax.plot(bins, h, color=col, lw=1.3, label=ch, alpha=0.85)
            ax.fill_between(bins, h, alpha=0.15, color=col)
        ax.set_xlabel("Pixel Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_title("RGB CHANNEL HISTOGRAMS", fontsize=8, color=C_NEUTRAL, pad=4)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        stats_txt = (f"R:{color_data['channel_means']['R']:.3f}±{color_data['channel_stds']['R']:.3f}  "
                     f"G:{color_data['channel_means']['G']:.3f}±{color_data['channel_stds']['G']:.3f}  "
                     f"B:{color_data['channel_means']['B']:.3f}±{color_data['channel_stds']['B']:.3f}")
        ax.set_xlabel(stats_txt, fontsize=6.5, labelpad=3)

    # ────────────────────────────────────────────────────────────────────
    # PANEL 11 — White balance ratios radar / bar
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[3, 2:4])
    if color_data:
        wb     = color_data["wb_ratios"]
        labels = list(wb.keys()); vals = list(wb.values())
        x      = np.arange(len(labels))
        b      = ax.bar(x, vals, color=[C_MATCH, C_ACCENT, "#a78bfa"],
                        alpha=0.85, edgecolor=C_GRID, linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(1.0, color=C_NEUTRAL, ls="--", lw=1, alpha=0.7, label="R=G=B")
        ax.set_ylabel("Ratio", fontsize=7)
        ax.set_title("WHITE BALANCE RATIOS  (ISP fingerprint)", fontsize=8,
                     color=C_NEUTRAL, pad=4)
        ax.bar_label(b, fmt="%.4f", padding=3, fontsize=8, color=C_TEXT)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        ax.text(0.98, 0.95,
                f"Cr_std:{color_data['chroma_noise']['Cr_std']:.4f}  "
                f"Cb_std:{color_data['chroma_noise']['Cb_std']:.4f}",
                transform=ax.transAxes, fontsize=6.5, color=C_NEUTRAL,
                ha="right", va="top")

    # ────────────────────────────────────────────────────────────────────
    # PANEL 12 — Softmax confidence + decision summary
    # ────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[3, 4:6])
    ax.axis("off")

    conf_map  = emb_data["softmax_conf"]
    flags     = report["flags"]
    lines     = [
        "─── DECISION SUMMARY ───────────────────",
        f"Prediction    : {decision['prediction']}",
        f"Is Match      : {decision['is_match']}",
        f"Cosine Sim    : {top_sim:.6f}",
        f"Margin Gap    : {emb_data['margin_gap']:.6f}",
        f"Z-Score       : {emb_data['z_score']:.4f}",
        f"Softmax Conf  : {conf_map.get(top_phone,0)*100:.1f}%",
        "─── SOFTMAX CONFIDENCE ─────────────────",
    ]
    for ph, conf in sorted(conf_map.items(), key=lambda x: -x[1]):
        lines.append(f"  {ph:<20}  {conf*100:>5.1f}%")
    if flags:
        lines.append("─── FLAGS ───────────────────────────────")
        for fl in flags:
            lines.append(f"  ⚠ {fl}")
    else:
        lines.append("─── FLAGS ───────────────────────────────")
        lines.append("  ✔  No anomalies detected")

    y_pos = 0.98
    for line in lines:
        col  = C_NOMATCH if "⚠" in line else \
               C_MATCH   if "✔" in line else \
               C_ACCENT  if "───" in line else C_TEXT
        sz   = 6.5 if not "───" in line else 6.0
        ax.text(0.02, y_pos, line, transform=ax.transAxes,
                fontsize=sz, color=col, fontfamily="monospace",
                va="top")
        y_pos -= 0.065

    # ── Save ─────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / f"{img_path.stem}_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    plt.rcdefaults()
    print(f"  ✓ Figure: {out}")
    return out


# =============================================================================
# CONSOLE REPORT
# =============================================================================
def print_console_report(report: dict):
    em  = report["embedding"]
    dec = report["decision"]
    ip  = report["image"]
    freq= report["frequency"]
    col = report["color"]
    pr  = report["prnu"]

    sep  = "─"*62
    sep2 = "═"*62

    print(f"\n{sep2}")
    print(f"  FORENSIC ANALYSIS REPORT  —  {ip['filename']}")
    print(f"{sep2}")

    print(f"\n  IMAGE PROPERTIES")
    print(f"  {sep}")
    print(f"  Resolution   : {ip['width']} × {ip['height']} px  ({ip['megapixels']} MP)")
    print(f"  File Size    : {ip['file_size_kb']} KB")
    print(f"  JPEG Quality : {ip['jpeg_quality_est']}")
    print(f"  Luminance    : mean={ip.get('mean_luminance','?')}  std={ip.get('std_luminance','?')}")
    print(f"  Dynamic Range: {ip.get('dynamic_range','?')}")
    if ip.get("exif"):
        print(f"  EXIF         : {', '.join(f'{k}={v}' for k,v in ip['exif'].items())}")

    print(f"\n  EMBEDDING ANALYSIS  (128-dim L2-normalized)")
    print(f"  {sep}")
    print(f"  {'Phone':<22} {'Cosine':>8} {'Euclid':>8} {'L1':>8} {'Pearson':>9}")
    print(f"  {sep}")
    for ph, sim in em["ranked"]:
        ed   = em["per_phone"][ph]
        mark = " ←" if ph == em["top_match"] else "  "
        print(f"  {ph:<22} {ed['cosine_similarity']:>8.4f} "
              f"{ed['euclidean_distance']:>8.4f} "
              f"{ed['l1_distance']:>8.4f} "
              f"{ed['pearson_correlation']:>9.4f}{mark}")
    print(f"\n  Margin Gap   : {em['margin_gap']:.6f}"
          f"  (difference between top and runner-up)")
    print(f"  Z-Score      : {em['z_score']:.4f}"
          f"  (how many σ above mean similarity)")
    print(f"  Softmax Conf : "
          + "  ".join(f"{p}={v*100:.1f}%" for p,v in
                      sorted(em["softmax_conf"].items(), key=lambda x:-x[1])))

    if pr:
        print(f"\n  PRNU ANALYSIS  (sensor noise fingerprint)")
        print(f"  {sep}")
        print(f"  {'Phone':<22} {'NCC':>12} {'PCE':>10}")
        print(f"  {sep}")
        for ph in [r[0] for r in em["ranked"]]:
            if ph in pr:
                d = pr[ph]
                print(f"  {ph:<22} {d['ncc']:>12.8f} {d['pce']:>10.2f}")
        print(f"  (NCC > 0.01 typically indicates same camera source)")

    if freq:
        print(f"\n  FREQUENCY DOMAIN")
        print(f"  {sep}")
        print(f"  Spectral Entropy : {freq['spectral_entropy']:.6f}")
        print(f"  Log Mean / Std   : {freq['log_mean']:.4f} / {freq['log_std']:.4f}")
        print(f"  Top-3 Dominant Frequencies:")
        for df in freq.get("dominant_freqs",[])[:3]:
            print(f"    fy={df['freq_y']:+.4f}  fx={df['freq_x']:+.4f}  "
                  f"mag={df['magnitude']:.2f}")

    if col:
        print(f"\n  COLOR SCIENCE")
        print(f"  {sep}")
        cm = col["channel_means"]; cs = col["channel_stds"]
        wb = col["wb_ratios"]
        print(f"  Channel Means : R={cm['R']:.4f}  G={cm['G']:.4f}  B={cm['B']:.4f}")
        print(f"  Channel Stds  : R={cs['R']:.4f}  G={cs['G']:.4f}  B={cs['B']:.4f}")
        print(f"  WB Ratios     : R/G={wb['R/G']:.4f}  B/G={wb['B/G']:.4f}  R/B={wb['R/B']:.4f}")
        print(f"  Chroma Noise  : Cr_std={col['chroma_noise']['Cr_std']:.4f}  "
              f"Cb_std={col['chroma_noise']['Cb_std']:.4f}")
        hsv = col["hsv"]
        print(f"  HSV           : H={hsv['H_mean']:.2f}°  S={hsv['S_mean']:.2f}  "
              f"V={hsv['V_mean']:.2f}  S_std={hsv['S_std']:.2f}")

    print(f"\n  DECISION")
    print(f"  {sep}")
    print(f"  Prediction  : {dec['prediction']}")
    print(f"  Is Match    : {dec['is_match']}")
    print(f"  Confidence  : {dec['confidence']*100:.1f}%  (softmax)")
    print(f"  Margin Gap  : {dec['margin_gap']:.6f}")
    print(f"  Z-Score     : {dec['z_score']:.4f}")
    if dec["flags"]:
        print(f"  ⚠  FLAGS:")
        for fl in dec["flags"]: print(f"       {fl}")
    else:
        print(f"  ✔  No anomaly flags")
    print(f"\n{sep2}\n")


# =============================================================================
# MAIN
# =============================================================================
def analyse(img_path: Path, model, gallery: dict,
             gallery_vecs: dict, prnu_gallery: dict,
             data_dir: str, threshold: float,
             save_json: bool) -> dict:

    report = run_full_analysis(img_path, model, gallery,
                                gallery_vecs, prnu_gallery,
                                data_dir, threshold)
    print_console_report(report)
    fig_path = save_analysis_figure(report, img_path)

    if save_json:
        # strip large internal arrays from JSON export
        export = {k:v for k,v in report.items()
                  if not k.startswith("_") and k != "frequency" or k == "frequency"}
        export.pop("_query_emb",  None)
        export.pop("_query_vec",  None)
        export.pop("_gallery_vecs", None)
        if "frequency" in export:
            export["frequency"] = {k:v for k,v in export["frequency"].items()
                                   if k != "noise_magnitude"}
        jp = OUTPUT_DIR / f"{img_path.stem}_analysis.json"
        with open(jp,"w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"  ✓ JSON : {jp}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full quantitative camera fingerprint analysis")
    parser.add_argument("--image",     help="Single image path")
    parser.add_argument("--folder",    help="Folder of images")
    parser.add_argument("--data",      default="./data")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help(); sys.exit(0)

    print("Loading model + gallery...")
    model   = load_model()
    gallery = load_gallery()
    if not gallery:
        print("[ERROR] Gallery empty! Run: python 3_gallery.py --register_all")
        sys.exit(1)

    print("Building/loading gallery feature vectors...")
    gallery_vecs = build_gallery_feature_vectors(args.data)

    print("Building/loading PRNU gallery...")
    prnu_gallery = build_prnu_gallery(args.data)

    EXTS = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

    if args.image:
        analyse(Path(args.image), model, gallery, gallery_vecs,
                prnu_gallery, args.data, args.threshold, args.save_json)

    elif args.folder:
        imgs = [p for p in sorted(Path(args.folder).iterdir())
                if p.suffix in EXTS]
        print(f"\nAnalysing {len(imgs)} images in {args.folder}")
        for img in imgs:
            analyse(img, model, gallery, gallery_vecs,
                    prnu_gallery, args.data, args.threshold, args.save_json)

    print("✅ Analysis complete.  Results in ./cv_outputs/analysis/")