"""
=============================================================================
PHONE CAMERA FINGERPRINT SYSTEM  —  RTX 3050 Optimized
FILE 3: gallery.py
=============================================================================
Register any phone by embedding its photos.
No retraining needed for new phones.

Commands:
  python 3_gallery.py --register_all          (register all training phones)
  python 3_gallery.py --register --name "Samsung_S24" --folder ./photos/
  python 3_gallery.py --list
  python 3_gallery.py --visualize
  python 3_gallery.py --remove --name "OldPhone"
=============================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import json, sys, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from train_2 import CameraNet, PhoneDataset, EMB_DIM, IMG_SIZE, NUM_WORKERS
import torchvision.transforms.functional as TF

MODEL_PATH  = Path("./cv_outputs/models/best_fingerprint.pth")
GALLERY_DIR = Path("./cv_outputs/gallery")
OUTPUT_DIR  = Path("./cv_outputs")
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS   = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG",".bmp",".tiff"}


# ── Load model ────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Train first: python 2_train.py"
        )
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CameraNet(emb_dim=EMB_DIM).to(DEVICE)
    key   = "ema_state" if "ema_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[key])
    model.eval()
    r1 = ckpt.get("metrics",{}).get("R1",0)*100
    print(f"  Model loaded [{key}]  val R1={r1:.1f}%")
    return model


# ── Image → embedding ─────────────────────────────────────────────────────
def _val_transform(img: Image.Image) -> torch.Tensor:
    img = TF.center_crop(img, IMG_SIZE) if min(img.size) >= IMG_SIZE \
          else TF.resize(img, [IMG_SIZE, IMG_SIZE])
    t   = TF.to_tensor(img)
    return TF.normalize(t, [0.485,0.456,0.406], [0.229,0.224,0.225])


@torch.no_grad()
def embed_folder(model, folder: Path, n_max=80):
    imgs = [p for p in sorted(folder.iterdir()) if p.suffix in EXTS]
    if not imgs: raise ValueError(f"No images in {folder}")
    if len(imgs) > n_max:
        idx  = np.linspace(0, len(imgs)-1, n_max, dtype=int)
        imgs = [imgs[i] for i in idx]

    embs, fail = [], 0
    for p in imgs:
        try:
            t   = _val_transform(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            emb = model(t).cpu().numpy()[0]
            embs.append(emb)
        except Exception: fail += 1
    if not embs: raise ValueError("All images failed")
    if fail: print(f"    [warn] {fail} failed")

    mean = np.array(embs).mean(0)
    return mean / (np.linalg.norm(mean)+1e-8), len(embs)


# ── Gallery CRUD ──────────────────────────────────────────────────────────
def load_gallery():
    p = GALLERY_DIR/"gallery.json"
    if not p.exists(): return {}
    with open(p) as f: data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k,v in data.items()}


def save_gallery(g):
    with open(GALLERY_DIR/"gallery.json","w") as f:
        json.dump({k: v.tolist() for k,v in g.items()}, f, indent=2)
    print(f"  Gallery saved ({len(g)} phones)")


def register_phone(model, name, folder, gallery=None):
    if gallery is None: gallery = load_gallery()
    print(f"  Registering '{name}' ...", end=" ", flush=True)
    emb, n = embed_folder(model, Path(folder))
    gallery[name] = emb
    save_gallery(gallery)
    print(f"{n} images  norm={np.linalg.norm(emb):.4f}")
    return gallery


def register_all(model, data_dir):
    gallery = load_gallery()
    folders = sorted([d for d in Path(data_dir).iterdir()
                      if d.is_dir() and not d.name.startswith(".")])
    print(f"Registering {len(folders)} phones from {data_dir}")
    for folder in folders:
        try:    gallery = register_phone(model, folder.name, folder, gallery)
        except Exception as e: print(f"  [skip] {folder.name}: {e}")
    return gallery


def list_gallery():
    g = load_gallery()
    if not g: print("  Gallery empty. Run --register_all first."); return
    print(f"\n  📱 Gallery ({len(g)} phones)")
    print(f"  {'#':<4}{'Name':<30}{'Norm':>8}")
    print("  "+"─"*44)
    for i,(name,emb) in enumerate(sorted(g.items()),1):
        print(f"  {i:<4}{name:<30}{np.linalg.norm(emb):>8.4f}")


def remove_phone(name):
    g = load_gallery()
    if name in g: del g[name]; save_gallery(g); print(f"  Removed '{name}'")
    else:
        print(f"  Not found: '{name}'")
        m = [k for k in g if name.lower() in k.lower()]
        if m: print(f"  Similar: {m}")


# ── Visualize ─────────────────────────────────────────────────────────────
def visualize():
    g = load_gallery()
    if len(g) < 2: print("  Need ≥2 phones."); return

    names   = sorted(g.keys())
    embs    = np.array([g[n] for n in names])
    sim_mat = np.dot(embs, embs.T)
    short   = [n.replace("_","\n") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14+max(0,len(names)-6), 6))
    sns.heatmap(sim_mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=short, yticklabels=short,
                ax=axes[0], vmin=-0.5, vmax=1.0, center=0,
                linewidths=0.5, annot_kws={"size":9})
    axes[0].set_title("Camera Similarity Matrix")

    if len(names) >= 3:
        pca  = PCA(n_components=2)
        pts  = pca.fit_transform(embs)
        cols = plt.cm.tab20(np.linspace(0,1,len(names)))
        axes[1].scatter(pts[:,0], pts[:,1], s=300, c=cols, zorder=5)
        for i,(nm,pt) in enumerate(zip(names,pts)):
            axes[1].annotate(nm.replace("_","\n"),(pt[0],pt[1]),
                             xytext=(8,4), textcoords="offset points",
                             fontsize=8, color=cols[i], fontweight="bold")
        axes[1].set_title(f"Gallery PCA ({pca.explained_variance_ratio_.sum()*100:.0f}% var)")
        axes[1].grid(alpha=.2)

    plt.suptitle("Phone Gallery — Embedding Space", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR/"gallery_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {out}")


# ── identify (called by predict.py) ──────────────────────────────────────
@torch.no_grad()
def identify(model, img_path, gallery, top_k=6):
    t   = _val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    emb = model(t).cpu().numpy()[0]
    scores = {n: float(np.dot(emb, proto)) for n,proto in gallery.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], emb


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register_all", action="store_true")
    parser.add_argument("--register",     action="store_true")
    parser.add_argument("--remove",       action="store_true")
    parser.add_argument("--list",         action="store_true")
    parser.add_argument("--visualize",    action="store_true")
    parser.add_argument("--data",   default="./data")
    parser.add_argument("--name",   default="")
    parser.add_argument("--folder", default="")
    args = parser.parse_args()

    if args.list:
        list_gallery()
    elif args.visualize:
        visualize()
    elif args.remove:
        remove_phone(args.name)
    elif args.register_all:
        m = load_model(); register_all(m, args.data); visualize()
    elif args.register:
        if not args.name or not args.folder:
            print("Need --name and --folder")
        else:
            m = load_model(); register_phone(m, args.name, args.folder)
    else:
        parser.print_help()