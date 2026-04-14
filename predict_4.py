"""
=============================================================================
PHONE CAMERA FINGERPRINT SYSTEM  —  RTX 3050 Optimized
FILE 4: predict.py
=============================================================================
Identify which phone took any image.

Usage:
  python 4_predict.py --image photo.jpg
  python 4_predict.py --folder ./test_photos/
  python 4_predict.py --eval --data ./data
  python 4_predict.py --image photo.jpg --threshold 0.50
=============================================================================
"""

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2, json, sys, argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_2   import CameraNet, PhoneDataset, EMB_DIM, IMG_SIZE, PHONE_CLASSES
from gallery_3 import load_model, load_gallery, identify, _val_transform

OUTPUT_DIR = Path("./cv_outputs/predictions")
EVAL_DIR   = Path("./cv_outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = np.array([0.485, 0.456, 0.406])
STD    = np.array([0.229, 0.224, 0.225])


# ── CV metrics for display ────────────────────────────────────────────────
def cv_metrics(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return {}
    img  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    noise = cv2.absdiff(gray, blur)
    lap  = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny((gray*255).astype(np.uint8), 50, 150)
    return {
        "Sharpness":    f"{lap.var():.1f}",
        "Noise std":    f"{noise.std():.4f}",
        "Edge density": f"{(edges>0).mean()*100:.2f}%",
        "Saturation":   f"{hsv[:,:,1].mean():.1f}",
        "Brightness":   f"{hsv[:,:,2].mean():.1f}",
    }


# ── Activation map ────────────────────────────────────────────────────────
@torch.no_grad()
def activation_map(model, img_path):
    t    = _val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    acts = {}
    def hook(m,i,o): acts["f"] = o.detach()

    # hook last block of noise stream
    h = None
    for layer in reversed(list(model.noise_net)):
        if hasattr(layer, "weight"):
            h = layer.register_forward_hook(hook); break
    if h is None: return np.zeros((256,256))

    model(t); h.remove()
    if "f" not in acts: return np.zeros((256,256))
    feat = acts["f"].squeeze(0).mean(0).cpu().numpy()
    feat = (feat-feat.min())/(feat.max()-feat.min()+1e-8)
    return cv2.resize(feat, (256,256))


# ── Result figure ─────────────────────────────────────────────────────────
def save_result_figure(img_path, ranked, cv_m, threshold, model):
    pil_img = Image.open(img_path).convert("RGB")
    img_np  = np.array(pil_img.resize((256,256)))/255.0

    top_phone, top_sim = ranked[0]
    is_match   = top_sim >= threshold
    pred_label = top_phone if is_match else "❓ Unknown"
    tc         = "#27ae60" if is_match else "#e74c3c"

    # PRNU noise residual
    bgr  = cv2.imread(str(img_path))
    bgr  = cv2.resize(bgr,(256,256)) if bgr is not None else np.zeros((256,256,3),np.uint8)
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    blur = cv2.GaussianBlur(gray,(5,5),0)
    noise = gray - blur
    noise_d = (noise-noise.min())/(noise.max()-noise.min()+1e-8)

    # activation
    try:
        act  = activation_map(model, img_path)
        hmap = cv2.cvtColor(cv2.applyColorMap((act*255).astype(np.uint8),
               cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)/255.
        overlay = (0.55*img_np + 0.45*hmap).clip(0,1)
    except Exception:
        overlay = img_np.copy()

    fig = plt.figure(figsize=(20,7))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.3)

    ax = fig.add_subplot(gs[0]); ax.imshow(img_np)
    ax.set_title("Input Image",  fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[1]); ax.imshow(noise_d, cmap="viridis")
    ax.set_title("PRNU Noise\nResidue", fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[2]); ax.imshow(overlay)
    ax.set_title("Noise Stream\nActivation", fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[3])
    phones = [r[0] for r in ranked]; sims = [r[1] for r in ranked]
    clrs   = ["#27ae60" if s>=threshold else "#3498db" for s in sims]
    bars   = ax.barh(range(len(phones)), sims, color=clrs, edgecolor="white")
    ax.set_yticks(range(len(phones)))
    ax.set_yticklabels([p.replace("_","\n") for p in phones], fontsize=8)
    ax.axvline(threshold, color="red", ls="--", lw=1.5, label=f"Threshold={threshold}")
    ax.set_xlim(-0.1, 1.15); ax.set_xlabel("Cosine Similarity")
    ax.set_title("Phone Matches", fontsize=10)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)

    ax = fig.add_subplot(gs[4]); ax.axis("off")
    rows = [[k,v] for k,v in cv_m.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Metric","Value"],
                    cellLoc="left", loc="center", colWidths=[0.65,0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,2.1)
    ax.set_title("Camera CV Analysis", fontsize=10)

    fig.suptitle(f"📱  {pred_label}   similarity={top_sim:.4f}",
                 fontsize=14, fontweight="bold", color=tc, y=1.01)
    out = OUTPUT_DIR/f"{Path(img_path).stem}_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


# ── Predict one image ─────────────────────────────────────────────────────
def predict_one(model, gallery, img_path, threshold=0.50, top_k=6, save_fig=True):
    img_path   = Path(img_path)
    ranked, _  = identify(model, img_path, gallery, top_k=top_k)
    cv_m       = cv_metrics(img_path)
    top_phone, top_sim = ranked[0]
    is_match   = top_sim >= threshold

    print(f"\n  ┌─ {img_path.name}")
    print(f"  │  {'✅ '+top_phone if is_match else '❓  Unknown'}"
          f"  (sim={top_sim:.4f}  thr={threshold})")
    for ph,sim in ranked:
        bar  = "█"*int(max(0,sim)*30)
        flag = " ← MATCH" if sim>=threshold else ""
        print(f"  │  {ph:<25} {bar:<30} {sim:.4f}{flag}")
    print(f"  └─ sharpness={cv_m.get('Sharpness','?')}  noise={cv_m.get('Noise std','?')}")

    if save_fig:
        out = save_result_figure(img_path, ranked, cv_m, threshold, model)
        print(f"  → {out}")

    return {"file": img_path.name,
            "prediction": top_phone if is_match else "Unknown",
            "similarity": top_sim, "matched": is_match}


# ── Full test-set evaluation ──────────────────────────────────────────────
def evaluate_testset(model, gallery, data_dir, threshold=0.50):
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    test_ds = PhoneDataset(data_dir, split="test")
    y_true, y_pred = [], []
    print(f"\n  Evaluating {len(test_ds)} test images...")

    for i in range(len(test_ds)):
        path = Path(test_ds.samples[i][0])
        true = test_ds.inv_label[test_ds.samples[i][1]]
        ranked, _ = identify(model, path, gallery, top_k=3)
        pred = ranked[0][0] if ranked[0][1] >= threshold else "Unknown"
        y_true.append(true); y_pred.append(pred)

    acc = sum(t==p for t,p in zip(y_true,y_pred))/len(y_true)
    print(f"\n  Test Accuracy: {acc*100:.2f}%  ({int(acc*len(y_true))}/{len(y_true)})")

    labels = sorted(set(y_true+y_pred))
    print("\n"+classification_report(y_true,y_pred,labels=labels,zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    tl = sorted(set(y_true))
    fig,ax = plt.subplots(figsize=(max(7,len(tl)),max(6,len(tl)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[l.replace("_","\n") for l in tl],
                yticklabels=[l.replace("_","\n") for l in tl], ax=ax)
    ax.set_title(f"Confusion Matrix  (acc={acc*100:.1f}%)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    out = EVAL_DIR/"test_confusion_matrix.png"
    plt.savefig(out,dpi=150); plt.close()
    print(f"  ✓ {out}")
    return acc


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     help="Single image path")
    parser.add_argument("--folder",    help="Folder of images")
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--data",      default="./data")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--top_k",     type=int,   default=6)
    parser.add_argument("--no_fig",    action="store_true")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.eval]):
        parser.print_help(); sys.exit(0)

    print("Loading model + gallery...")
    model   = load_model()
    gallery = load_gallery()
    if not gallery:
        print("[ERROR] Gallery empty! Run: python 3_gallery.py --register_all")
        sys.exit(1)
    print(f"Gallery: {len(gallery)} phones  |  threshold={args.threshold}")

    if args.eval:
        evaluate_testset(model, gallery, args.data, args.threshold)
    elif args.image:
        predict_one(model, gallery, args.image,
                    threshold=args.threshold, top_k=args.top_k,
                    save_fig=not args.no_fig)
    elif args.folder:
        exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
        imgs = [p for p in sorted(Path(args.folder).iterdir()) if p.suffix in exts]
        print(f"\nBatch: {len(imgs)} images")
        results = [predict_one(model, gallery, img,
                               threshold=args.threshold, top_k=args.top_k,
                               save_fig=not args.no_fig) for img in imgs]
        matched = sum(1 for r in results if r["matched"])
        print(f"\n{'='*58}")
        print(f"  Matched {matched}/{len(results)}")
        print(f"  {'File':<33} {'Prediction':<24} {'Sim':>6}")
        print("  "+"─"*65)
        for r in results:
            print(f"  {'✅' if r['matched'] else '❓'} "
                  f"{r['file']:<31} {r['prediction']:<24} {r['similarity']:>6.3f}")
    print("\n✅ Done.")