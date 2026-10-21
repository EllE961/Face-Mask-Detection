"""Figures from results/train_log.json and results/metrics.json.

    python scripts/make_figures.py
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "assets")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def dataset_stats():
    """Class counts, faces per image and face sizes from the XML files (cached in results/)."""
    path = os.path.join(RES, "dataset_stats.json")
    xmls = sorted(glob.glob(os.path.join(ROOT, "data", "annotations", "*.xml")))
    if not xmls:
        return json.load(open(path))
    names, per_image, widths = [], [], []
    for x in xmls:
        objs = list(ET.parse(x).getroot().iter("object"))
        per_image.append(len(objs))
        for o in objs:
            names.append(o.find("name").text)
            b = o.find("bndbox")
            widths.append(int(b.find("xmax").text) - int(b.find("xmin").text))
    stats = {"images": len(xmls), "faces": len(names),
             "class_counts": {n: names.count(n) for n in ("with_mask", "without_mask", "mask_weared_incorrect")},
             "faces_per_image": per_image, "face_widths": widths}
    os.makedirs(RES, exist_ok=True)
    json.dump(stats, open(path, "w"))
    return stats


def dataset(st):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    labels = ["with_mask", "without_mask", "mask worn\nincorrectly"]
    counts = list(st["class_counts"].values())
    a1.bar(range(3), counts, color=[TEMPER, EMBER, PURPLE], width=0.62)
    for i, c in enumerate(counts):
        a1.text(i, c + 40, f"{c:,}", ha="center", va="bottom", fontsize=9, color=INK2)
    a1.set_xticks(range(3))
    a1.set_xticklabels(labels, fontsize=8.5)
    a1.set_ylim(0, max(counts) * 1.15)
    a1.set_title(f"{st['faces']:,} labeled faces")
    per = np.array(st["faces_per_image"])
    a2.hist(per, bins=np.arange(0.5, min(per.max(), 20) + 1.5), color=TEMPER, edgecolor=PANEL)
    a2.set_xlabel("faces in the image")
    a2.set_title(f"Faces per image ({st['images']} images)")
    w = np.array(st["face_widths"])
    a3.hist(w, bins=np.arange(0, 200, 6), color=EMBER, edgecolor=PANEL)
    a3.set_xlabel("face box width (px)")
    a3.set_title(f"Face size, median {int(np.median(w))} px")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "dataset.png")


def training(log):
    loss = np.array(log["iter_loss"])
    k = 25
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(np.arange(len(loss)) + 1, loss, color=EMBER, alpha=0.45, lw=1)
    ax.plot(np.arange(k, len(loss) + 1), smooth, color=EMBER)
    for e in range(1, len(log["epoch_loss"])):
        ax.axvline(e * log["iters_per_epoch"], color=INK3, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(1, len(loss))
    ax.set_xlabel("iteration (batch of 2 images)")
    ax.set_ylabel("summed Faster R-CNN loss")
    ax.set_title("Training loss per iteration")
    if log.get("stopped_early"):
        what = (f"{len(loss)} iterations ({2 * len(loss)} of {log['train_images']} training images) before the time cap, "
                f"on {log.get('device', 'cuda')} at min_size {log.get('min_size', 800)}.")
    else:
        what = f"{log['epochs']} epochs × {log['iters_per_epoch']} iterations on {log['train_images']} images."
    fig.text(0.012, -0.03, f"{what} Thin line: each iteration, thick line: mean of the last {k}.",
             color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "training_loss.png")


def evaluation(m):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    for (name, c), color in zip(m["pr_curves"].items(), (TEMPER, EMBER)):
        pc = m["per_class"][name]
        a1.plot(c["recall"], c["precision"], color=color,
                label=f"{name}: AP50 {pc['ap50']:.3f}  ({pc['gt_boxes']} faces)")
        a1.scatter([pc["recall_at_0.7"]], [pc["precision_at_0.7"]], color=color, s=36, zorder=3)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("recall")
    a1.set_ylabel("precision")
    a1.set_title("Precision-recall on the held-out 20%")
    a1.legend(loc="lower left")
    a1.text(0.99, 0.97, "dots: the app's 0.7 score cut", color=INK3, fontsize=8.5, ha="right", va="top")

    rows = ["with_mask", "without_mask", "mask_weared_incorrect"]
    cols = ["Masked", "No Mask", "missed"]
    cm = np.array([[m["confusion_at_0.7"][r][c] for c in cols] for r in rows], float)
    share = cm / cm.sum(axis=1, keepdims=True)
    a2.imshow(share, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            a2.text(j, i, f"{int(cm[i, j])}\n{share[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9,
                    color="#141110" if share[i, j] > 0.55 else INK2)
    a2.set_xticks(range(len(cols)))
    a2.set_xticklabels(["app says Masked", "app says No Mask", "no box"])
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels(["with_mask", "without_mask", "mask worn\nincorrectly"])
    a2.grid(False)
    a2.set_title("What each annotated face gets at score ≥ 0.7")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "evaluation.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    dataset(dataset_stats())
    training(json.load(open(os.path.join(RES, "train_log.json"))))
    evaluation(json.load(open(os.path.join(RES, "metrics.json"))))
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
