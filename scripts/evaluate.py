"""Score fasterrcnn_model.pth on the images train_model.py holds out.

train_model.py splits the dataset with train_test_split(test_size=0.2, random_state=42)
but never evaluates on the 20% it holds out. This rebuilds the same split, runs the
model with the post-processing face_mask_app.py uses (class-agnostic NMS at 0.5) and
writes results/metrics.json and results/test_predictions.json.

    python scripts/evaluate.py            # run from the repo root, after train_model.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.ops import box_iou, nms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train_model import FaceMaskDataset  # noqa: E402

IMAGES, ANNOTATIONS = "data/images", "data/annotations"
NAMES = {1: "No Mask", 2: "Masked"}  # train_model.py: with_mask -> 2, anything else -> 1
APP_SCORE_THR, NMS_THR, IOU_THR = 0.7, 0.5, 0.5


def read_xml(img_name):
    path = os.path.join(ANNOTATIONS, os.path.splitext(img_name)[0] + ".xml")
    out = []
    for obj in ET.parse(path).getroot().iter("object"):
        b = obj.find("bndbox")
        box = [int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        out.append((obj.find("name").text, box))
    return out


def average_precision(scores, tp, n_gt):
    """VOC all-point interpolated AP."""
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, dtype=float)[order]
    fp = 1 - tp
    rec = np.cumsum(tp) / max(n_gt, 1)
    prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), rec.tolist(), prec.tolist()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("fasterrcnn_model.pth", map_location=device)
    model.eval()

    dataset = FaceMaskDataset(IMAGES, ANNOTATIONS)
    _, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    to_tensor = transforms.ToTensor()

    records = []
    for i in test_idx:
        name = dataset.image_files[i]
        img = to_tensor(Image.open(os.path.join(IMAGES, name)).convert("RGB")).to(device)
        with torch.no_grad():
            p = model([img])[0]
        keep = nms(p["boxes"], p["scores"], NMS_THR)  # same as face_mask_app.run_inference
        gt = read_xml(name)
        records.append({
            "image": name,
            "gt_names": [g[0] for g in gt],
            "gt_boxes": [g[1] for g in gt],
            "gt_labels": [2 if g[0] == "with_mask" else 1 for g in gt],
            "boxes": p["boxes"][keep].cpu().numpy().round(1).tolist(),
            "scores": p["scores"][keep].cpu().numpy().round(4).tolist(),
            "labels": p["labels"][keep].cpu().numpy().tolist(),
        })

    # Per-class AP@0.5 over all kept detections, and precision/recall at the app's 0.7 cut.
    metrics = {"split": {"test_images": len(test_idx), "train_images": len(dataset) - len(test_idx)},
               "per_class": {}, "pr_curves": {}}
    for c, cname in NAMES.items():
        scores, tps, n_gt = [], [], 0
        tp_app = fp_app = 0
        for r in records:
            g = torch.tensor([b for b, l in zip(r["gt_boxes"], r["gt_labels"]) if l == c], dtype=torch.float32).reshape(-1, 4)
            n_gt += len(g)
            dets = sorted([(s, b) for s, b, l in zip(r["scores"], r["boxes"], r["labels"]) if l == c], key=lambda x: -x[0])
            used = set()
            for s, b in dets:
                hit = False
                if len(g):
                    ious = box_iou(torch.tensor([b]), g)[0]
                    j = int(ious.argmax())
                    if ious[j] >= IOU_THR and j not in used:
                        used.add(j)
                        hit = True
                scores.append(s)
                tps.append(int(hit))
                if s >= APP_SCORE_THR:
                    tp_app += hit
                    fp_app += not hit
        ap, rec, prec = average_precision(scores, tps, n_gt)
        metrics["per_class"][cname] = {
            "gt_boxes": n_gt, "ap50": round(ap, 4),
            "precision_at_0.7": round(tp_app / max(tp_app + fp_app, 1), 4),
            "recall_at_0.7": round(tp_app / max(n_gt, 1), 4),
            "tp_at_0.7": tp_app, "fp_at_0.7": fp_app,
        }
        step = max(1, len(rec) // 400)
        metrics["pr_curves"][cname] = {"recall": [round(x, 4) for x in rec[::step]],
                                       "precision": [round(x, 4) for x in prec[::step]]}
    metrics["map50"] = round(float(np.mean([v["ap50"] for v in metrics["per_class"].values()])), 4)

    # Class-agnostic matching at the app threshold: what does each ground-truth face get called?
    rows = ["with_mask", "without_mask", "mask_weared_incorrect"]
    confusion = {r: {"Masked": 0, "No Mask": 0, "missed": 0} for r in rows}
    false_alarms = {"Masked": 0, "No Mask": 0}
    for r in records:
        g = torch.tensor(r["gt_boxes"], dtype=torch.float32).reshape(-1, 4)
        dets = sorted([(s, b, l) for s, b, l in zip(r["scores"], r["boxes"], r["labels"]) if s >= APP_SCORE_THR],
                      key=lambda x: -x[0])
        taken = {}
        for s, b, l in dets:
            match = None
            if len(g):
                ious = box_iou(torch.tensor([b]), g)[0]
                for j in ious.argsort(descending=True).tolist():
                    if ious[j] < IOU_THR:
                        break
                    if j not in taken:
                        match = j
                        break
            if match is None:
                false_alarms[NAMES[l]] += 1
            else:
                taken[match] = l
        for j, name in enumerate(r["gt_names"]):
            confusion[name][NAMES[taken[j]] if j in taken else "missed"] += 1
    metrics["confusion_at_0.7"] = confusion
    metrics["unmatched_detections_at_0.7"] = false_alarms

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("results/test_predictions.json", "w") as f:
        json.dump(records, f)
    print(json.dumps({k: v for k, v in metrics.items() if k != "pr_curves"}, indent=2))


if __name__ == "__main__":
    main()
