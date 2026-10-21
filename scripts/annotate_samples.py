"""Run the app's detection code on held-out images and save a grid of the annotated results.

Uses face_mask_app.run_inference (score >= 0.7, NMS 0.5) and the same box colors and labels
the webcam loop draws, on images from the 20% split that train_model.py does not train on.

    python scripts/annotate_samples.py            # writes docs/assets/test_samples.png
"""
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from face_mask_app import CLASSES, load_model, run_inference  # noqa: E402
from train_model import FaceMaskDataset  # noqa: E402


def draw(frame, boxes, scores, labels):
    """Same drawing as the loop in face_mask_app.main()."""
    for box, score, label in zip(boxes, scores, labels):
        color = (0, 255, 0) if label == 2 else (0, 0, 255)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{CLASSES[label]}: {score:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main(n=6, seed=3, height=330):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("fasterrcnn_model.pth", device)
    dataset = FaceMaskDataset("data/images", "data/annotations")
    _, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    picks = random.Random(seed).sample(test_idx, n)
    tiles = []
    for i in picks:
        frame = cv2.imread(os.path.join("data/images", dataset.image_files[i]))
        frame = draw(frame, *run_inference(frame, model, device))
        h, w = frame.shape[:2]
        tiles.append(cv2.resize(frame, (int(w * height / h), height), interpolation=cv2.INTER_AREA))
        print(dataset.image_files[i])
    rows, gap = [], 8
    per_row = n // 2
    for r in range(2):
        row = tiles[r * per_row:(r + 1) * per_row]
        parts = []
        for t in row:
            parts += [t, np.full((height, gap, 3), 11, np.uint8)]
        rows.append(np.hstack(parts[:-1]))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(11, 12, 13)) for r in rows]
    grid = np.vstack([rows[0], np.full((gap, width, 3), 11, np.uint8), rows[1]])
    img = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    os.makedirs("docs/assets", exist_ok=True)
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(
        "docs/assets/test_samples.png", optimize=True)


if __name__ == "__main__":
    main()
