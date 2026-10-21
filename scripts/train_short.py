"""A shorter, logged run of train_model.py.

Same dataset class, split (train_test_split, test_size=0.2, random_state=42), model
(Faster R-CNN ResNet-50 FPN with COCO weights and a 3-class head) and optimiser
(SGD, lr 0.001, momentum 0.95, weight decay 0.05, batch 2) as train_model.py. The
differences: images are loaded lazily instead of all at once; the number of epochs, the
detector's input size, a wall-clock cap and the device are arguments; and every
iteration's loss is written to results/train_log.json.

    python scripts/train_short.py --epochs 1 --min-size 480 --max-minutes 6 --device cpu
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train_model import FaceMaskDataset, collate_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min-size", type=int, default=800, help="torchvision default is 800 (max 1333)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this long and save (0 = no cap)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    dataset = FaceMaskDataset("data/images", "data/annotations")
    train_idx, _ = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=args.min_size, max_size=int(args.min_size * 1333 / 800))
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 3)
    model.to(device)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.95, weight_decay=0.05)

    log = {"epochs": args.epochs, "min_size": args.min_size, "device": args.device, "stopped_early": False, "iters_per_epoch": len(loader), "train_images": len(train_idx),
           "iter_loss": [], "epoch_loss": [], "epoch_seconds": []}
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, t0, done = 0.0, time.time(), 0
        for images, targets in loader:
            if args.max_minutes and time.time() - start > args.max_minutes * 60:
                log["stopped_early"] = True
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            done += 1
            log["iter_loss"].append(round(loss.item(), 4))
            if len(log["iter_loss"]) % 25 == 0:
                print(f"  iter {len(log['iter_loss'])}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
        log["epoch_loss"].append(round(total / max(done, 1), 4))
        log["epoch_seconds"].append(round(time.time() - t0, 1))
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {log['epoch_loss'][-1]:.4f} Time: {log['epoch_seconds'][-1]:.0f}s", flush=True)
        if log["stopped_early"]:
            break

    torch.save(model, "fasterrcnn_model.pth")
    os.makedirs("results", exist_ok=True)
    with open("results/train_log.json", "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()
