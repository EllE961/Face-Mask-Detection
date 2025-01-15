import os
import cv2
import time
import torch
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from sklearn.model_selection import train_test_split
import torch.optim as optim

class FaceMaskDataset(Dataset):
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        xml_path = os.path.join(self.annotations_dir, img_name.replace(".png", ".xml"))
        xml_path = xml_path.replace(".jpg", ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.iter("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            if name == "with_mask":
                labels.append(2)
            else:
                labels.append(1)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        return image, target

    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    images_dir = "data/images"
    annotations_dir = "data/annotations"
    dataset = FaceMaskDataset(images_dir, annotations_dir)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    model.to(device)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.95, weight_decay=0.05)
    epochs = 10
    losses_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_t = time.time()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        avg_loss = total_loss / len(train_loader)
        losses_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Time: {time.time()-start_t:.2f}s")
    torch.save(model, "fasterrcnn_model.pth")
    plt.plot(range(1, epochs+1), losses_list, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()
