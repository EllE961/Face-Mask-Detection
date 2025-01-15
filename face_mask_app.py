import torch
import cv2
import numpy as np
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image

CLASSES = ["Background", "No Mask", "Masked"]

def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def run_inference(frame, model, device, score_thr=0.7, nms_thr=0.5):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor_img = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(tensor_img)
    boxes = preds[0]["boxes"]
    scores = preds[0]["scores"]
    labels = preds[0]["labels"]
    keep = nms(boxes, scores, nms_thr)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    mask = scores >= score_thr
    return boxes[mask], scores[mask], labels[mask]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "fasterrcnn_model.pth"
    model = load_model(model_path, device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores, labels = run_inference(frame, model, device)
        for box, score, label in zip(boxes, scores, labels):
            color = (0, 255, 0) if label == 2 else (0, 0, 255)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            txt = f"{CLASSES[label]}: {score:.2f}"
            cv2.putText(frame, txt, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
