
from ultralytics import YOLO
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Load a model
    model = YOLO('./runs/detect/train/weights/best.pt')

    # Run batched inference on a list of images
    model(r"./img",save=True, imgsz=640)