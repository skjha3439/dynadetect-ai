from dotenv import load_dotenv
import os
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.getenv("TRAIN_EPOCHS", 50))
BATCH  = int(os.getenv("TRAIN_BATCH", 16))
IMG_SIZE = int(os.getenv("TRAIN_IMG_SIZE", 640))
DATA_YAML = os.getenv("DATA_YAML", "coco128.yaml")
OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", "our_model.pt")

print(f"[TRAIN] Device: {DEVICE}")
print(f"[TRAIN] Epochs: {EPOCHS}")
print(f"[TRAIN] Batch: {BATCH}")

def train():
    print("[TRAIN] Loading YOLOv8 base model...")

    # Start from pretrained YOLOv8 — transfer learning
    model = YOLO("yolov8l.pt")

    print(f"[TRAIN] Starting training on {DATA_YAML}...")
    print("[TRAIN] This is our ONE-TIME training!")

    torch.cuda.empty_cache()
    # ── Train the model ──────────────────────────────────
    results = model.train(
        data=DATA_YAML,         # dataset config
        epochs=EPOCHS,          # number of epochs
        batch=4,            # batch size
        imgsz=416,         # image size
        device=DEVICE,          # GPU/CPU
        patience=10,            # early stopping
        save=True,              # save best weights
        project="runs/train",   # output folder
        name="dynadetect",      # experiment name
        pretrained=True,        # use pretrained weights
        optimizer="AdamW",      # optimizer
        lr0=0.001,              # learning rate
        weight_decay=0.0005,    # regularization
        augment=True,           # data augmentation
        cos_lr=True,   
        workers=2,   
        cache=False,        # cosine LR schedule
        verbose=True
    )

    # ── Save our trained model ───────────────────────────
    best_weights = "runs/train/dynadetect/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"[TRAIN] ✅ Model saved to: {OUTPUT_MODEL}")
    else:
        print("[TRAIN] ⚠️ Best weights not found, check runs/train/")

    print("[TRAIN] Training complete!")
    print(f"[TRAIN] Now update .env: YOLO_MODEL={OUTPUT_MODEL}")
    return results


def validate():
    """Validate our trained model"""
    model = YOLO(OUTPUT_MODEL)
    metrics = model.val(data=DATA_YAML, device=DEVICE)
    print(f"[VAL] mAP50: {metrics.box.map50:.3f}")
    print(f"[VAL] mAP50-95: {metrics.box.map:.3f}")
    return metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "val":
        validate()
    else:
        train()
