from dotenv import load_dotenv
import os
import json
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Load environment variables from .env ────────────────
load_dotenv()

MODEL_ID        = os.getenv("MODEL_ID", "IDEA-Research/grounding-dino-base")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8l.pt")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B/32")
THRESHOLD       = float(os.getenv("DETECTION_THRESHOLD", 0.3))
GDINO_THRESHOLD = float(os.getenv("GDINO_THRESHOLD", 0.25))
DIMENSION       = int(os.getenv("FAISS_DIMENSION", 512))

# ── Device Setup ─────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {device.upper()}")

# ────────────────────────────────────────────────────────
# LAYER 1a: YOLOv8 — Fast detector for COCO 80 classes
# ────────────────────────────────────────────────────────
print(f"[L1a] Loading YOLOv8: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.overrides['iou'] = 0.45  # reduce overlapping boxes

# Get all YOLO known class names
YOLO_CLASSES = set(yolo_model.names.values())
print(f"[L1a] YOLOv8 knows {len(YOLO_CLASSES)} classes")

# ────────────────────────────────────────────────────────
# LAYER 1b: Grounding DINO — Zero-shot detector for ANY object
# ────────────────────────────────────────────────────────
print(f"[L1b] Loading Grounding DINO: {MODEL_ID}")
gdino_processor = AutoProcessor.from_pretrained(MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print(f"[L1b] Grounding DINO ready!")

# ────────────────────────────────────────────────────────
# LAYER 2: CLIP Backbone — Rich visual-language embeddings
# ────────────────────────────────────────────────────────
print(f"[L2] Loading CLIP: {CLIP_MODEL_NAME}")
clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
print(f"[L2] CLIP ready!")

# ────────────────────────────────────────────────────────
# LAYER 4: FAISS Vector Store — Prototype matching
# ────────────────────────────────────────────────────────
print(f"[L4] Initializing FAISS index, dimension: {DIMENSION}")
index = faiss.IndexFlatL2(DIMENSION)
prototype_store = {}  # name -> embedding
prototype_names = []  # ordered list for index lookup


# ────────────────────────────────────────────────────────
# EMBEDDING FUNCTIONS
# ────────────────────────────────────────────────────────

def extract_image_embedding(image_path):
    """Extract CLIP embedding from an image file"""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def extract_text_embedding(text):
    """Extract CLIP embedding from a text string"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokens)
    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


# ────────────────────────────────────────────────────────
# OBJECT REGISTRY
# ────────────────────────────────────────────────────────

def load_objects():
    """Load all registered objects from JSON"""
    with open("object_registry.json", "r") as f:
        return json.load(f)["objects"]


def save_objects(objects):
    """Save objects list to JSON"""
    with open("object_registry.json", "w") as f:
        json.dump({"objects": objects}, f, indent=2)


def add_new_object(name):
    """
    Layer 4: Add new object using TEXT embedding.
    Works for ANY object — no retraining needed!
    CLIP already understands millions of concepts.
    """
    objects = load_objects()
    if name not in objects:
        # Register via CLIP text embedding
        embedding = extract_text_embedding(name)
        index.add(embedding)
        prototype_names.append(name)
        prototype_store[name] = embedding
        objects.append(name)
        save_objects(objects)
        print(f"[L4] '{name}' registered via CLIP text embedding!")
    return objects


def register_object_with_image(name, image_path):
    """
    Layer 4: Register new object using IMAGE embedding.
    More accurate than text for visual matching.
    """
    objects = load_objects()
    if name not in objects:
        embedding = extract_image_embedding(image_path)
        index.add(embedding)
        prototype_names.append(name)
        prototype_store[name] = embedding
        objects.append(name)
        save_objects(objects)
        print(f"[L4] '{name}' registered via image embedding!")
    return objects


def match_to_prototype(embedding):
    """
    Layer 4: Match detection to nearest prototype in FAISS.
    Uses normalized cosine similarity.
    """
    if index.ntotal == 0:
        return "unknown", 0.0
    distances, indices = index.search(embedding, k=1)
    idx = indices[0][0]
    score = float(1 / (1 + distances[0][0]))
    if idx < len(prototype_names):
        return prototype_names[idx], score
    return "unknown", score


# ────────────────────────────────────────────────────────
# LAYER 1b: GROUNDING DINO DETECTION
# ────────────────────────────────────────────────────────

def detect_with_gdino(image, object_names):
    """
    Use Grounding DINO to detect ANY object by name.
    This is the core zero-shot detection capability.
    Works for glasses, pen, id card — anything!
    """
    if not object_names:
        return []

    # Build text prompt from object names
    text_prompt = " . ".join(object_names) + " ."

    inputs = gdino_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # Post process results
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=GDINO_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )

    detections = []
    for box, score, label in zip(
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"]
    ):
        x1, y1, x2, y2 = map(int, box.tolist())
        detections.append({
            "label": str(label),
            "yolo_label": "grounding_dino",
            "match_score": round(float(score), 2),
            "confidence": round(float(score), 2),
            "box": [x1, y1, x2, y2],
            "detector": "grounding_dino"
        })

    return detections


# ────────────────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ────────────────────────────────────────────────────────

def detect_objects(image_path):
    """
    Full 7-Layer detection pipeline:

    Step 1 → Load all registered objects
    Step 2 → Split into YOLO-known vs custom objects
    Step 3 → YOLOv8 detects COCO objects (fast)
    Step 4 → CLIP matches each detection to prototypes
    Step 5 → Grounding DINO detects ALL objects (zero-shot)
    Step 6 → Merge and return all detections

    Result: Detects ANYTHING — with or without retraining!
    """
    image = Image.open(image_path).convert("RGB")
    all_objects = load_objects()
    detections = []

    # ── Split objects into YOLO-known vs custom ──────────
    yolo_objects  = [o for o in all_objects if o in YOLO_CLASSES]
    custom_objects = [o for o in all_objects if o not in YOLO_CLASSES]

    print(f"[DETECT] YOLO objects: {yolo_objects}")
    print(f"[DETECT] Custom objects (DINO): {custom_objects}")

    # ── Step 3+4: YOLOv8 for known COCO objects ─────────
    yolo_results = yolo_model(image_path, device=device)
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_label = yolo_model.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 2)

            # Skip low confidence
            if confidence < THRESHOLD:
                continue

            # CLIP prototype matching for better labeling
            try:
                crop = image.crop((x1, y1, x2, y2))
                crop.save("temp_crop.jpg")
                crop_embedding = extract_image_embedding("temp_crop.jpg")
                matched_label, match_score = match_to_prototype(crop_embedding)
                # Use matched label only if confident enough
                final_label = matched_label if match_score > THRESHOLD else yolo_label
            except Exception:
                final_label = yolo_label
                match_score = 0.0

            detections.append({
                "label": final_label,
                "yolo_label": yolo_label,
                "match_score": round(match_score, 2),
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
                "detector": "yolov8"
            })

    # ── Step 5: Grounding DINO for ALL registered objects
    # Run DINO on ALL objects for maximum coverage
    if all_objects:
        print(f"[L1b] Running Grounding DINO on: {all_objects}")
        gdino_detections = detect_with_gdino(image, all_objects)
        detections.extend(gdino_detections)
        print(f"[L1b] DINO found {len(gdino_detections)} detections")

    # ── Cleanup temp files ───────────────────────────────
    if os.path.exists("temp_crop.jpg"):
        os.remove("temp_crop.jpg")

    print(f"[DETECT] Total detections: {len(detections)}")
    return detections
