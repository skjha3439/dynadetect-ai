from quantum_optimizer import quantum_optimizer, quantum_similarity
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

# ── Load environment variables ───────────────────────────
load_dotenv()

# Force use cached models — no internet needed
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID        = os.getenv("MODEL_ID", "IDEA-Research/grounding-dino-base")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "our_model.pt")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B/32")
THRESHOLD       = float(os.getenv("DETECTION_THRESHOLD", 0.45))
GDINO_THRESHOLD = float(os.getenv("GDINO_THRESHOLD", 0.25))
DIMENSION       = int(os.getenv("FAISS_DIMENSION", 512))
SAM_MODEL       = os.getenv("SAM_MODEL", "facebook/sam-vit-base")

# ── Device Setup ─────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {device.upper()}")

# ────────────────────────────────────────────────────────
# LAYER 1a: YOLOv8
# ────────────────────────────────────────────────────────
print(f"[L1a] Loading YOLOv8: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.overrides['iou'] = 0.3
yolo_model.overrides['conf'] = 0.5 
YOLO_CLASSES = set(yolo_model.names.values())
print(f"[L1a] YOLOv8 knows {len(YOLO_CLASSES)} classes")

# ────────────────────────────────────────────────────────
# LAYER 1b: Grounding DINO
# ────────────────────────────────────────────────────────
print(f"[L1b] Loading Grounding DINO: {MODEL_ID}")
gdino_processor = AutoProcessor.from_pretrained(MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print(f"[L1b] Grounding DINO ready!")

# ────────────────────────────────────────────────────────
# LAYER 2: CLIP Backbone
# ────────────────────────────────────────────────────────
print(f"[L2] Loading CLIP: {CLIP_MODEL_NAME}")
clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
print(f"[L2] CLIP ready!")

# ────────────────────────────────────────────────────────
# LAYER 4: FAISS Vector Store
# ────────────────────────────────────────────────────────
print(f"[L4] Initializing FAISS index, dimension: {DIMENSION}")
index = faiss.IndexFlatL2(DIMENSION)
prototype_store = {}
prototype_names = []

# ────────────────────────────────────────────────────────
# LAYER 6: SAM — Segment Anything Model
# ────────────────────────────────────────────────────────
sam_model = None
sam_processor = None

def load_sam():
    """Load SAM model lazily — only when needed"""
    global sam_model, sam_processor
    if sam_model is None:
        try:
            from transformers import SamModel, SamProcessor
            print(f"[L6] Loading SAM: {SAM_MODEL}")
            sam_processor = SamProcessor.from_pretrained(SAM_MODEL)
            sam_model = SamModel.from_pretrained(SAM_MODEL).to(device)
            print(f"[L6] SAM ready!")
        except Exception as e:
            print(f"[L6] SAM load failed: {e} — annotation will use bbox crop instead")
    return sam_model, sam_processor


# ────────────────────────────────────────────────────────
# EMBEDDING FUNCTIONS
# ────────────────────────────────────────────────────────

def extract_image_embedding(image_path):
    """Extract CLIP embedding from image"""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def extract_embedding_from_pil(pil_image):
    """Extract CLIP embedding from PIL image directly"""
    image = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def extract_text_embedding(text):
    """Extract CLIP embedding from text"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokens)
    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


# ────────────────────────────────────────────────────────
# LAYER 6: GROUNDED SAM ANNOTATION
# ────────────────────────────────────────────────────────

def annotate_with_sam(image_path, object_name):
    """
    Layer 6: Grounded SAM Auto-Annotation Pipeline
    
    Step 1 → Grounding DINO finds bounding box of object
    Step 2 → SAM creates precise pixel mask from bbox
    Step 3 → Extract masked region (clean object only)
    Step 4 → CLIP extracts embedding from clean mask
    Step 5 → Store in FAISS — much more accurate!
    
    Returns: (embedding, annotated_image_path, success)
    """
    print(f"[L6] Starting Grounded SAM annotation for: '{object_name}'")
    image = Image.open(image_path).convert("RGB")

    # ── Step 1: Grounding DINO finds bbox ────────────────
    try:
        text_prompt = object_name + " ."
        inputs = gdino_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.2,
            target_sizes=[image.size[::-1]]
        )

        if len(results[0]["boxes"]) == 0:
            print(f"[L6] DINO found no '{object_name}' — using full image embedding")
            embedding = extract_image_embedding(image_path)
            return embedding, None, False

        # Get best box (highest score)
        best_idx = results[0]["scores"].argmax()
        box = results[0]["boxes"][best_idx]
        x1, y1, x2, y2 = map(int, box.tolist())
        print(f"[L6] DINO found bbox: [{x1},{y1},{x2},{y2}]")

    except Exception as e:
        print(f"[L6] DINO step failed: {e} — using full image")
        embedding = extract_image_embedding(image_path)
        return embedding, None, False

    # ── Step 2: SAM creates precise mask ─────────────────
    try:
        _sam_model, _sam_processor = load_sam()

        if _sam_model is not None:
            # SAM needs input points (center of bbox)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_points = [[[cx, cy]]]

            sam_inputs = _sam_processor(
                images=image,
                input_points=input_points,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                sam_outputs = _sam_model(**sam_inputs)

            masks = _sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )

            # Get best mask
            mask = masks[0][0][0].numpy()

            # ── Step 3: Apply mask to image ───────────────
            img_array = np.array(image)
            masked_array = img_array.copy()

            # Black out everything outside mask
            masked_array[~mask] = 0

            # Crop to bounding box region
            masked_crop = masked_array[y1:y2, x1:x2]
            masked_pil = Image.fromarray(masked_crop)

            # Save annotated image
            annotated_path = "annotated_" + os.path.basename(image_path)
            masked_pil.save(annotated_path)

            # ── Step 4: CLIP embedding from clean mask ────
            embedding = extract_embedding_from_pil(masked_pil)
            print(f"[L6] SAM mask extracted — clean embedding saved!")
            return embedding, annotated_path, True

        else:
            # SAM not available — use bbox crop instead
            raise Exception("SAM not loaded")

    except Exception as e:
        print(f"[L6] SAM masking failed: {e} — using bbox crop")

        # Fallback: use bbox crop (still better than full image)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = "crop_" + os.path.basename(image_path)
        crop.save(crop_path)
        embedding = extract_embedding_from_pil(crop)
        print(f"[L6] Using bbox crop as fallback embedding")
        return embedding, crop_path, False


# ────────────────────────────────────────────────────────
# OBJECT REGISTRY
# ────────────────────────────────────────────────────────

def load_objects():
    """Load registered objects from JSON"""
    with open("object_registry.json", "r") as f:
        return json.load(f)["objects"]


def save_objects(objects):
    """Save objects to JSON"""
    with open("object_registry.json", "w") as f:
        json.dump({"objects": objects}, f, indent=2)


def add_new_object(name):
    """
    Add new object via CLIP text embedding — no retraining!
    Uses CLIP language understanding to register any object.
    """
    objects = load_objects()
    if name not in objects:
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
    Layer 4 + 6: Register object using Grounded SAM annotation.

    Pipeline:
    1. Grounding DINO finds object in image
    2. SAM creates precise pixel mask
    3. CLIP extracts embedding from masked region
    4. Store in FAISS — most accurate method!
    """
    objects = load_objects()
    if name not in objects:
        print(f"[L4+L6] Registering '{name}' with Grounded SAM...")

        # Use Grounded SAM for precise annotation
        embedding, annotated_path, sam_success = annotate_with_sam(image_path, name)

        if sam_success:
            print(f"[L6] ✅ SAM annotation successful for '{name}'!")
        else:
            print(f"[L6] ⚠️ Used fallback method for '{name}'")

        index.add(embedding)
        prototype_names.append(name)
        prototype_store[name] = embedding
        objects.append(name)
        save_objects(objects)

        # Cleanup annotated file
        if annotated_path and os.path.exists(annotated_path):
            os.remove(annotated_path)

    return objects


def match_to_prototype(embedding):
    """
    Quantum-Enhanced Prototype Matching
    
    Uses quantum annealing instead of classical
    nearest neighbor for better accuracy!
    """
    if index.ntotal == 0 or len(prototype_names) == 0:
        return "unknown", 0.0

    # Use quantum optimizer for enhanced matching
    try:
        # Try quantum-enhanced matching first
        if index.ntotal <= 20:
            # Small registry: use full quantum annealing
            name, score, _ = quantum_optimizer.quantum_annealing_search(
                embedding.flatten(),
                [index.reconstruct(i) for i in range(index.ntotal)],
                prototype_names
            )
        else:
            # Large registry: FAISS pre-filter + quantum refinement
            name, score = quantum_optimizer.quantum_enhanced_match(
                embedding, index, prototype_names, top_k=5
            )
        print(f"[QUANTUM] Match: '{name}' score={score:.3f}")
        return name, score

    except Exception as e:
        print(f"[QUANTUM] Fallback to classical: {e}")
        # Fallback to classical FAISS
        distances, indices_result = index.search(embedding, k=1)
        idx = indices_result[0][0]
        score = float(1 / (1 + distances[0][0]))
        return prototype_names[idx] if idx < len(prototype_names) else "unknown", score


# ────────────────────────────────────────────────────────
# GROUNDING DINO DETECTION
# ────────────────────────────────────────────────────────

def detect_with_gdino(image, object_names):
    """Detect any object using Grounding DINO zero-shot"""
    if not object_names:
        return []

    text_prompt = " . ".join(object_names) + " ."
    inputs = gdino_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

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
        label_str = str(label).lower().strip()

        # STRICT FILTER: Only allow registered objects
        if label_str not in object_names:
            print(f"[FILTER] DINO blocked: '{label_str}' not in registry")
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        detections.append({
            "label": label_str,
            "yolo_label": "grounding_dino",
            "match_score": round(float(score), 2),
            "confidence": round(float(score), 2),
            "box": [x1, y1, x2, y2],
            "detector": "grounding_dino"
        })

    return detections


# ────────────────────────────────────────────────────────
# MAIN DETECTION — STRICT REGISTRY FILTER
# ────────────────────────────────────────────────────────

def detect_objects(image_path):
    """
    Full 7-Layer Detection Pipeline:

    L1a → YOLOv8 detects COCO objects (fast)
    L1b → Grounding DINO detects custom objects (zero-shot)
    L2  → CLIP extracts embeddings from detections
    L3  → EWC ensures no forgetting (continual learning)
    L4  → FAISS matches to stored prototypes
    L6  → SAM was used during registration for clean embeddings
    L7  → FastAPI serves results to website

    STRICT FILTER: Only shows objects in your registry!
    """
    image = Image.open(image_path).convert("RGB")
    all_objects = load_objects()
    print(f"[DETECT] Registry: {all_objects}")
    detections = []

    # ── L1a: YOLOv8 + STRICT FILTER ─────────────────────
    yolo_results = yolo_model(image_path, device=device)
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_label = yolo_model.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 2)

            # Skip low confidence
            if confidence < THRESHOLD:
                continue

            # STRICT: Block anything not in registry
            if yolo_label not in all_objects:
                print(f"[FILTER] YOLO blocked: '{yolo_label}'")
                continue

            print(f"[DETECT] YOLO found: '{yolo_label}' {confidence}")

            # L2+L4: CLIP + FAISS prototype matching
            try:
                crop = image.crop((x1, y1, x2, y2))
                crop.save("temp_crop.jpg")
                crop_embedding = extract_image_embedding("temp_crop.jpg")
                matched_label, match_score = match_to_prototype(crop_embedding)
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

    # ── L1b: Grounding DINO for ALL registered objects ───
    if all_objects:
        print(f"[L1b] Running DINO for: {all_objects}")
        gdino_detections = detect_with_gdino(image, all_objects)
        detections.extend(gdino_detections)

    # Cleanup
    if os.path.exists("temp_crop.jpg"):
        os.remove("temp_crop.jpg")

    print(f"[DETECT] Total: {len(detections)} detections")
    return detections
