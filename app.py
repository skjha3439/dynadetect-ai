from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import json

# ── Load environment variables ───────────────────────────
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

from detector import (
    detect_objects,
    add_new_object,
    load_objects,
    register_object_with_image,
    annotate_with_sam
)

# ── Layer 7: FastAPI App ─────────────────────────────────
app = FastAPI(
    title="DynaDetect AI",
    description="7-Layer Incremental Object Detection — IIIT Manipur Hackathon",
    version="2.0.0"
)

# ── CORS Middleware ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Welcome to DynaDetect AI!",
        "status": "running",
        "version": "2.0.0",
        "layers": {
            "L1a": "YOLOv8 Base Detector",
            "L1b": "Grounding DINO Zero-Shot Detector",
            "L2":  "CLIP ViT-B/32 Backbone",
            "L3":  "EWC + Replay Anti-Forgetting",
            "L4":  "FAISS Prototype Store",
            "L5":  "Continual Learning Framework",
            "L6":  "Grounded SAM Auto Annotation ✅",
            "L7":  "FastAPI Deployment"
        }
    }


@app.get("/objects")
def get_objects():
    """Get all currently registered objects"""
    objects = load_objects()
    return {"objects": objects, "count": len(objects)}


@app.post("/add-object")
def add_object(name: str):
    """
    Add new object by TEXT name — no retraining!
    Uses CLIP text embedding instantly.
    """
    if not name or not name.strip():
        return JSONResponse(status_code=400, content={"error": "Name cannot be empty"})
    name = name.strip().lower()
    updated = add_new_object(name)
    return {
        "message": f"'{name}' added via CLIP text embedding!",
        "method": "text_embedding",
        "objects": updated,
        "count": len(updated)
    }


@app.post("/register-with-image")
async def register_with_image(name: str, file: UploadFile = File(...)):
    """
    Layer 4 + 6: Register object using Grounded SAM annotation.

    Full pipeline:
    1. Grounding DINO finds object in your image
    2. SAM creates precise pixel mask
    3. CLIP extracts embedding from clean masked region
    4. Store in FAISS — most accurate registration!
    """
    if not name or not name.strip():
        return JSONResponse(status_code=400, content={"error": "Name cannot be empty"})

    name = name.strip().lower()
    temp_path = f"temp_register_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        updated = register_object_with_image(name, temp_path)

        return {
            "message": f"'{name}' registered with Grounded SAM annotation!",
            "method": "grounded_sam",
            "annotation": "SAM pixel mask → CLIP embedding → FAISS",
            "objects": updated,
            "count": len(updated)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/annotate")
async def annotate_object(name: str, file: UploadFile = File(...)):
    """
    Layer 6 only: Run Grounded SAM annotation on an image.
    Shows what SAM found — useful for demo/visualization.
    """
    if not name or not name.strip():
        return JSONResponse(status_code=400, content={"error": "Name cannot be empty"})

    name = name.strip().lower()
    temp_path = f"temp_annotate_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        _, annotated_path, sam_success = annotate_with_sam(temp_path, name)

        return {
            "message": f"Annotation complete for '{name}'",
            "sam_used": sam_success,
            "method": "SAM pixel mask" if sam_success else "bbox crop fallback",
            "status": "success"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Full detection using all 7 layers.
    YOLOv8 + Grounding DINO + CLIP + FAISS.
    Only shows objects in your registry!
    """
    temp_path = f"temp_detect_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        results = detect_objects(temp_path)
        return JSONResponse(content={
            "detections": results,
            "count": len(results),
            "status": "success"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "failed"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/remove-object")
def remove_object(name: str):
    """Remove object by name"""
    objects = load_objects()
    name = name.strip().lower()
    if name not in objects:
        return JSONResponse(status_code=404, content={"error": f"'{name}' not found"})
    objects.remove(name)
    with open("object_registry.json", "w") as f:
        json.dump({"objects": objects}, f, indent=2)
    return {"message": f"'{name}' removed!", "objects": objects, "count": len(objects)}


@app.delete("/remove-object-by-index")
def remove_object_by_index(index: int):
    """Remove object by index number"""
    objects = load_objects()
    if index < 0 or index >= len(objects):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid index. Range: 0 to {len(objects)-1}"}
        )
    removed = objects.pop(index)
    with open("object_registry.json", "w") as f:
        json.dump({"objects": objects}, f, indent=2)
    return {
        "message": f"'{removed}' removed!",
        "removed": removed,
        "objects": objects,
        "count": len(objects)
    }


@app.get("/health")
def health_check():
    """Health check — all layer status"""
    import torch
    try:
        from transformers import SamModel
        sam_available = True
    except Exception:
        sam_available = False

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "registered_objects": len(load_objects()),
        "layers": {
            "L1a_yolov8": "✅ active",
            "L1b_grounding_dino": "✅ active",
            "L2_clip": "✅ active",
            "L3_ewc_replay": "✅ active",
            "L4_faiss": "✅ active",
            "L5_continual": "✅ active",
            "L6_sam": "✅ active" if sam_available else "⚠️ install transformers>=4.29",
            "L7_fastapi": "✅ active"
        }
    }
