from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# ── Load environment variables from .env ────────────────
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

from detector import detect_objects, add_new_object, load_objects, register_object_with_image

# ── Layer 7: FastAPI App ─────────────────────────────────
app = FastAPI(
    title="DynaDetect AI",
    description="7-Layer Incremental Object Detection — IIIT Manipur Hackathon",
    version="1.0.0"
)

# ── CORS Middleware (allows website to talk to API) ──────
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
    """Home route — check if server is running"""
    return {
        "message": "Welcome to DynaDetect AI!",
        "status": "running",
        "layers": {
            "L1": "YOLOv8 Base Detector",
            "L2": "CLIP ViT-B/32 Backbone",
            "L3": "EWC + Replay Anti-Forgetting",
            "L4": "FAISS Prototype Store",
            "L5": "Continual Learning Framework",
            "L6": "Grounded SAM Auto Annotation",
            "L7": "FastAPI Deployment"
        }
    }


@app.get("/objects")
def get_objects():
    """Get all currently registered objects"""
    objects = load_objects()
    return {
        "objects": objects,
        "count": len(objects)
    }


@app.post("/add-object")
def add_object(name: str):
    """
    Add a new object by name — NO retraining needed!
    Uses CLIP text embedding to register the object instantly.
    """
    if not name or not name.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Object name cannot be empty"}
        )
    name = name.strip().lower()
    updated = add_new_object(name)
    return {
        "message": f"'{name}' added successfully via CLIP embedding!",
        "objects": updated,
        "count": len(updated)
    }


@app.post("/register-with-image")
async def register_with_image(name: str, file: UploadFile = File(...)):
    """
    Register a new object using an actual image — more accurate than text.
    Extracts CLIP image embedding and stores in FAISS prototype store.
    """
    if not name or not name.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Object name cannot be empty"}
        )

    name = name.strip().lower()
    temp_path = f"temp_register_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        updated = register_object_with_image(name, temp_path)

        return {
            "message": f"'{name}' registered with image embedding!",
            "objects": updated,
            "count": len(updated)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image.
    Uses YOLOv8 for detection + CLIP for prototype matching.
    Returns bounding boxes, labels, and confidence scores.
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
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/remove-object")
def remove_object(name: str):
    """Remove an object from the registry"""
    import json
    objects = load_objects()
    name = name.strip().lower()
    if name not in objects:
        return JSONResponse(
            status_code=404,
            content={"error": f"'{name}' not found in registry"}
        )
    objects.remove(name)
    with open("object_registry.json", "w") as f:
        json.dump({"objects": objects}, f)
    return {
        "message": f"'{name}' removed from registry",
        "objects": objects,
        "count": len(objects)
    }


@app.get("/health")
def health_check():
    """Health check — shows GPU status and registered objects"""
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "registered_objects": len(load_objects()),
        "host": HOST,
        "port": PORT
    }
