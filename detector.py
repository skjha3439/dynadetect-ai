import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load model ONCE when the app starts
MODEL_ID = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

def load_objects():
    with open("object_registry.json", "r") as f:
        data = json.load(f)
    return data["objects"]

def add_new_object(name):
    objects = load_objects()
    if name not in objects:
        objects.append(name)
        with open("object_registry.json", "w") as f:
            json.dump({"objects": objects}, f)
    return objects

def detect_objects(image_path):
    image = Image.open(image_path).convert("RGB")
    objects = load_objects()
    text_prompt = " . ".join(objects) + " ."

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    detections = []
    for box, score, label in zip(
        results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    ):
        detections.append({
            "label": label,
            "score": round(score.item(), 2),
            "box": box.tolist()
        })

    return detections
