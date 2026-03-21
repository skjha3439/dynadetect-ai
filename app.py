from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os
from detector import detect_objects, add_new_object, load_objects

app = FastAPI()

# See all registered objects
@app.get("/objects")
def get_objects():
    return {"objects": load_objects()}

# Add a new object WITHOUT retraining
@app.post("/add-object")
def add_object(name: str):
    updated = add_new_object(name)
    return {"message": f"'{name}' added!", "objects": updated}

# Upload image and detect
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    results = detect_objects(temp_path)
    os.remove(temp_path)  # cleanup

    return JSONResponse(content={"detections": results})
