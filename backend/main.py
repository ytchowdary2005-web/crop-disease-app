"""
Crop Disease Detection API
---------------------------
Loads 4 trained Keras models (chilli, finger_millet/ragi, rice, sugarcane)
and exposes a single /predict endpoint that routes to the right model
based on the crop selected by the user.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image

app = FastAPI(title="Crop Disease Detection API")

# Allow the frontend (running on a different port/file) to call this API.
# For local dev this is fine wide-open; tighten allow_origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# CONFIG: one entry per crop.
#   - path: where the .h5 file lives (see backend/models/)
#   - img_size: (width, height) the model was trained on
#   - classes: label order MUST match train_generator.class_indices from
#     training (Keras sorts folder names alphabetically by default -
#     check the notebook output where you printed class_indices).
# ---------------------------------------------------------------------------
CROP_CONFIG = {
    "chilli": {
        "path": "models/chilli_model.h5",
        "img_size": (128, 128),
        "classes": ["Chilli__healthy", "Chilli_Leaf_Spot", "Chilli_Veinal_Mottle_Virus", "Chilli_Whitefly",
                    "Chilli_Yellowish", "Chilli_Anthracnos", "Chilli_Damping_Off", "Chilli_Leaf_Curl_virus"],  # TODO: replace with real 8 labels
    },
    "finger_millet": {
    "path": "models/finger_millet_model.h5",
    "img_size": (224, 224),
    "classes": ["downy_mildew", "healthy", "mottle", "seedling", "smut", "wilt"],
},
"rice": {
    "path": "models/rice_model_v2.h5",
    "img_size": (224, 224),
    "classes": ["Bacterial Blight Disease", "Blast Disease", "Brown Spot Disease", "False Smut Disease"],
},
   "sugarcane": {
    "path": "models/sugarcane_model.h5",
    "img_size": (224, 224),
    "classes": ["RedRot", "healthy", "olddata"],
},
}

# Cache for loaded models so each one is only loaded once, not on every request.
_loaded_models = {}


def get_model(crop: str):
    if crop not in CROP_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{crop}'")
    if crop not in _loaded_models:
        print(f"Loading model for {crop} ...")
        _loaded_models[crop] = load_model(CROP_CONFIG[crop]["path"])
    return _loaded_models[crop]


def preprocess_image(image_bytes: bytes, size: tuple) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension -> (1, H, W, 3)
    return arr


@app.get("/")
def health_check():
    return {"status": "ok", "available_crops": list(CROP_CONFIG.keys())}


@app.post("/predict")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    crop = crop.lower().strip()
    config = CROP_CONFIG.get(crop)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{crop}'. Options: {list(CROP_CONFIG.keys())}")

    model = get_model(crop)
    image_bytes = await file.read()

    try:
        input_arr = preprocess_image(image_bytes, config["img_size"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    prediction = model.predict(input_arr)[0]  # shape: (num_classes,)
    predicted_idx = int(np.argmax(prediction))
    confidence = float(prediction[predicted_idx])

    return {
        "crop": crop,
        "predicted_class": config["classes"][predicted_idx],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            label: round(float(p) * 100, 2)
            for label, p in zip(config["classes"], prediction)
        },
    }
