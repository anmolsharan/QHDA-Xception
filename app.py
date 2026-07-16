import os

# =========================================================
# TENSORFLOW SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import traceback
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from quantum_layer import QuantumLayer
from google.cloud import storage

tf.get_logger().setLevel("ERROR")

# =========================================================
# PATHS
# =========================================================

CONFIG_PATH = "config.json"
CLASSES_PATH = "classes.json"
MODEL_PATH = "xcept_quant_model.keras"

# =========================================================
# LOAD CONFIG
# =========================================================

print("=" * 60)
print("LOADING CONFIGURATION FILES...")
print("=" * 60)

try:

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)

    IMG_SIZE = config["img_size"]

    print("✅ Configuration loaded successfully")
    print(f"✅ Image Size: {IMG_SIZE}")
    print(f"✅ Classes: {classes}")

except Exception as e:

    print("❌ Failed to load configuration files")
    traceback.print_exc()

    raise Exception("Configuration loading failed")

# =========================================================
# DOWNLOAD & LOAD MODEL
# =========================================================

print("=" * 60)
print("LOADING TENSORFLOW MODEL...")
print("=" * 60)

MODEL_LOADED = False
model = None

try:

    if not os.path.exists(MODEL_PATH):

        print("Downloading model from Google Cloud Storage...")

        client = storage.Client()
        bucket = client.bucket("qhda-models")
        blob = bucket.blob("xcept_quant_model.keras")

        blob.download_to_filename(MODEL_PATH)

        print("✅ Model downloaded successfully.")

    else:

        print("✅ Model already exists.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "QuantumLayer": QuantumLayer
        }
    )

    MODEL_LOADED = True

    print("✅ MODEL LOADED SUCCESSFULLY")

except Exception:

    print("❌ MODEL LOADING FAILED")
    traceback.print_exc()
    raise

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Brain Tumor Quantum API",
    description="Brain Tumor Classification using Quantum Deep Learning",
    version="1.0.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# VALID FILE EXTENSIONS
# =========================================================

ALLOWED_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png"
]

# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def preprocess(image_bytes):

    try:

        # bytes → numpy
        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        # decode image
        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:
            raise Exception("OpenCV failed to decode image")

        # BGR → RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # resize
        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        # normalize
        img = img.astype(np.float32) / 255.0

        # batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        return img

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail="Image preprocessing failed"
        )

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/", tags=["Health"])
def home():

    return {
        "message": "Brain Tumor Quantum API Running",
        "model_loaded": MODEL_LOADED,
        "classes": classes
    }

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", tags=["Health"])
def health():

    return {
        "status": "healthy" if MODEL_LOADED else "model_failed",
        "model_loaded": MODEL_LOADED,
        "total_classes": len(classes),
        "classes": classes
    }

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict", tags=["Prediction"])
async def predict(
    image: UploadFile = File(...)
):

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Image file missing"
        )

    if image.filename == "":

        raise HTTPException(
            status_code=400,
            detail="Empty filename"
        )

    # =====================================================
    # CHECK EXTENSION
    # =====================================================

    ext = image.filename.split(".")[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=f"Allowed formats: {ALLOWED_EXTENSIONS}"
        )

    # =====================================================
    # READ FILE
    # =====================================================

    try:

        contents = await image.read()

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail="Failed to read uploaded image"
        )

    # =====================================================
    # PREPROCESS IMAGE
    # =====================================================

    img = preprocess(contents)

    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        preds = model.predict(
            img,
            verbose=0
        )[0]

        pred_idx = int(np.argmax(preds))

        prediction = classes[pred_idx]

        confidence = float(preds[pred_idx])

        all_probs = {
            cls: float(prob)
            for cls, prob in zip(classes, preds)
        }

        print("=" * 60)
        print("PREDICTION SUCCESS")
        print(f"Prediction : {prediction}")
        print(f"Confidence : {confidence:.4f}")
        print("=" * 60)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "all_probabilities": all_probs
        }

    except Exception:

        print("❌ Prediction failed")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

# =========================================================
# STARTUP LOGS
# =========================================================

print("=" * 60)
print("🚀 API STARTED")
print(f"✅ MODEL LOADED : {MODEL_LOADED}")
print(f"✅ CLASSES      : {classes}")
print("=" * 60)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )