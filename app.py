import os

# =========================================================
# TENSORFLOW SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import base64
import traceback
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
    CAM_LAYER_NAME = config.get("cam_layer", "block14_sepconv2_act")

    print("✅ Configuration loaded successfully")
    print(f"✅ Image Size: {IMG_SIZE}")
    print(f"✅ Classes: {classes}")
    print(f"✅ Grad-CAM target layer: {CAM_LAYER_NAME}")

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
# GRAD-CAM++ SETUP
# =========================================================
#
# The Xception backbone may either be:
#   (a) "flat"   - its layers were merged directly into the outer
#                  functional model (e.g. built with input_tensor=...)
#   (b) "nested" - the backbone exists as its own sub-model, called
#                  once as a single layer inside the outer model
#                  (e.g. features = Xception(...)(inputs))
#
# We auto-detect which case we're in so this keeps working even if
# the model is retrained/rebuilt slightly differently later.
# =========================================================

print("=" * 60)
print("SETTING UP GRAD-CAM...")
print("=" * 60)

GRADCAM_READY = False
gradcam_feature_model = None
gradcam_head_layers = []


def _locate_backbone_and_head(full_model, cam_layer_name):
    """
    Returns (feature_model, head_layers).

    feature_model: tf.keras.Model mapping the backbone's own input ->
                    [cam_layer_output, backbone_output]
    head_layers:   ordered list of layers to apply AFTER the backbone
                    output to reach the final model output (empty if
                    the cam layer already lives in the outer model).
    """

    # --- Case (a): flat model, cam layer is directly on full_model ---
    try:
        target_layer = full_model.get_layer(cam_layer_name)
        feature_model = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=[target_layer.output, full_model.output]
        )
        return feature_model, []
    except ValueError:
        pass

    # --- Case (b): nested backbone sub-model ---
    backbone = None
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer(cam_layer_name)
                backbone = layer
                break
            except ValueError:
                continue

    if backbone is None:
        raise ValueError(
            f"Could not locate layer '{cam_layer_name}' in the model, "
            f"either directly or inside a nested sub-model."
        )

    cam_output = backbone.get_layer(cam_layer_name).output

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[cam_output, backbone.output]
    )

    # Everything in the outer model that comes after the backbone
    # (e.g. GlobalAveragePooling2D -> Dense -> QuantumLayer -> Dense).
    # Assumes a simple sequential head after the backbone, which matches
    # this project's architecture.
    head_layers = []
    started = False

    for layer in full_model.layers:
        if layer is backbone:
            started = True
            continue
        if started and not isinstance(layer, tf.keras.layers.InputLayer):
            head_layers.append(layer)

    return feature_model, head_layers


try:

    gradcam_feature_model, gradcam_head_layers = _locate_backbone_and_head(
        model,
        CAM_LAYER_NAME
    )

    GRADCAM_READY = True

    print(f"✅ Grad-CAM ready (head layers after backbone: "
          f"{[l.name for l in gradcam_head_layers]})")

except Exception:

    print("⚠️  Grad-CAM setup failed — /predict/gradcam will be unavailable")
    traceback.print_exc()

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

def preprocess(image_bytes, return_raw=False):

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

        # raw uint8 RGB copy (used later for heatmap overlay / GIF)
        raw_resized = img.copy()

        # normalize
        img_norm = img.astype(np.float32) / 255.0

        # batch dimension
        img_norm = np.expand_dims(
            img_norm,
            axis=0
        )

        if return_raw:
            return img_norm, raw_resized

        return img_norm

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail="Image preprocessing failed"
        )

# =========================================================
# SCORE-CAM CORE
# =========================================================
#
# We initially tried Grad-CAM / Grad-CAM++, both of which need a gradient
# computed THROUGH the PennyLane QuantumLayer. In production this crashed:
# PennyLane's TF-autograph-generated code hits a broken branch the moment
# the quantum layer's forward pass runs inside an active tf.GradientTape —
# even before .gradient() is called. This is an internal PennyLane/autograph
# incompatibility, not something fixable from app.py.
#
# Score-CAM (Wang et al., 2020) sidesteps this entirely: it needs NO
# gradients, only plain forward passes — the exact call pattern /predict
# already uses successfully. For each of the top-K most active channels in
# the conv feature map, we mask the original image with that channel's
# (upsampled, normalized) activation map, run a normal forward pass, and use
# the resulting class confidence as that channel's importance weight.
# =========================================================

SCORECAM_TOP_K = 64  # number of channels to use; higher = slower but finer detail


def score_cam(feature_model, head_layers, raw_image_uint8, class_idx, top_k=SCORECAM_TOP_K):
    """
    Computes a Score-CAM heatmap (values in [0, 1], shape IMG_SIZE x IMG_SIZE)
    for the given class index, using only forward passes (no gradients).
    """

    img_norm = raw_image_uint8.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    conv_out, backbone_out = feature_model.predict(img_batch, verbose=0)
    conv_out = conv_out[0]  # (h, w, channels)

    # pick the most active channels to keep the number of extra forward
    # passes manageable (this is the standard "Faster Score-CAM" trick)
    channel_scores = np.sum(np.abs(conv_out), axis=(0, 1))
    k = min(top_k, channel_scores.shape[0])
    top_indices = np.argsort(channel_scores)[-k:]

    masks = []
    for idx in top_indices:
        chan = conv_out[:, :, idx]
        chan = cv2.resize(chan, (IMG_SIZE, IMG_SIZE))
        c_min, c_max = chan.min(), chan.max()
        if (c_max - c_min) < 1e-8:
            norm_mask = np.zeros_like(chan, dtype=np.float32)
        else:
            norm_mask = ((chan - c_min) / (c_max - c_min)).astype(np.float32)
        masks.append(norm_mask)

    masks = np.stack(masks, axis=0)  # (k, IMG_SIZE, IMG_SIZE)

    # mask the ORIGINAL image (not the backbone features) with each channel's
    # activation map, then batch all of them through one forward pass.
    # A fully-blacked-out image (all-zero mask) is appended as a baseline so
    # we can measure how much each channel's mask *increases* confidence
    # above doing nothing, instead of using raw (noisier) confidence values.
    baseline_mask = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    all_masks = np.concatenate([masks, baseline_mask], axis=0)  # (k+1, H, W)

    masked_batch = img_norm[None, :, :, :] * all_masks[:, :, :, None]  # (k+1, H, W, 3)

    _, masked_backbone_out = feature_model.predict(masked_batch, verbose=0)

    if head_layers:
        x = tf.convert_to_tensor(masked_backbone_out, dtype=tf.float32)
        for layer in head_layers:
            x = layer(x, training=False)
        masked_preds = x.numpy()
    else:
        masked_preds = masked_backbone_out

    raw_scores = masked_preds[:, class_idx]  # (k+1,)
    baseline_score = raw_scores[-1]
    channel_deltas = raw_scores[:-1] - baseline_score  # (k,) — gain over doing nothing

    # softmax-normalize so channels that actually help the class stand out
    # sharply from ones that barely matter (per the original Score-CAM paper)
    exp_deltas = np.exp(channel_deltas - np.max(channel_deltas))
    weights = exp_deltas / np.sum(exp_deltas)

    cam = np.tensordot(weights, masks, axes=([0], [0]))  # (IMG_SIZE, IMG_SIZE)
    cam = np.maximum(cam, 0)

    # sharpen: drop the bottom ~15% of activation so low-confidence "haze"
    # covering the whole brain doesn't survive into the final heatmap
    if cam.max() > 0:
        threshold = np.percentile(cam, 15)
        cam = np.where(cam >= threshold, cam - threshold, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def colorize_heatmap(heatmap, size):
    """Resizes a 0-1 heatmap to `size` and applies a JET colormap. Returns RGB uint8."""

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (size, size))
    heatmap_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def blend(original_rgb, heatmap_rgb, alpha):
    return cv2.addWeighted(original_rgb, 1 - alpha, heatmap_rgb, alpha, 0)


def encode_png_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise Exception("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def build_gradcam_gif_base64(original_rgb, heatmap_rgb, steps=12, hold_frames=6, duration_ms=80):
    """
    Builds an animated GIF fading from the original image to the Grad-CAM
    overlay and back, so the highlighted region is easy to spot.
    """

    frames = []

    # fade in: original -> full overlay
    for i in range(steps + 1):
        a = i / steps
        frames.append(Image.fromarray(blend(original_rgb, heatmap_rgb, a)))

    # hold on the overlay
    frames += [frames[-1]] * hold_frames

    # fade out: overlay -> original
    for i in range(steps, -1, -1):
        a = i / steps
        frames.append(Image.fromarray(blend(original_rgb, heatmap_rgb, a)))

    # hold on the original
    frames += [frames[-1]] * hold_frames

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )

    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/", tags=["Health"])
def home():

    return {
        "message": "Brain Tumor Quantum API Running",
        "model_loaded": MODEL_LOADED,
        "gradcam_available": GRADCAM_READY,
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
        "gradcam_available": GRADCAM_READY,
        "total_classes": len(classes),
        "classes": classes
    }

# =========================================================
# SHARED VALIDATION / READ HELPERS
# =========================================================

async def _read_and_validate_image(image: UploadFile):

    if image is None:
        raise HTTPException(status_code=400, detail="Image file missing")

    if image.filename == "":
        raise HTTPException(status_code=400, detail="Empty filename")

    ext = image.filename.split(".")[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Allowed formats: {ALLOWED_EXTENSIONS}"
        )

    try:
        contents = await image.read()
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Failed to read uploaded image")

    return contents

# =========================================================
# PREDICTION ENDPOINT (unchanged)
# =========================================================

@app.post("/predict", tags=["Prediction"])
async def predict(
    image: UploadFile = File(...)
):

    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await _read_and_validate_image(image)

    img = preprocess(contents)

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
# PREDICTION + GRAD-CAM++ ENDPOINT
# =========================================================

@app.post("/predict/gradcam", tags=["Prediction"])
async def predict_with_gradcam(
    image: UploadFile = File(...),
    target_class: str = Query(
        default=None,
        description="Optional class name to explain instead of the predicted class"
    ),
    alpha: float = Query(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Heatmap overlay opacity for the static heatmap image"
    )
):

    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not GRADCAM_READY:
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM is not available for this model (setup failed at startup)"
        )

    contents = await _read_and_validate_image(image)

    img, raw_resized = preprocess(contents, return_raw=True)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        preds = model.predict(img, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        prediction = classes[pred_idx]
        confidence = float(preds[pred_idx])

        all_probs = {
            cls: float(prob)
            for cls, prob in zip(classes, preds)
        }

    except Exception:

        print("❌ Prediction failed")
        traceback.print_exc()

        raise HTTPException(status_code=500, detail="Prediction failed")

    # -----------------------------------------------------
    # RESOLVE CLASS TO EXPLAIN
    # -----------------------------------------------------

    if target_class is not None:
        if target_class not in classes:
            raise HTTPException(
                status_code=400,
                detail=f"target_class must be one of {classes}"
            )
        explain_idx = classes.index(target_class)
    else:
        explain_idx = pred_idx

    # -----------------------------------------------------
    # SCORE-CAM
    # -----------------------------------------------------

    try:

        heatmap = score_cam(
            gradcam_feature_model,
            gradcam_head_layers,
            raw_resized,
            explain_idx
        )

        heatmap_rgb = colorize_heatmap(heatmap, IMG_SIZE)

        overlay_rgb = blend(raw_resized, heatmap_rgb, alpha)

        heatmap_b64 = encode_png_base64(heatmap_rgb)
        overlay_b64 = encode_png_base64(overlay_rgb)
        gif_b64 = build_gradcam_gif_base64(raw_resized, heatmap_rgb)

        print("=" * 60)
        print("SCORE-CAM SUCCESS")
        print(f"Prediction    : {prediction}")
        print(f"Explaining    : {classes[explain_idx]}")
        print("=" * 60)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "explained_class": classes[explain_idx],
            "heatmap_png_base64": f"data:image/png;base64,{heatmap_b64}",
            "overlay_png_base64": f"data:image/png;base64,{overlay_b64}",
            "gradcam_gif_base64": f"data:image/gif;base64,{gif_b64}"
        }

    except Exception:

        print("❌ Score-CAM generation failed")
        traceback.print_exc()

        raise HTTPException(status_code=500, detail="Score-CAM generation failed")

# =========================================================
# STARTUP LOGS
# =========================================================

print("=" * 60)
print("🚀 API STARTED")
print(f"✅ MODEL LOADED   : {MODEL_LOADED}")
print(f"✅ GRAD-CAM       : {GRADCAM_READY}")
print(f"✅ CLASSES        : {classes}")
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