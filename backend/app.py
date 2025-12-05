# app.py
import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
# Model path (default: saved_model.h5 in same folder)
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "saved_model.h5")
IMAGE_SIZE = tuple(int(x) for x in (os.environ.get("IMAGE_SIZE") or "128,128").split(","))
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
LABELS = os.environ.get("LABELS")
if LABELS:
    LABELS = [s.strip() for s in LABELS.split(",")]
else:
    LABELS = ["cats", "dogs"]  # default mapping; confirm train class order

# ---------------- MODEL BUILD & LOAD ----------------
def build_model(input_shape=(128,128,3)):
    # Use pooling='max' so base outputs a (None, 2048) vector to match saved .h5
    base = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        pooling='max',
        input_shape=input_shape
    )
    base.trainable = False

    m = models.Sequential(name="reconstructed_xception_top")
    m.add(base)
    # Top layers: names must match those in the saved .h5
    m.add(layers.Dense(128, activation='relu', name="dense"))
    m.add(layers.Dense(128, activation='relu', name="dense_1"))
    m.add(layers.Dense(32, activation='relu', name="dense_2"))
    m.add(layers.Dense(1, activation='sigmoid', name="dense_3"))
    return m

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Put saved_model.h5 in the same folder or set MODEL_PATH env var.")

print("[INFO] Building model architecture...")
model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
print("[INFO] Loading weights (by_name=True, skip_mismatch=True)...")
model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
print(f"[INFO] Rebuilt model and loaded weights from {MODEL_PATH}")

# Use Xception preprocessing (must match training preprocessing)
preprocess_fn = tf.keras.applications.xception.preprocess_input

# ---------------- HELPERS ----------------
def read_imagefile(file_bytes, target_size=IMAGE_SIZE):
    """Read raw bytes -> preprocessed batch tensor ready for model.predict"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")  # uint8 -> float32
    # Apply Xception preprocessing (scales / centers as needed)
    arr = preprocess_fn(arr)
    # Add batch dim
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr

def interpret_prediction(preds):
    """Convert model output to label + probability"""
    preds = np.array(preds)
    if preds.ndim >= 2 and preds.shape[-1] == 1:
        prob = float(preds[0][0])
        idx = 1 if prob >= THRESHOLD else 0
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        return {"label": label, "label_index": idx, "probability": prob, "raw": preds.tolist()}
    if preds.ndim >= 2 and preds.shape[-1] > 1:
        idx = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds, axis=1)[0])
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
        return {"label": label, "label_index": idx, "probability": prob, "raw": preds.tolist()}
    return {"raw": preds.tolist()}

# ---------------- ROUTES ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "labels": LABELS
    })

@app.route("/", methods=["GET"])
def index():
    return """
    <h3>Simple CNN model server</h3>
    <p>POST an image file to <code>/predict</code> (form key: <code>file</code>).</p>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" />
      <button type="submit">Upload & Predict</button>
    </form>
    <p>Health: <a href="/health">/health</a></p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file part in the request (use key 'file')"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    try:
        img_bytes = f.read()
        x = read_imagefile(img_bytes, target_size=IMAGE_SIZE)
        preds = model.predict(x)
        result = interpret_prediction(preds)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    import base64
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "send JSON with key 'image' containing base64 string"}), 400
    try:
        b64 = data["image"]
        if "," in b64:  # strip data URI prefix
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        x = read_imagefile(img_bytes, target_size=IMAGE_SIZE)
        preds = model.predict(x)
        return jsonify(interpret_prediction(preds))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug True for dev only
    app.run(host="0.0.0.0", port=port, debug=True)
