import os
import io
import base64
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# ML libraries
import tensorflow as tf  # TF2 SavedModel
import easyocr

# -----------------------
# CLI / Config
# -----------------------
parser = argparse.ArgumentParser(description="ID Card Detector - Flask UI")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--min_score", type=float, default=0.50)
parser.add_argument("--easyocr_cache", default="easyocr_model_cache")
args = parser.parse_args()

# -----------------------
# Globals / Paths
# -----------------------
APP_ROOT = Path.cwd()
MODEL_DIR = APP_ROOT / "model" / "saved_model"
LABELMAP_PATH = APP_ROOT / "data" / "labelmap.pbtxt"
EASYOCR_CACHE = APP_ROOT / args.easyocr_cache
EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

# -----------------------
# Labelmap
# -----------------------
def parse_labelmap(labelmap_path):
    classes = {}
    if not labelmap_path.exists():
        return {1: {"id": 1, "name": "object"}}

    current_id = None
    current_name = None
    with open(labelmap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id:"):
                current_id = int(line.split(":")[1].strip())
            elif line.startswith("name:") or line.startswith("display_name:"):
                current_name = line.split(":", 1)[1].strip().replace('"', "").replace("'", "")
            elif line == "}":
                if current_id and current_name:
                    classes[current_id] = {"id": current_id, "name": current_name}
                current_id = None
                current_name = None
    return classes if classes else {1: {"id": 1, "name": "object"}}

CATEGORY_INDEX = parse_labelmap(LABELMAP_PATH)

# -----------------------
# Draw boxes
# -----------------------
def draw_boxes_on_image(image_bgr, boxes, classes, scores, min_score=0.5):
    h, w = image_bgr.shape[:2]
    best_idx = int(np.argmax(scores)) if len(scores) > 0 else None

    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc < min_score:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        color = (0, 255, 0) if i == best_idx else (0, 0, 255)
        cls_id = int(classes[i])
        cls_name = CATEGORY_INDEX.get(cls_id, {"name": "object"})["name"]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_bgr,
            f"{cls_name}:{int(sc * 100)}%",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image_bgr

# -----------------------
# Load TensorFlow Model ONLY
# -----------------------
tf_model = None
tf_infer = None

print("Loading TF SavedModel:", MODEL_DIR)
tf_model = tf.saved_model.load(str(MODEL_DIR))
tf_infer = tf_model.signatures.get("serving_default")

_, sig_kwargs = tf_infer.structured_input_signature
tf_input_key = list(sig_kwargs.keys())[0]

# -----------------------
# EasyOCR
# -----------------------
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(
            ['en', 'tr'], 
            gpu=False, 
            model_storage_directory=str(EASYOCR_CACHE)
        )
    return reader

# -----------------------
# Main inference (TF ONLY)
# -----------------------
def run_inference(image_rgb, do_ocr=True, min_score=args.min_score):

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    # TensorFlow SavedModel detection
    tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)
    out = tf_infer(**{tf_input_key: tensor})

    boxes = out["detection_boxes"][0].numpy()
    scores = out["detection_scores"][0].numpy()
    classes = out["detection_classes"][0].numpy().astype(int)

    # Draw HUD
    hud_bgr = draw_boxes_on_image(image_bgr.copy(), boxes, classes, scores, min_score)
    hud_rgb = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB)

    results = []

    # Extract crops + OCR
    for i in range(len(boxes)):
        if scores[i] < min_score:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        crop = image_bgr[y1:y2, x1:x2]

        ocr_lines = []
        if do_ocr and crop.size > 0:
            r = get_reader()
            text = r.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=0)
            ocr_lines = text

        results.append({
            "crop": crop,
            "bbox": [x1, y1, x2, y2],
            "score": float(scores[i]),
            "ocr": ocr_lines
        })

    return hud_rgb, results

# -----------------------
# Flask API
# -----------------------
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/detect", methods=["POST"])
def detect():
    img = request.files["file"].read()

    pil = Image.open(io.BytesIO(img)).convert("RGB")
    img_rgb = np.array(pil)

    hud_rgb, outputs = run_inference(img_rgb)

    # Encode HUD
    _, buf = cv2.imencode(".png", cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR))
    hud_b64 = base64.b64encode(buf).decode()

    # Pack results
    res_list = []
    for out in outputs:
        _, cbuf = cv2.imencode(".png", out["crop"])
        crop_b64 = base64.b64encode(cbuf).decode()

        res_list.append({
            "score": out["score"],
            "bbox": out["bbox"],
            "crop": f"data:image/png;base64,{crop_b64}",
            "ocr": out["ocr"]
        })

    return jsonify({
        "hud": f"data:image/png;base64,{hud_b64}",
        "outputs": res_list
    })

if __name__ == "__main__":
    print("Running Flask at http://127.0.0.1:5000")
    app.run(host=args.host, port=args.port, debug=False)
