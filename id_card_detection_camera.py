# id_card_detection_camera.py
# Camera script (TF2 SavedModel) — saving disabled by default unless --save is passed

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
import time
from pathlib import Path

# ----------------------------
# Lightweight helpers
# ----------------------------
def parse_labelmap(labelmap_path):
    classes = {}
    current_id = None
    current_name = None
    if not os.path.exists(labelmap_path):
        return {1: {'id': 1, 'name': 'object'}}
    with open(labelmap_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                try:
                    current_id = int(line.split(':', 1)[1].strip())
                except Exception:
                    current_id = None
            elif line.startswith('name:') or line.startswith('display_name:'):
                val = line.split(':', 1)[1].strip().strip('"').strip("'")
                current_name = val
            elif line.startswith('item'):
                current_id = None
                current_name = None
            elif line == '}':
                if current_id is not None and current_name is not None:
                    classes[current_id] = {'id': current_id, 'name': current_name}
                current_id = None
                current_name = None
    if not classes:
        classes = {1: {'id': 1, 'name': 'object'}}
    return classes

def draw_boxes(image, boxes, classes, scores, category_index, min_score, line_thickness=3):
    h, w = image.shape[:2]
    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else None
    for i in range(len(boxes)):
        score = float(scores[i]) if scores is not None else 1.0
        if score < min_score:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cls_id = int(classes[i]) if classes is not None and i < len(classes) else 1
        cls_name = category_index.get(cls_id, {'name': 'object'})['name']
        color = (0, 255, 0) if (best_idx is not None and i == best_idx) else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{cls_name}: {int(score*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# ----------------------------
# CLI args
# ----------------------------
parser = argparse.ArgumentParser(description='ID Card Detection (camera, TF2 SavedModel)')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
parser.add_argument('--min_score', type=float, default=0.50, help='Minimum score threshold (0-1)')
parser.add_argument('--ocr', action='store_true',
                    help='Enable OCR on selected snapshots (1-9 keys)')
parser.add_argument('--auto_save', action='store_true',
                    help='Auto-save detections periodically (requires --save)')
parser.add_argument('--auto_interval', type=float, default=1.0,
                    help='Auto-save interval in seconds')
parser.add_argument('--save', action='store_true',
                    help='Enable saving crops and OCR results to disk (disabled by default)')
args = parser.parse_args()

# ----------------------------
# Paths & models
# ----------------------------
CWD_PATH = os.getcwd()
MODEL_NAME = 'model'
PATH_TO_SAVED_MODEL = os.path.join(CWD_PATH, MODEL_NAME, 'saved_model')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')
category_index = parse_labelmap(PATH_TO_LABELS)

# Load TF2 SavedModel (serving_default)
if not os.path.exists(PATH_TO_SAVED_MODEL):
    print("Warning: SavedModel not found at", PATH_TO_SAVED_MODEL)
    detect_module = None
    infer = None
else:
    detect_module = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    infer = detect_module.signatures.get('serving_default', None)
    if infer is None:
        infer = next(iter(detect_module.signatures.values()))
    _, sig_kwargs = infer.structured_input_signature
    input_keys = list(sig_kwargs.keys())
    if not input_keys:
        raise RuntimeError('SavedModel signature has no inputs')
    input_key = input_keys[0]

# Initialize OCR reader once if requested
reader = None
if args.ocr:
    try:
        import easyocr
        reader = easyocr.Reader(['tr', 'en'], gpu=False)
    except Exception as e:
        print("EasyOCR initialization failed:", e)
        reader = None

# ----------------------------
# Camera helpers
# ----------------------------
def open_camera(index: int):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam

def close_camera(cam):
    try:
        cam.release()
    except Exception:
        pass

video = open_camera(args.camera)

is_paused = False
last_frame = None
ocr_text_lines = []  # last OCR output for the panel
min_score_thresh = args.min_score

# Save helpers
def ts():
    return time.strftime("%Y%m%d_%H%M%S")

def save_crop_and_ocr(crop_img, ocr_lines, idx_label):
    out = Path.cwd()
    t = ts()
    img_path = out / f"crop_{t}_{idx_label}.png"
    txt_path = out / f"ocr_{t}_{idx_label}.txt"
    try:
        cv2.imwrite(str(img_path), crop_img)
    except Exception as e:
        print("[SAVE] image write failed:", e)
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            if isinstance(ocr_lines, list):
                f.write("\n".join(ocr_lines))
            else:
                f.write(str(ocr_lines))
    except Exception as e:
        print("[SAVE] text write failed:", e)
    print(f"[SAVED] {img_path}  {txt_path}")

_last_auto_save_time = 0.0

WINDOW_NAME = 'ID CARD DETECTOR'

# ----------------------------
# Main loop
# ----------------------------
try:
    while True:
        # Read frame (or reuse last when paused)
        if not is_paused:
            ret, frame = video.read()
            if not ret:
                time.sleep(0.02)
                continue
            last_frame = frame.copy()
        else:
            if last_frame is None:
                ret, frame = video.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                last_frame = frame.copy()
            frame = last_frame.copy()

        # Run detection (TF2 ONLY)
        if infer is not None:
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(frame, axis=0), dtype=tf.uint8
            )
            outputs = infer(**{input_key: input_tensor})
            boxes = outputs.get('detection_boxes')[0].numpy()
            scores = outputs.get('detection_scores')[0].numpy()
            classes = outputs.get('detection_classes', None)
            if classes is not None:
                classes = classes[0].numpy().astype(np.int32)
            else:
                classes = np.ones((boxes.shape[0],), dtype=np.int32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)

        # Draw boxes on frame (mutates frame)
        draw_boxes(frame, boxes, classes, scores,
                   category_index, min_score_thresh, line_thickness=4)

        # Build snapshot panel (top 9 detections)
        h, w = frame.shape[:2]
        dets = []
        for i in range(len(boxes)):
            sc = float(scores[i]) if scores is not None else 1.0
            if sc < min_score_thresh:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
            x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            dets.append((sc, crop))
        dets.sort(key=lambda t: t[0], reverse=True)
        dets = dets[:9]

        # Create panel image with thumbnails and OCR text
        panel = None
        thumb_w, thumb_h = 200, 120
        margin = 8
        if dets:
            tiles = []
            for idx, (sc, crop) in enumerate(dets, start=1):
                if crop.size == 0:
                    continue
                timg = cv2.resize(crop, (thumb_w, thumb_h))
                header = np.full((24, thumb_w, 3), 230, dtype=np.uint8)
                label = f"{idx}: {int(sc*100)}%"
                cv2.putText(header, label, (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                tile = np.vstack([header, timg])
                tiles.append(tile)
            if tiles:
                blocks = []
                for i, tile in enumerate(tiles):
                    if i > 0:
                        blocks.append(np.full((margin, thumb_w, 3), 255, dtype=np.uint8))
                    blocks.append(tile)
                panel = np.vstack(blocks)
        else:
            panel = np.full((thumb_h+24, thumb_w, 3), 245, dtype=np.uint8)
            cv2.putText(panel, 'No detections', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # OCR text area
        if ocr_text_lines:
            text_h = 28 + 22 * len(ocr_text_lines)
            text_canvas = np.full((text_h, panel.shape[1], 3), 250, dtype=np.uint8)
            cv2.putText(text_canvas, 'OCR Result:', (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            y = 44
            for line in ocr_text_lines[:10]:
                cv2.putText(text_canvas, str(line), (6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
                y += 22
            panel = np.vstack([
                panel,
                np.full((margin, panel.shape[1], 3), 255, dtype=np.uint8),
                text_canvas
            ])

        # Compose HUD
        if panel is not None:
            scale = frame.shape[0] / panel.shape[0]
            new_w = max(1, int(panel.shape[1] * scale))
            panel = cv2.resize(panel, (new_w, frame.shape[0]))
            composed = np.hstack([frame, panel])
            hud = composed
        else:
            hud = frame

        cv2.putText(hud,
                    'q: quit  p: pause/resume  s: stop camera  b: start camera  1-9: OCR',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(hud,
                    'q: quit  p: pause/resume  s: stop camera  b: start camera  1-9: OCR',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if is_paused:
            cv2.putText(hud, 'PAUSED', (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(WINDOW_NAME, hud)

        # Read key
        key = cv2.waitKey(1) & 0xFF

        # Manual OCR + (save if --save)
        if args.ocr and dets and key in [ord(str(d)) for d in range(1, 10)]:
            try:
                sel = int(chr(key)) - 1
                if sel < 0 or sel >= len(dets):
                    print(f"[WARN] No snapshot for index {sel+1}")
                else:
                    score, crop = dets[sel]
                    if reader is not None:
                        try:
                            result = reader.readtext(crop, detail=0)
                        except Exception as e:
                            print("[OCR] readtext failed:", e)
                            result = []
                    else:
                        result = []

                    print(f"\n[OCR {sel+1}]")
                    for line in result:
                        print(line)
                    ocr_text_lines = list(result) if isinstance(result, list) else [str(result)]

                    # Save outputs only if user passed --save
                    if args.save:
                        save_crop_and_ocr(crop, ocr_text_lines, sel+1)
            except Exception as e:
                print("OCR error:", e)
                ocr_text_lines = [f"OCR error: {e}"]

        # AUTO_SAVE: save all current detections periodically
        now = time.time()
        if args.auto_save and args.save and dets and (
            now - _last_auto_save_time >= max(0.1, args.auto_interval)
        ):
            for i, (sc, crop) in enumerate(dets):
                ocr_lines = []
                if args.ocr and reader is not None:
                    try:
                        ocr_lines = reader.readtext(crop, detail=0)
                    except Exception as e:
                        ocr_lines = [f"OCR error: {e}"]
                save_crop_and_ocr(crop, ocr_lines, i+1)
            _last_auto_save_time = now

        # Controls
        if key == ord('p'):
            is_paused = not is_paused
        if key == ord('s'):
            if video is not None:
                close_camera(video)
                video = None
                is_paused = True
        if key == ord('b'):
            if video is None:
                video = open_camera(args.camera)
                is_paused = False
        if key == ord('q'):
            break

finally:
    try:
        if video is not None:
            video.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
