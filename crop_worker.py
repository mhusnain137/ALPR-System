"""
crop_worker.py
--------------
Runs INSIDE the 'vggface' conda environment (Ultralytics YOLO lives there).

Given the path to a source image, this script:
    1. Loads the YOLO license-plate-detection model
    2. Runs detection on the image
    3. Draws boxes + confidence on a copy and saves it (annotated image)
    4. Crops every detected plate and saves each crop to disk
    5. Prints ONE JSON object to stdout describing what it did

app.py (running elsewhere) calls this via subprocess, e.g.:
    conda run -n vggface python crop_worker.py <image_path> \
        --crops-dir static/output/crops --annotated-path static/output/annotated/xyz.jpg
and parses the JSON from stdout. Only JSON goes to stdout -- everything
else (progress/errors during normal operation) should go to stderr, so it
never corrupts the JSON the caller is trying to parse.

Usage:
    python crop_worker.py <image_path> [--crops-dir DIR] [--annotated-path PATH] [--conf 0.4]
"""

import os
import sys
import json
import uuid
import argparse

import cv2

# ---------------------------------------------------------------------------
# PyTorch 2.6 flipped the default of torch.load to weights_only=True, which
# breaks loading Ultralytics YOLO checkpoints (they pickle model classes,
# not just tensors). Patch torch.load BEFORE importing ultralytics so this
# takes effect for every load it does internally.
# ---------------------------------------------------------------------------
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO  # noqa: E402  (import after the torch.load patch)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "license_plate_detector.pt")
DEFAULT_CONF = 0.4

_model = None


def get_model():
    """Loads the YOLO model once per process (subprocess is short-lived anyway,
    but this keeps things tidy if the function is ever reused)."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_and_crop(image_path, crops_dir, annotated_path, conf_threshold=DEFAULT_CONF):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image at: {image_path}")

    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotated_path) or ".", exist_ok=True)

    model = get_model()
    results = model(frame, imgsz=1280, conf=conf_threshold)[0]

    h, w = frame.shape[:2]

    crops = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Keep the original (unpadded) box for drawing on the annotated frame.
        draw_x1, draw_y1, draw_x2, draw_y2 = x1, y1, x2, y2

        # Pad the crop by 8% of the box's largest dimension, capped at 15px
        # so small plate boxes don't pull in surrounding text/logos (e.g. "GWM").
        pad = min(int(max(x2 - x1, y2 - y1) * 0.08), 15)

        pad_x1 = max(0, x1 - pad)
        pad_y1 = max(0, y1 - pad)
        pad_x2 = min(w, x2 + pad)
        pad_y2 = min(h, y2 + pad)

        # Crop BEFORE drawing the box, so the saved crop stays clean.
        crop_img = frame[pad_y1:pad_y2, pad_x1:pad_x2]
        if crop_img.size == 0:
            continue

        # Upscale the crop 3x with cubic interpolation for readability.
        crop_img = cv2.resize(
            crop_img,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )

        crop_name = f"crop_{uuid.uuid4().hex}.jpg"
        crop_path = os.path.join(crops_dir, crop_name)
        cv2.imwrite(crop_path, crop_img)

        crops.append({
            "crop_path": crop_path,
            "conf": round(conf, 4),
            "bbox": [draw_x1, draw_y1, draw_x2, draw_y2],
            "padded_bbox": [pad_x1, pad_y1, pad_x2, pad_y2],
        })

        # Draw box + confidence on the full frame for the annotated copy.
        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{conf:.2f}", (draw_x1, max(0, draw_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    cv2.imwrite(annotated_path, frame)

    return {
        "source_image": image_path,
        "annotated_path": annotated_path,
        "plate_count": len(crops),
        "crops": crops,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect + crop license plates with YOLO.")
    parser.add_argument("image_path", help="Path to the source image to run detection on.")
    parser.add_argument("--crops-dir", default="output/crops", help="Where to save cropped plates.")
    parser.add_argument("--annotated-path", default="output/annotated.jpg", help="Where to save the annotated image.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Detection confidence threshold.")
    args = parser.parse_args()

    try:
        result = detect_and_crop(args.image_path, args.crops_dir, args.annotated_path, args.conf)
        result["ok"] = True
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    # ONLY JSON on stdout -- this is what app.py parses via subprocess.
    print(json.dumps(result))

    if not result.get("ok", False):
        sys.exit(1)


if __name__ == "__main__":
    main()