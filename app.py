"""
Plate Reader -- Flask app (single-environment architecture)

Detection (YOLO) and OCR (PaddleOCR) now BOTH run inside this same process
(the 'alpr' conda env), so there's no more subprocess/env-switching --
crop_worker.py and ocr_worker.py are imported directly as modules and their
functions are called in-process.

    - crop_worker.detect_and_crop()  -> YOLO detects + crops plate(s),
                                         saves an annotated copy with boxes drawn
    - ocr_worker.read_plate()        -> PaddleOCR reads each crop

Two ways to feed it an image:
  1. Upload a photo from your PC
  2. Paste an image link (e.g. copied from Chrome)

Pipeline per image:
  1. Save the source image you gave it
  2. crop_worker.detect_and_crop(): YOLO detects + crops plate(s),
     saves an annotated copy with boxes drawn
  3. ocr_worker.read_plate(): PaddleOCR reads each crop
  4. Every result gets appended to output/plate_texts.csv
  5. Results + annotated image are rendered in the browser

Run:
    conda activate alpr
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import csv
import uuid
import requests
import cv2
from flask import Flask, render_template, request, url_for
from datetime import datetime

# Direct imports -- both workers now live in the same env as this app.
import crop_worker
import ocr_worker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")          # original source images
CROPS_FOLDER = os.path.join(BASE_DIR, "static", "output", "crops")    # cropped plates (served to browser)
ANNOTATED_FOLDER = os.path.join(BASE_DIR, "static", "output", "annotated")
RESULTS_CSV = os.path.join(BASE_DIR, "output", "plate_texts.csv")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

CSV_FIELDNAMES = ["image_name", "detected_text", "confidence", "model", "timestamp"]

DETECTION_CONF = 0.4

for folder in (UPLOAD_FOLDER, CROPS_FOLDER, ANNOTATED_FOLDER, os.path.dirname(RESULTS_CSV)):
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def append_row_to_csv(row):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_crop_worker(image_path, crops_dir, annotated_path, conf=DETECTION_CONF):
    """Calls crop_worker.detect_and_crop() in-process. Returns the same dict
    shape the old subprocess JSON used to return: {"ok": bool, "crops": [...], ...}"""
    try:
        result = crop_worker.detect_and_crop(image_path, crops_dir, annotated_path, conf)
        result["ok"] = True
    except Exception as e:
        raise RuntimeError(f"crop_worker failed: {e}")

    return result


def run_ocr_worker(crop_paths):
    """Calls ocr_worker.read_plate() in-process for each crop. Returns a list
    of result dicts, same shape as the old subprocess JSON array."""
    results = []
    for path in crop_paths:
        try:
            results.append({**ocr_worker.read_plate(path), "ok": True})
        except Exception as e:
            results.append({"image_path": path, "ok": False, "error": str(e)})
    return results


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, error=None, annotated_url=None)


@app.route("/predict", methods=["POST"])
def predict():
    saved_path = None

    # ---- Option 1: uploaded file from PC ----
    uploaded_file = request.files.get("image_file")
    if uploaded_file and uploaded_file.filename:
        if not allowed_file(uploaded_file.filename):
            return render_template(
                "index.html", results=None, annotated_url=None,
                error="Is file type ko support nahi karta. Sirf jpg, jpeg, png, bmp, webp use karo.",
            )
        ext = uploaded_file.filename.rsplit(".", 1)[1].lower()
        source_name = f"source_{uuid.uuid4().hex}.{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, source_name)
        uploaded_file.save(saved_path)

    # ---- Option 2: pasted image URL (e.g. from Chrome) ----
    else:
        image_url = (request.form.get("image_url") or "").strip()
        if not image_url:
            return render_template(
                "index.html", results=None, annotated_url=None,
                error="Ek image upload karo ya link paste karo.",
            )
        try:
            resp = requests.get(image_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as e:
            return render_template(
                "index.html", results=None, annotated_url=None,
                error=f"Link se image download nahi ho saki: {e}",
            )

        content_type = resp.headers.get("Content-Type", "")
        ext = "jpg"
        if "png" in content_type:
            ext = "png"
        elif "webp" in content_type:
            ext = "webp"
        elif "bmp" in content_type:
            ext = "bmp"

        source_name = f"source_{uuid.uuid4().hex}.{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, source_name)
        with open(saved_path, "wb") as f:
            f.write(resp.content)

    # ---- Sanity-check the saved file is a real image before running detection ----
    frame = cv2.imread(saved_path)
    if frame is None:
        return render_template(
            "index.html", results=None, annotated_url=None,
            error="Ye file ek valid image nahi lag rahi. Doosri image try karo.",
        )

    # ---- Step 1: detect + crop plate(s) -- in-process call into crop_worker ----
    annotated_name = f"annotated_{uuid.uuid4().hex}.jpg"
    annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_name)

    try:
        crop_result = run_crop_worker(saved_path, CROPS_FOLDER, annotated_path)
    except Exception as e:
        return render_template(
            "index.html", results=None, annotated_url=None, error=str(e),
        )

    annotated_url = url_for("static", filename=f"output/annotated/{annotated_name}")
    crops = crop_result.get("crops", [])

    if not crops:
        return render_template(
            "index.html", results=[], annotated_url=annotated_url, error=None,
        )

    # ---- Step 2: OCR all crops -- in-process call into ocr_worker ----
    crop_paths = [c["crop_path"] for c in crops]
    try:
        ocr_results = run_ocr_worker(crop_paths)
    except Exception as e:
        return render_template(
            "index.html", results=None, annotated_url=annotated_url, error=str(e),
        )

    ocr_by_path = {r["image_path"]: r for r in ocr_results}

    # ---- Step 3: match OCR back to detections, save crop + append CSV, render ----
    results = []
    for c in crops:
        crop_path = c["crop_path"]
        detection_conf = c["conf"]
        ocr = ocr_by_path.get(crop_path, {})

        if ocr.get("ok", False):
            plate_text = ocr.get("plate_text", "")
            model_year = ocr.get("model_year", "")
            confidence = ocr.get("confidence", 0.0)
        else:
            plate_text, model_year,  confidence =  "", "", 0.0

        crop_name = os.path.basename(crop_path)
        crop_url = url_for("static", filename=f"output/crops/{crop_name}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        row = {
            "image_name": crop_name,
            "detected_text": plate_text,
            "confidence": confidence,
            "model": model_year,
            "timestamp": timestamp,
        }
        append_row_to_csv(row)

        results.append({
            "crop_url": crop_url,
            "plate_text": plate_text,
            "model_year": model_year,
            "confidence": confidence,
            "detection_conf": round(detection_conf, 2),
        })

    return render_template(
        "index.html", results=results, annotated_url=annotated_url, error=None,
    )


if __name__ == "__main__":
    app.run(debug=False)