"""
ocr_worker.py
-------------
Runs INSIDE the 'alpr' conda environment (PaddleOCR lives there).

Given one or more cropped-plate image paths, this script:
    1. Runs PaddleOCR on each crop (original image, then several
       preprocessed variants if the original result isn't strong)
    2. Cleans/normalizes the text -- same rules your old read_new_plates.py
       used: mixed letter/digit token splitting, digit/letter-confusion
       fixes on the number block, province extraction (with fuzzy
       matching), model-year extraction, and format-aware scoring to pick
       the best candidate among all the OCR passes
    3. Prints ONE JSON array to stdout, one object per input image

app.py calls this via subprocess, e.g.:
    conda run -n alpr python ocr_worker.py crop1.jpg crop2.jpg
and parses the JSON array from stdout. Only JSON goes to stdout -- anything
else (PaddleOCR's own download/progress messages) goes to stderr so it
never corrupts the JSON the caller is trying to parse.

Usage:
    python ocr_worker.py <crop_path_1> [<crop_path_2> ...]
"""

import os
import re
import sys
import ssl
import json
import argparse
import numpy as np
from datetime import datetime

import cv2

# Windows-specific SSL fix (ASN1: NOT_ENOUGH_DATA) that can hit PaddleOCR the
# first time it downloads its detection/recognition model weights.
ssl._create_default_https_context = ssl._create_unverified_context

from paddleocr import PaddleOCR  # noqa: E402  (import after the SSL patch)

# Raised from 0.15 -> 0.35 upstream. At 0.15 almost every stray detection
# (dust, plate-frame edges, shadows, screw heads) was being accepted as
# "text" and getting concatenated into plate_text. That's the #1 source of
# garbage characters like a stray "M" / "8 T" / "0".
CONF_THRESHOLD = 0.30

# Only characters that can actually appear on a plate. This alone
# eliminates most of the '@', '$', random symbol garbage, and gives the OCR
# engine less room to hallucinate the wrong character.
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Common OCR misreads: letter -> the digit it actually looks like.
OCR_LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "4",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6", "B": "8",  # B is closer to 8, keep separate from G->6
    "T": "7",
    "P": "9", "g": "9",
}

SPECIAL_CHAR_MAP = {
    "@": "Q",
    "$": "S",
    "!": "I",
    "|": "I",
}

PROVINCE_KEYWORDS = {
    "PUNJAB": "Punjab",
    "ISLAMABAD": "Islamabad",
    "ICT": "Islamabad",  # "ICT" = Islamabad Capital Territory, printed alongside "ISLAMABAD"
    "SINDH": "Sindh",
    "KPK": "KPK",
    "KHYBER": "KPK",
    "BALOCHISTAN": "Balochistan",
    "BALOCH": "Balochistan",
    "GILGIT": "Gilgit-Baltistan",
}

# Rough Pakistani plate shape: 2-3 letters, optional 2-digit year,
# then a 3-4 digit number block. Used to SCORE candidates instead of
# trusting confidence alone -- a high-confidence garbage read should lose
# to a lower-confidence read that actually looks like a real plate.
PLATE_FORMAT_RE = re.compile(
    r'^[A-Z]{2,3}[- ]?\d{3,4}$|^[A-Z]{2,3}[- ]?\d{2}[- ]?\d{3,4}$'
)

_paddle_reader = None


def get_paddle_reader():
    """Loads the PaddleOCR reader once per process."""
    global _paddle_reader
    if _paddle_reader is None:
        _paddle_reader = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
        )
    return _paddle_reader


def preprocess_image(frame):
    """
    Generate multiple enhanced versions of the image for OCR. Returns a
    list of variants to try (sharpened grayscale, Otsu threshold, adaptive
    threshold) instead of a single processed image.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    clahe_img = clahe.apply(gray)

    sharpen = cv2.filter2D(
        clahe_img,
        -1,
        np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]),
    )

    otsu = cv2.threshold(
        sharpen,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    adaptive = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

    # inverted = 255 - otsu

    return [
        clahe_img,
        sharpen,
        otsu,
        adaptive,
    ]


def force_digits_only(token):
    """Forces every character in a token to a digit using common OCR
    letter<->digit confusions. Anything with no known mapping is dropped."""
    fixed_chars = []
    for ch in token:
        if ch.isdigit():
            fixed_chars.append(ch)
        elif ch.upper() in OCR_LETTER_TO_DIGIT:
            fixed_chars.append(OCR_LETTER_TO_DIGIT[ch.upper()])
    return "".join(fixed_chars)


def levenshtein(a, b):
    """Edit distance between two strings (used for fuzzy province matching)."""
    if len(a) < len(b):
        a, b = b, a
    prev_row = range(len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                prev_row[j] + 1,        # delete
                curr_row[j - 1] + 1,    # insert
                prev_row[j - 1] + cost,  # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


def extract_model_year(text):
    """Pakistani plates are often [LETTERS] [2-digit year] [NUMBER],
    e.g. 'LEA 17 8999' -> '17' is the registration year/model."""
    tokens = [t.strip(",") for t in text.split() if t.strip(",") != ""]
    if len(tokens) < 3:
        return ""
    for t in tokens[1:-1]:
        if t.isdigit() and len(t) == 2:
            return t
    return ""


def extract_province(text):
    """Extracts the province from the OCR text (with fuzzy matching) and
    removes it from the plate text. Returns (cleaned_text, province)."""
    tokens = text.upper().split()
    province = ""
    cleaned_tokens = []
    for token in tokens:
        if token in PROVINCE_KEYWORDS:
            province = PROVINCE_KEYWORDS[token]
            continue
        matched = False
        for keyword, standard_name in PROVINCE_KEYWORDS.items():
            if levenshtein(token, keyword) <= 2:  # allow up to 2 typos
                province = standard_name
                matched = True
                break
        if not matched:
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens), province


def remove_special_characters(text):
    # Replace known OCR special-character mistakes
    for ch, replacement in SPECIAL_CHAR_MAP.items():
        text = text.replace(ch, replacement)

    # Treat hyphens as word separators (not deletions) -- otherwise two
    # distinct words on either side of a "-" (e.g. "ICT-ISLAMABAD") get
    # fused into one unmatchable token once the hyphen disappears.
    text = text.replace("-", " ")

    # Remove remaining unwanted characters
    return re.sub(r'[^A-Za-z0-9\s]', '', text)


def split_mixed_token(token):
    """Splits a token like 'IN979' (letters directly touching digits with
    no space) into separate letter-run and digit-run tokens, e.g.
    'IN979' -> ['IN', '979']. If the token is already pure letters or
    pure digits, returns it unchanged as a single-item list."""
    parts = re.findall(r'[A-Za-z]+|\d+', token)
    return parts if parts else [token]


def clean_plate_text(text):
    raw_tokens = text.split()

    # First, break apart any token where letters and digits are stuck
    # together with no space (e.g. "IN979" -> ["IN", "979"]).
    tokens = []
    for t in raw_tokens:
        tokens.extend(split_mixed_token(t))

    cleaned_tokens = []

    for t in tokens:
        digit_count = sum(ch.isdigit() for ch in t)
        is_number_block = len(t) in (3, 4) and digit_count >= len(t) / 2

        if is_number_block:
            cleaned_tokens.append(force_digits_only(t))
        else:
            cleaned_tokens.append(t.upper())

    return " ".join(cleaned_tokens)


def score_plate_text(text):
    """
    Format-aware score used to pick the best candidate, instead of relying
    on raw OCR confidence alone. A garbage string can score high confidence
    yet look nothing like a real plate -- this pulls those down and pushes
    properly-shaped reads up.

    Returns a float; higher is better. Combined with confidence as a
    tie-breaker in read_plate().
    """
    if not text:
        return 0.0

    if PLATE_FORMAT_RE.match(text):
        return 1.0

    tokens = text.split()
    # Partial credit: looks roughly plate-shaped (letters then digits,
    # 2-3 tokens, no leftover junk tokens of length 1 like a stray "M").
    if 2 <= len(tokens) <= 3:
        junk_tokens = sum(1 for t in tokens if len(t) == 1)
        has_letter_block = any(t.isalpha() for t in tokens)
        has_number_block = any(t.isdigit() and len(t) >= 3 for t in tokens)
        if junk_tokens == 0 and has_letter_block and has_number_block:
            return 0.6

    return 0.0


def run_paddleocr(frame):
    reader = get_paddle_reader()

    result = reader.ocr(
        frame,
        cls=False,
        det=True,
        rec=True,
    )

    print("PaddleOCR Result:", result, file=sys.stderr)

    plate_text = ""
    confidences = []

    if result and result[0]:
        # Left to right sort, so multi-line/multi-block detections read in
        # the correct order instead of whatever order Paddle returns them.
        lines = sorted(
            result[0],
            key=lambda x: min(pt[0] for pt in x[0]),
        )

        for line in lines:
            text = line[1][0]
            conf = float(line[1][1])

            if conf >= CONF_THRESHOLD:
                plate_text += text + " "
                confidences.append(conf)

    if confidences:
        best_conf = sum(confidences) / len(confidences)
    else:
        best_conf = 0.0

    plate_text = plate_text.strip().upper()
    plate_text = remove_special_characters(plate_text)
    plate_text = clean_plate_text(plate_text)
    plate_text, province = extract_province(plate_text)
    model_year = extract_model_year(plate_text)

    return {
        "plate_text": plate_text,
        "confidence": round(best_conf, 2),
        "province": province,
        "model_year": model_year,
    }


def read_plate(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # ---------- ORIGINAL IMAGE ----------
    best_result = run_paddleocr(frame)
    best_result["score"] = score_plate_text(best_result["plate_text"])

    # If the result is already strong, don't waste time on preprocessing.
    if (
        best_result["confidence"] >= 0.95
        and best_result["score"] >= 1.0
    ):
        best_result["image_path"] = image_path
        best_result.pop("score", None)
        return best_result

    # ---------- PREPROCESSED VARIANTS ----------
    processed_images = preprocess_image(frame)

    for img in processed_images:
        result = run_paddleocr(img)
        result["score"] = score_plate_text(result["plate_text"])

        current_rank = (result["score"], result["confidence"])
        best_rank = (best_result["score"], best_result["confidence"])

        if current_rank > best_rank:
            best_result = result
        elif (
            current_rank == best_rank
            and len(result["plate_text"]) > len(best_result["plate_text"])
        ):
            best_result = result

    best_result["image_path"] = image_path
    best_result.pop("score", None)

    return best_result


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR + cleanup on one or more plate crops.")
    parser.add_argument("image_paths", nargs="+", help="Path(s) to cropped plate image(s).")
    args = parser.parse_args()

    results = []
    for path in args.image_paths:
        try:
            results.append({**read_plate(path), "ok": True})
        except Exception as e:
            results.append({"image_path": path, "ok": False, "error": str(e)})

    # ONLY JSON on stdout -- this is what app.py parses via subprocess.
    print(json.dumps(results))

    if any(not r.get("ok", False) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()