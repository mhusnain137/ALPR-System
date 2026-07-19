# Automatic License Plate Recognition

An Automatic License Plate Recognition (ALPR) system built with **YOLOv8**, **PaddleOCR**, **OpenCV**, and **Flask**. The application detects license plates from uploaded images, extracts the plate text using OCR, identifies the registration year (if available), and stores the results in a CSV file with timestamps.

---

## Features

- License plate detection using YOLOv8
- Optical Character Recognition (OCR) using PaddleOCR
- Image preprocessing for improved OCR accuracy
- Automatic extraction of registration/model year
- Upload images directly or provide an image URL
- Displays detected plates with annotated bounding boxes
- Saves cropped license plate images
- Stores recognition results in a CSV file
- User-friendly Flask web interface

---

## Technologies Used

- Python 3.9
- Flask
- YOLOv8 (Ultralytics)
- PaddleOCR
- OpenCV
- NumPy

---

## Project Structure

```
License-Plate-Recognition/
│
├── app.py
├── crop_worker.py
├── ocr_worker.py
├── license_plate_detector.pt
├── requirements.txt
│
├── static/
│   ├── uploads/
│   └── output/
│       ├── annotated/
│       └── crops/
│
├── output/
│   └── plate_texts.csv
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## Workflow

1. Upload an image or provide an image URL.
2. YOLOv8 detects license plate(s).
3. Each detected plate is cropped automatically.
4. PaddleOCR extracts the plate text.
5. The text is cleaned and normalized.
6. The registration/model year is extracted (when available).
7. The annotated image and cropped plates are displayed.
8. Results are saved in `plate_texts.csv`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mhusnain137/License-Plate-Recognition.git
```

Navigate to the project folder:

```bash
cd License-Plate-Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Output

For each processed image, the application generates:

- Detected license plate text
- OCR confidence score
- Registration/model year (if available)
- Cropped license plate image
- Annotated source image
- CSV record containing all recognition results

---

## Future Improvements

- Real-time webcam support
- Video stream processing
- Multiple OCR engine support
- REST API endpoints
- Docker deployment
- Database integration
- Improved OCR accuracy for challenging images

---

## How It Works

1. **Image Input**
   - The user uploads an image or provides an image URL through the Flask web interface.

2. **Image Processing**
   - The source image is saved locally for further processing.

3. **License Plate Detection**
   - `crop_worker.py` uses **YOLOv8** to detect license plates, crop each detected plate, and generate an annotated image with bounding boxes.

4. **Text Recognition**
   - `ocr_worker.py` uses **PaddleOCR** to recognize the license plate text. Multiple preprocessing techniques are applied to improve OCR accuracy.

5. **Post-processing**
   - The recognized text is cleaned and normalized, and the registration/model year is extracted when available.

6. **Result Storage**
   - The recognized information (plate text, confidence score, model year, and timestamp) is automatically appended to `output/plate_texts.csv`.

7. **Display**
   - The application displays the annotated image, cropped license plate(s), and OCR results in the browser.

---

## License

This project is intended for educational and research purposes.

---

## Author

**Muhammad Husnain**

Computer Science Student | Machine Learning & Computer Vision Enthusiast
