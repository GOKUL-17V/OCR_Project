OCR Text Extraction – Shipping Label _1_ Line
Overview

This tool extracts the exact text line containing the pattern _1_ from shipping label images using an OCR pipeline built with Tesseract (pytesseract) and OpenCV.
It is designed for high accuracy and fast processing, with a lightweight Streamlit interface for easy testing.

👉 Live App: https://ocrproject-gokul.streamlit.app/

Key Features

Preprocessing of images using OpenCV

Line-level OCR with Tesseract

Accurate detection of the line containing _1_

JSON output for each processed image

Simple Streamlit UI for uploading and testing

Basic test cases included for validation

Project Structure
project-root/
├── README.md
├── requirements.txt
├── src/
│   ├── ocr_engine.py
│   ├── preprocessing.py
│   ├── text_extraction.py
│   └── utils.py
├── app.py

How to Install & Run
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
streamlit run app.py
