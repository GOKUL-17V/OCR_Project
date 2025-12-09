import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_image, resize_image
from ocr_engine import extract_text_combined
from text_extraction import extract_target_text, clean_extracted_text

test_image = "ReverseWay Bill/reverseWaybill-156387426414724544_1.jpg"

if os.path.exists(test_image):
    print(f"Testing with: {test_image}")
    print("Preprocessing image...")
    preprocessed = preprocess_image(test_image)
    preprocessed = resize_image(preprocessed)
    
    print("Performing OCR...")
    ocr_results = extract_text_combined(preprocessed, use_easyocr=True, use_tesseract=True)
    print("Extracting target text...")
    extraction_result = extract_target_text(ocr_results, pattern="_1_")
    
    if extraction_result['target_text']:
        print(f"\n✅ SUCCESS!")
        print(f"Target Text: {extraction_result['target_text']}")
        print(f"Source Engine: {extraction_result['source_engine']}")
        if extraction_result['confidence']:
            print(f"Confidence: {extraction_result['confidence']}%")
    else:
        print("\n❌ Pattern '_1_' not found")
        print("\nAll OCR text:")
        for engine, texts in ocr_results.items():
            print(f"\n{engine}:")
            for text, conf, bbox in texts[:10]:  # Show first 10
                print(f"  - {text} (conf: {conf}%)")
else:
    print(f"Image not found: {test_image}")


