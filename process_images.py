import os
import sys
import json
import cv2
from pathlib import Path
from typing import Dict, List
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_image, resize_image
from ocr_engine import extract_text_combined
from text_extraction import extract_target_text, clean_extracted_text
from utils import save_json_output, get_image_files, calculate_accuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_single_image(image_path: str, output_dir: str = "results") -> Dict:
    logger.info(f"Processing image: {image_path}") 
    result = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'target_text': None,
        'source_engine': None,
        'confidence': None,
        'all_ocr_text': [],
        'error': None
    }
    
    try:
        preprocessed = preprocess_image(image_path)
        preprocessed = resize_image(preprocessed)
        ocr_results = extract_text_combined(preprocessed, use_easyocr=True, use_tesseract=True)
        all_texts = []
        for engine, texts in ocr_results.items():
            for text, conf, bbox in texts:
                all_texts.append(text)
        result['all_ocr_text'] = list(set(all_texts)) 
        
        extraction_result = extract_target_text(ocr_results, pattern="_1_")
        
        if extraction_result['target_text']:
            result['target_text'] = clean_extracted_text(extraction_result['target_text'])
            result['source_engine'] = extraction_result['source_engine']
            result['confidence'] = extraction_result['confidence']
        else:
            result['error'] = "Target pattern '_1_' not found in OCR results"
            logger.warning(f"Pattern not found in {image_path}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {image_path}: {str(e)}")
    
    return result


def process_all_images(input_dir: str = "ReverseWay Bill", output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)
    image_files = get_image_files(input_dir)
    logger.info(f"Found {len(image_files)} images to process")
    
    all_results = []
    
    for image_path in image_files:
        result = process_single_image(image_path, output_dir)
        all_results.append(result)
        image_name = Path(image_path).stem
        json_path = os.path.join(output_dir, "json", f"{image_name}.json")
        save_json_output(result, json_path)
        
        logger.info(f"Processed {image_path}: {result.get('target_text', 'NOT FOUND')}")
    combined_output = {
        'total_images': len(all_results),
        'successful_extractions': sum(1 for r in all_results if r.get('target_text')),
        'failed_extractions': sum(1 for r in all_results if not r.get('target_text')),
        'results': all_results
    }
    
    combined_json_path = os.path.join(output_dir, "all_results.json")
    save_json_output(combined_output, combined_json_path)
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"Total images: {combined_output['total_images']}")
    logger.info(f"Successful: {combined_output['successful_extractions']}")
    logger.info(f"Failed: {combined_output['failed_extractions']}")
    
    return all_results


if __name__ == "__main__":
    results = process_all_images()
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    for result in results:
        status = "✓" if result.get('target_text') else "✗"
        print(f"{status} {result['image_name']}: {result.get('target_text', 'NOT FOUND')}")


