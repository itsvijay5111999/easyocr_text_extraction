import cv2
import easyocr
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import re
import difflib
import json

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    target_width = 800
    scale_factor = target_width / img.shape[1]
    width = target_width
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=25, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised_gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        blockSize=31, C=10)
    return thresholded

def parse_key_value_lines(lines):
    kv_pairs = {}
    for line in lines:
        for sep in [':', ';', '-', '=']:
            if sep in line:
                parts = line.split(sep, 1)
                key = parts[0].strip().upper()
                value = parts[1].strip()
                if key and value:
                    kv_pairs[key] = value
                break
    return kv_pairs

def detect_state(lines):
    us_states = [
        "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
        "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
        "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
        "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE",
        "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
        "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
        "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
        "WISCONSIN", "WYOMING"
    ]
    state_abbr = {
        "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA",
        "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE", "FL": "FLORIDA", "GA": "GEORGIA",
        "HI": "HAWAII", "ID": "IDAHO", "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA",
        "KS": "KANSAS", "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND",
        "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI",
        "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA", "NH": "NEW HAMPSHIRE",
        "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK", "NC": "NORTH CAROLINA",
        "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA", "OR": "OREGON", "PA": "PENNSYLVANIA",
        "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA", "SD": "SOUTH DAKOTA", "TN": "TENNESSEE",
        "TX": "TEXAS", "UT": "UTAH", "VT": "VERMONT", "VA": "VIRGINIA", "WA": "WASHINGTON",
        "WV": "WEST VIRGINIA", "WI": "WISCONSIN", "WY": "WYOMING"
    }
    for line in lines:
        upper = line.upper()
        for state in us_states:
            if state in upper:
                return state.title()
        for abbr, state in state_abbr.items():
            if re.search(r'\b{}\b'.format(abbr), upper):
                return state.title()
        matches = difflib.get_close_matches(upper, us_states, n=1, cutoff=0.8)
        if matches:
            return matches[0].title()
    return "Not Found"

def extract_pa_dl_number(lines, kv_pairs):
    for k in kv_pairs:
        if 'DLN' in k or 'DL' in k:
            digits = re.sub(r'\D', '', kv_pairs[k])
            if len(digits) == 8:
                return digits
    for line in lines:
        if 'DLN' in line.upper():
            digits = re.sub(r'\D', '', line)
            if len(digits) == 8:
                return digits
        match = re.search(r'(\d{2}\s\d{3}\s\d{3})', line)
        if match:
            digits = re.sub(r'\D', '', match.group(1))
            if len(digits) == 8:
                return digits
    for line in lines:
        digits = re.sub(r'\D', '', line)
        if len(digits) == 8:
            return digits
    return "Not Found"

def extract_general_dl_number(lines, kv_pairs):
    for k in kv_pairs:
        if any(label in k for label in ['DLN', 'DL', 'LIC', 'ID']):
            value = kv_pairs[k]
            cleaned = re.sub(r'[^A-Z0-9]', '', value)
            if 6 <= len(cleaned) <= 20 and not cleaned.isalpha():
                return cleaned
    for line in lines:
        match = re.search(r'\bWDL[A-Z0-9]{9}\b', line)
        if match:
            return match.group(0)
    for line in lines:
        guess = ''.join(re.findall(r'[A-Z0-9]', line))
        if 6 <= len(guess) <= 20 and not guess.isalpha():
            return guess
    return "Not Found"

def extract_name(lines):
    LABELS = [
        "DRIVER", "LICENSE", "DLN", "ID", "SEX", "DOB", "CLASS", "RESTR", "EYES",
        "HEIGHT", "CITY", "ZIP", "BIRTH", "EXP", "ADDR", "ORGANDONOR", "VISITPA", "DD",
        "END", "SAMPLE", "ENHANCED"
    ]
    for i, line in enumerate(lines):
        upper = line.upper()
        if any(label in upper for label in LABELS):
            # Check next 2 lines for likely name
            for offset in [1, 2]:
                if i + offset < len(lines):
                    next_line = lines[i + offset]
                    words = [w for w in next_line.split() if w.isalpha() and len(w) > 1]
                    if len(words) >= 2:
                        return " ".join(word.title() for word in words)
        # Heuristic: line with two words, both capitalized, not a label
        words = [w for w in line.split() if w.isalpha() and len(w) > 1]
        if len(words) >= 2 and not any(label in upper for label in LABELS):
            return " ".join(word.title() for word in words)
    # Fallback: first single word not a label
    for line in lines:
        upper = line.upper()
        if not any(label in upper for label in LABELS):
            words = [w for w in line.split() if w.isalpha() and len(w) > 1]
            if len(words) == 1:
                return words[0].title()
    return "Not Found"

def extract_exp_date(lines, kv_pairs):
    # Try key-value pairs first
    for k in kv_pairs:
        if 'EXP' in k:
            val = kv_pairs[k]
            m = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', val)
            if m:
                return m.group(0)
    # Search all lines for date pattern, prefer after "EXP"
    for idx, line in enumerate(lines):
        if "EXP" in line.upper():
            # Check this and next line for a date
            for offset in [0, 1]:
                if idx + offset < len(lines):
                    m = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', lines[idx + offset])
                    if m:
                        return m.group(1)
                    m2 = re.search(r'(\d{2}[/-]\d{4})', lines[idx + offset])
                    if m2:
                        return m2.group(1)
    # Fallback: any date pattern in all lines
    for line in lines:
        m = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', line)
        if m:
            return m.group(1)
        m2 = re.search(r'(\d{2}[/-]\d{4})', line)
        if m2:
            return m2.group(1)
    return "Not Found"

def extract_sex(lines, kv_pairs):
    for k in kv_pairs:
        if 'SEX' in k:
            val = kv_pairs[k].strip().upper()
            if val in ['M', 'F']:
                return val
    for idx, line in enumerate(lines):
        m = re.search(r'SEX[:\s-]*([MF])', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if 'SEX' in line.upper() and idx + 1 < len(lines):
            next_val = lines[idx + 1].strip().upper()
            if next_val in ['M', 'F']:
                return next_val
    # Fallback: isolated M/F
    for line in lines:
        if line.strip().upper() in ['M', 'F']:
            return line.strip().upper()
    return "Not Found"

def parse_driver_license_details(ocr_result_lines):
    details = {
        "DL No": "Not Found",
        "Exp Date": "Not Found",
        "Sex": "Not Found",
        "Name": "Not Found",
        "State": "Not Found"
    }
    kv_pairs = parse_key_value_lines(ocr_result_lines)
    details['State'] = detect_state(ocr_result_lines)

    if details['State'] == "Pennsylvania":
        details['DL No'] = extract_pa_dl_number(ocr_result_lines, kv_pairs)
    else:
        details['DL No'] = extract_general_dl_number(ocr_result_lines, kv_pairs)

    details['Exp Date'] = extract_exp_date(ocr_result_lines, kv_pairs)
    details['Sex'] = extract_sex(ocr_result_lines, kv_pairs)
    details['Name'] = extract_name(ocr_result_lines)

    return details, kv_pairs

def save_json_output(output_folder, filename, data):
    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def extract_text_from_images(image_paths):
    if not image_paths:
        print("No images selected for processing.")
        return {}
    reader = easyocr.Reader(['en'])
    all_extracted_details = {}
    output_folder = "output"
    for image_path in image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            all_extracted_details[os.path.basename(image_path)] = {"Error": "Image not processed."}
            continue
        ocr_result = reader.readtext(processed_img, detail=0)
        print(f"Raw OCR Result for {os.path.basename(image_path)}:\n{ocr_result}")
        details, kv_pairs = parse_driver_license_details(ocr_result)
        print(f"Extracted Specific Details for {os.path.basename(image_path)}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

        # Save JSON output
        json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
        save_json_output(output_folder, json_filename, {
            "extracted_details": details
        })

        all_extracted_details[os.path.basename(image_path)] = details

        # Print JSON output for quick view
        print("JSON output:")
        print(json.dumps({
            "raw_ocr": ocr_result,
            "extracted_details": details
        }, indent=4, ensure_ascii=False))
    return all_extracted_details

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select one or more image files (e.g., Driver's Licenses, Passports) to process for specific details.")
    image_files = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
    )
    if not image_files:
        print("No images selected. Exiting.")
    else:
        image_files_list = list(image_files)
        extract_text_from_images(image_files_list)
    root.destroy()
