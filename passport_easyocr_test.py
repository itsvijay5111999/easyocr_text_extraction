import os
import json
import cv2
import numpy as np
import easyocr
from datetime import datetime
from tkinter import Tk, filedialog

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thresh

def find_mrz_region(image):
    # Try classic detection first (as in your code)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, rectKernel)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    mrzBox = None
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
        if percentWidth > 0.7 and percentHeight > 0.03:
            mrzBox = (x, y, w, h)
            break
    # Fallback: if not found, try to crop bottom 20-25% of the image (where MRZ is usually found)
    if mrzBox is None:
        h_crop = int(H * 0.22)
        y_start = H - h_crop
        mrzBox = (0, y_start, W, h_crop)
    return mrzBox


def extract_mrz_text_easyocr(image, mrz_box, reader):
    if mrz_box is None:
        return None
    x, y, w, h = mrz_box
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    x, y = max(0, x - pX), max(0, y - pY)
    w, h = w + (pX * 2), h + (pY * 2)
    mrz_region = image[y:y + h, x:x + w]
    mrz_region_rgb = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2RGB)
    results = reader.readtext(mrz_region_rgb, detail=0, paragraph=False)
    mrz_text = "\n".join(results).replace(" ", "")
    return mrz_text, mrz_region

def clean_and_split_mrz(mrz_text):
    mrz_text = mrz_text.replace(' ', '').replace('\r', '')
    lines = [line.strip() for line in mrz_text.split('\n') if line.strip()]
    if len(lines) == 1 and len(lines[0]) >= 88:
        lines = [lines[0][:44], lines[0][44:88]]
    elif len(lines) > 2:
        lines = sorted(lines, key=len, reverse=True)[:2]
        lines = sorted(lines, key=lambda x: mrz_text.find(x))
    lines = [(line + '<'*44)[:44] for line in lines]
    return lines if len(lines) == 2 else None

def mrz_date_to_formats(mrz_date):
    if not mrz_date or len(mrz_date) != 6 or not mrz_date.isdigit():
        return "", ""
    yy = int(mrz_date[:2])
    mm = int(mrz_date[2:4])
    dd = int(mrz_date[4:6])
    current_year = datetime.now().year % 100
    century = 1900 if yy > current_year else 2000
    try:
        dt = datetime(century + yy, mm, dd)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%d-%m-%Y")
    except ValueError:
        return "", ""

def parse_mrz_data(mrz_text):
    import re
    sex_map = {'M': 'Male', 'F': 'Female', 'X': 'Unspecified', '<': 'Unspecified'}
    lines = clean_and_split_mrz(mrz_text)
    if not lines or len(lines) < 2:
        return None
    line1, line2 = lines
    line1 = re.sub(r'[^A-Z0-9<]', '<', line1.upper())
    line2 = re.sub(r'[^A-Z0-9<]', '<', line2.upper())
    line1 = (line1 + '<'*44)[:44]
    line2 = (line2 + '<'*44)[:44]
    print("MRZ line 1:", line1)
    print("MRZ line 2:", line2)
    parsed_data = {}
    try:
        parsed_data['document_type'] = line1[0]
        parsed_data['issuing_country'] = line1[2:5]
        name_field = line1[5:44]
        if '<<' in name_field:
            surname, given_names = name_field.split('<<', 1)
            parsed_data['surname'] = surname.replace('<', ' ').strip()
            parsed_data['given_names'] = given_names.replace('<', ' ').strip() or "Not found"
        else:
            parsed_data['surname'] = name_field.replace('<', ' ').strip()
            parsed_data['given_names'] = "Not found"
        parsed_data['passport_number'] = line2[0:9].replace('<', '')
        parsed_data['nationality'] = line2[10:13]
        parsed_data['date_of_birth'] = line2[13:19]
        ymd, dmy = mrz_date_to_formats(parsed_data['date_of_birth'])
        parsed_data['date_of_birth_yyyy_mm_dd'] = ymd
        parsed_data['date_of_birth_dd_mm_yyyy'] = dmy
        sex_raw = line2[20] if len(line2) > 20 else '<'
        parsed_data['sex'] = sex_map.get(sex_raw.upper(), 'Unspecified')
        parsed_data['expiry_date'] = line2[21:27]
        ymd, dmy = mrz_date_to_formats(parsed_data['expiry_date'])
        parsed_data['expiry_date_yyyy_mm_dd'] = ymd
        parsed_data['expiry_date_dd_mm_yyyy'] = dmy
    except Exception as e:
        print(f"Error parsing MRZ: {e}")
        return None
    return parsed_data

def save_results(result, output_folder="output", base_filename="passport_data"):
    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, f"{base_filename}.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=4)
    txt_path = os.path.join(output_folder, f"{base_filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("Raw MRZ Text:\n")
        txt_file.write(repr(result['raw_mrz_text']) + "\n\n")
        txt_file.write("Parsed Data:\n")
        if result['parsed_data']:
            for key, value in result['parsed_data'].items():
                txt_file.write(f"{key.replace('_', ' ').title()}: {value}\n")
        else:
            txt_file.write("Could not parse MRZ data\n")

def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Passport Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not file_path:
        print("No file selected.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not load image from {file_path}")
        return

    preprocessed = preprocess_image(image)
    mrz_box = find_mrz_region(image)
    result = None

    reader = easyocr.Reader(['en'], gpu=False)

    if mrz_box is not None:
        mrz_text, mrz_region = extract_mrz_text_easyocr(image, mrz_box, reader)
        parsed_data = parse_mrz_data(mrz_text)
        result = {
            'raw_mrz_text': mrz_text,
            'parsed_data': parsed_data
        }
    else:
        print("Automatic MRZ detection failed. Please provide a clearer image.")
        return

    print("\nPassport Extraction Result:")
    print(json.dumps(result, indent=2))
    save_results(result, output_folder="output", base_filename="passport_data")
    print("\nResults saved to 'output/passport_data.json' and 'output/passport_data.txt'")

if __name__ == "__main__":
    main()
