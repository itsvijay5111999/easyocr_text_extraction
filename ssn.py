import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import string
from tkinter import Tk, filedialog

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open image!")
    # Upscale if too small
    h, w = img.shape[:2]
    if min(h, w) < 800:
        scale = 800 / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sharpen
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    # Otsu's thresholding
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological opening to remove noise
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Deskew
    coords = np.column_stack(np.where(opened > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = opened.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(opened, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = opened
    return deskewed

def clean_lines(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        # Remove lines with too few alphanumerics or too short
        if len(line.strip()) < 4:
            continue
        alnum_ratio = sum(c.isalnum() for c in line) / max(len(line),1)
        if alnum_ratio > 0.5:
            filtered.append(line)
    return '\n'.join(filtered)

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select SSN Image",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp;*.tiff")]
)

if file_path:
    processed_img = preprocess_image(file_path)
    pil_img = Image.fromarray(processed_img)
    # Restrict OCR to uppercase, hyphen, and digits
    config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '

    text = pytesseract.image_to_string(pil_img, config=config)
    cleaned = clean_lines(text)
    print("----- Cleaned Extracted Text -----")
    print(cleaned)
    print("----------------------------------")
    # Extract SSN and name as before
    ssn_pattern = r'(?!666|000|9\d{2})\d{3}-\d{2}-\d{4}'
    ssn_matches = re.findall(ssn_pattern, cleaned)
    ssn_number = ssn_matches[0] if ssn_matches else "Not found"
    name = "Not found"
    for line in cleaned.split('\n'):
        words = line.strip().split()
        if len(words) >= 2 and all(word.isalpha() and word.isupper() for word in words):
            name = ' '.join(words)
            break
    print(f"SSN Number: {ssn_number}")
    print(f"Name: {name}")
else:
    print("No file selected.")
