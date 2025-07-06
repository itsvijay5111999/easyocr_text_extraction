import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from tkinter import Tk, filedialog

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open image!")
    # Upscale if small for better OCR
    h, w = img.shape[:2]
    if min(h, w) < 800:
        scale = 800 / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sharpen the image
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    # Otsu's thresholding for binarization
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_ssn_and_name(text):
    # SSN regex: matches valid SSNs like 123-45-6789
    ssn_pattern = r'(?!666|000|9\d{2})\d{3}-\d{2}-\d{4}'
    ssn_matches = re.findall(ssn_pattern, text)
    ssn_number = ssn_matches[0] if ssn_matches else "Not found"
    # Name: look for the line after "ESTABLISHED FOR" or all-uppercase line with 2+ words
    name = "Not found"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "ESTABLISHED FOR" in line.upper() and i+1 < len(lines):
            possible_name = lines[i+1].strip()
            # Accept names with at least two words, all alphabetic
            if len(possible_name.split()) >= 2 and all(w.isalpha() for w in possible_name.split()):
                name = possible_name
                break
    if name == "Not found":
        for line in lines:
            words = line.strip().split()
            if len(words) >= 2 and all(word.isalpha() and word.isupper() for word in words):
                name = ' '.join(words)
                break
    return ssn_number, name

# Manual file selection dialog
root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select SSN Image",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp;*.tiff")]
)

if not image_path:
    print("No file selected.")
    exit()

processed_img = preprocess_image(image_path)
pil_img = Image.fromarray(processed_img)
config = r'--oem 3 --psm 6 -l eng'
text = pytesseract.image_to_string(pil_img, config=config)

print("----- Extracted Text -----")
print(text)
print("--------------------------")

ssn_number, name = extract_ssn_and_name(text)
print(f"SSN Number: {ssn_number}")
print(f"Name: {name}")

