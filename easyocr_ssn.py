import cv2
import numpy as np
import easyocr
from tkinter import Tk, filedialog
import re
from matplotlib import pyplot as plt

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    return thresh

import re

def extract_fields_easyocr(result):
    # Sort lines by vertical position (top to bottom)
    lines = sorted(result, key=lambda x: x[0][0][1])
    texts = [text.strip() for (bbox, text, conf) in lines]
    confs = [conf for (bbox, text, conf) in lines]

    # SSN: Find line with at least 8 digits, fix format
    ssn = "Not found"
    for text in texts:
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) >= 9:
            # Take the first 9 digits
            ssn_digits = digits[:9]
            if len(ssn_digits) == 9:
                ssn = f"{ssn_digits[:3]}-{ssn_digits[3:5]}-{ssn_digits[5:]}"
                break

    # Name: Find two consecutive lines with high confidence that are all uppercase
    name = "Not found"
    for i in range(len(texts) - 1):
        if (texts[i].isalpha() and texts[i].isupper() and
            texts[i+1].isalpha() and texts[i+1].isupper() and
            confs[i] > 0.7 and confs[i+1] > 0.7):
            name = texts[i] + " " + texts[i+1]
            name_idx = i+1
            break

    # Signature: Next line after name, not all uppercase, not empty, not "SIGNATURE"
    signature = "Not found"
    if 'name_idx' in locals() and name_idx + 1 < len(texts):
        sig_candidate = texts[name_idx + 1]
        if len(sig_candidate) > 2 and not sig_candidate.isupper() and "SIGN" not in sig_candidate.upper():
            signature = sig_candidate

    return ssn, name, signature


# --- Manual file selection dialog ---
root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select SSN Image",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp;*.tiff")]
)
if not image_path:
    print("No file selected.")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("Could not open image! Check the file path and format.")
    exit()

proc_img = preprocess_image(img)

reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(proc_img)

print("----- EasyOCR Raw Output -----")
for bbox, text, conf in result:
    print(f"Text: '{text}' | Confidence: {conf:.2f}")
print("------------------------------")

ssn, name, signature = extract_fields_easyocr(result)

print("----- Extracted Fields -----")
print(f"SSN Number: {ssn}")
print(f"Printed Name: {name}")
print(f"Signature: {signature}")

import os
import json
import time

# Prepare output data
output_data = {
    "SSN_Number": ssn,
    "Printed_Name": name,
    "Signature": signature
}

# Create output folder if it doesn't exist
output_folder = "ssn_output"
os.makedirs(output_folder, exist_ok=True)

# Use image filename and timestamp for uniqueness
base_name = os.path.splitext(os.path.basename(image_path))[0]
json_filename = f"{base_name}_{int(time.time())}.json"
json_path = os.path.join(output_folder, json_filename)

# Save as JSON
with open(json_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nJSON output saved to: {json_path}")



# Optional: Show the preprocessed image
plt.imshow(proc_img, cmap='gray')
plt.title('Preprocessed for OCR')
plt.axis('off')
plt.show()
