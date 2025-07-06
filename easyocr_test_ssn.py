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

    ssn = "Not found"
    name = "Not found"
    signature = "Not found"

    # SSN Extraction - Most flexible
    for i, text in enumerate(texts):
        # Clean text: remove non-digits, then try to match common patterns
        cleaned_text = ''.join(c for c in text if c.isdigit() or c == '-')
        
        # Pattern 1: xxxx-xxxx (if SSN is partial)
        if re.match(r'^\d{4,5}-\d{4}$', cleaned_text) and confs[i] > 0.3:
            # Check for leading digits in preceding lines
            if i > 0:
                prev_text = ''.join(c for c in texts[i-1] if c.isdigit())
                if len(prev_text) >= 3:
                    ssn_digits = prev_text[:3] + cleaned_text.replace('-', '')
                    if len(ssn_digits) >= 9:
                        ssn = f"{ssn_digits[:3]}-{ssn_digits[3:5]}-{ssn_digits[5:9]}"
                        break
        # Pattern 2: xxx-xx-xxxx or xxx-xxxxx
        elif re.match(r'^\d{3}-\d{2,5}-\d{4}$', cleaned_text) and confs[i] > 0.3:
            digits = ''.join(c for c in cleaned_text if c.isdigit())
            if len(digits) >= 9:
                ssn = f"{digits[:3]}-{digits[3:5]}-{digits[5:9]}"
                break
        # Pattern 3: XXX-XX-XXXX (literal on card)
        elif 'XXX-XX-XXXX' in text.upper() and confs[i] > 0.3:
            ssn = 'XXX-XX-XXXX'
            break
        # Pattern 4: 9 consecutive digits (if no dashes)
        elif len(''.join(c for c in text if c.isdigit())) >= 9 and confs[i] > 0.3:
            digits = ''.join(c for c in text if c.isdigit())[:9]
            ssn = f"{digits[:3]}-{digits[3:5]}-{digits[5:9]}"
            break


    # Name Extraction - More flexible, allows for "i" in name and joins parts
    name_candidates = []
    found_keyword = False
    for i, text in enumerate(texts):
        # Identify the "THIS NUMBER HAS BEEN ESTABLISHED FOR" line
        if "ESTABLISHED FOR" in text.upper() and confs[i] > 0.1:
            found_keyword = True
            continue # Skip this line

        if found_keyword and confs[i] > 0.5: # Consider lines after the keyword
            # Try to build a two-word name
            current_name_parts = []
            if text.isalpha() and text.isupper():
                current_name_parts.append(text)
                # Check for next line being the second part of name
                if i + 1 < len(texts) and texts[i+1].isalpha() and texts[i+1].isupper() and confs[i+1] > 0.5:
                    current_name_parts.append(texts[i+1])
                    name = " ".join(current_name_parts)
                    break
                elif len(text.split()) == 2 and text.replace('i', '').isalpha() and text.isupper(): # Handle "JOHN SMITH" as one line
                    name = text
                    break
            elif len(text.split()) == 2 and text.replace('i', '').isalpha() and text.isupper() and confs[i] > 0.5:
                # Direct match for "JOHN SMITH" as one line, even if it has 'i'
                name = text
                break
            
    # Fallback for name: two consecutive all-uppercase alpha lines, high confidence
    if name == "Not found":
        for i in range(len(texts) - 1):
            if (texts[i].replace('i','').isalpha() and texts[i].isupper() and
                texts[i+1].replace('i','').isalpha() and texts[i+1].isupper() and
                confs[i] > 0.8 and confs[i+1] > 0.8 and
                "SIGN" not in texts[i].upper() and "SIGN" not in texts[i+1].upper()):
                name = texts[i] + " " + texts[i+1]
                break
        if name == "Not found" and len(texts) >= 1: # Single word name fallback, less ideal
             if texts[0].replace('i','').isalpha() and texts[0].isupper() and confs[0] > 0.8:
                 name = texts[0] # Very risky, might pick up "SECURITY"

    # Signature Extraction - still tricky with garbled text
    # Search for a line after the potential name or after "SIGNATURE" keyword
    signature = "Not found"
    name_index = -1
    if name != "Not found":
        name_index = -1
        # Find the line index of the extracted name (can be composite)
        for i, t in enumerate(texts):
            if name in t and confs[i] > 0.8: # high confidence match
                name_index = i
                break
        
        if name_index != -1 and name_index + 1 < len(texts):
            sig_candidate = texts[name_index + 1]
            if (len(sig_candidate) > 2 and not sig_candidate.isupper() and 
                "SIGNATURE" not in sig_candidate.upper() and confs[name_index+1] > 0.3):
                signature = sig_candidate
    
    # Fallback for signature if name not found or bad signature
    if signature == "Not found":
        for i, text in enumerate(texts):
            if "SIGNATURE" in text.upper():
                if i + 1 < len(texts):
                    sig_candidate = texts[i+1]
                    if len(sig_candidate) > 2 and not sig_candidate.isupper() and confs[i+1] > 0.3:
                        signature = sig_candidate
                break

    return ssn or "Not found", name or "Not found", signature or "Not found"



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

# Optional: Show the preprocessed image
plt.imshow(proc_img, cmap='gray')
plt.title('Preprocessed for OCR')
plt.axis('off')
plt.show()
