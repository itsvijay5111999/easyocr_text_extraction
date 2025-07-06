import cv2
import numpy as np
import pytesseract
import re
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

class PassportReader:
    def __init__(self):
        # Uncomment and set this path if using Windows and Tesseract is not in PATH
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def find_mrz_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / float(h)
        if aspect_ratio < 2.0:
            return None
        return (x, y, w, h)

    def extract_mrz_text(self, image, mrz_box):
        if mrz_box is None:
            return None
        x, y, w, h = mrz_box
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        x, y = max(0, x - pX), max(0, y - pY)
        w, h = w + (pX * 2), h + (pY * 2)
        mrz_region = image[y:y + h, x:x + w]
        mrz_text = pytesseract.image_to_string(mrz_region, config='--psm 6')
        mrz_text = mrz_text.replace(" ", "")
        return mrz_text, mrz_region

    def parse_mrz_data(self, mrz_text):
        if not mrz_text:
            return None
        lines = [line.strip() for line in mrz_text.split('\n') if line.strip()]
        if len(lines) < 2:
            return None

        # Pad or trim lines to 44 characters and clean up
        line1 = lines[0].replace(' ', '').replace('\n', '').replace('\r', '')
        line2 = lines[1].replace(' ', '').replace('\n', '').replace('\r', '')
        line1 = re.sub(r'[^A-Z0-9<]', '<', line1.upper())
        line2 = re.sub(r'[^A-Z0-9<]', '<', line2.upper())
        line1 = (line1 + '<' * 44)[:44]
        line2 = (line2 + '<' * 44)[:44]

        parsed_data = {}
        try:
            # First line
            parsed_data['document_type'] = line1[0]
            parsed_data['issuing_country'] = line1[2:5]
            name_field = line1[5:44]
            if '<<' in name_field:
                surname, given_names = name_field.split('<<', 1)
                parsed_data['surname'] = surname.replace('<', ' ').strip()
                parsed_data['given_names'] = given_names.replace('<', ' ').strip()
            else:
                parsed_data['surname'] = name_field.replace('<', ' ').strip()
                parsed_data['given_names'] = ''
            # Second line
            parsed_data['passport_number'] = line2[0:9].replace('<', '')
            parsed_data['nationality'] = line2[10:13]
            parsed_data['date_of_birth'] = line2[13:19]
            parsed_data['sex'] = line2[20]
            parsed_data['expiry_date'] = line2[21:27]
        except Exception as e:
            print(f"Error parsing MRZ: {e}")
            return None
        return parsed_data

    def extract_passport_details(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        mrz_box = self.find_mrz_region(image)
        if mrz_box is None:
            print("MRZ region not found in the image")
            return None
        result = self.extract_mrz_text(image, mrz_box)
        if result is None:
            print("Could not extract MRZ text")
            return None
        mrz_text, mrz_region = result
        parsed_data = self.parse_mrz_data(mrz_text)
        return {
            'raw_mrz_text': mrz_text,
            'parsed_data': parsed_data
            # 'mrz_region': mrz_region
        }

def main():
    # Hide the root window of tkinter
    root = Tk()
    root.withdraw()
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Passport Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not file_path:
        print("No file selected.")
        return
    passport_reader = PassportReader()
    result = passport_reader.extract_passport_details(file_path)
    print(result)
    if result:
        print("\nRaw MRZ Text:")
        print(repr(result['raw_mrz_text']))  # Show raw output for debugging

        print("\nParsed Data:")
        if result['parsed_data']:
            for key, value in result['parsed_data'].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print("Could not parse MRZ data")

        # if result['mrz_region'] is not None:
        #     # Save the MRZ region as an image file
        #     cv2.imwrite("mrz_region.png", result['mrz_region'])
        #     print("MRZ region saved as mrz_region.png")
        #     # Display the MRZ region using matplotlib
        #     plt.imshow(cv2.cvtColor(result['mrz_region'], cv2.COLOR_BGR2RGB))
        #     plt.title("MRZ Region")
        #     plt.axis('off')
        #     plt.show()
    else:
        print("Failed to extract passport details")

if __name__ == "__main__":
    main()
