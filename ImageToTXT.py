# Note: Ensure you have the necessary permissions to delete files in the directory.
#       Adjust the y_threshold in group_text_by_line if needed for better line grouping.
#       Make sure to have OpenCV and EasyOCR installed in your Python environment.
#       You can install them using pip:
#       pip install opencv-python easyocr
#       This script assumes that the images are in the current working directory.
#       Adjust the glob pattern if your images are in a different format or location.
#       The script processes all .jpg files in the current directory.
#       Ensure that the images are clear and contain text for better OCR results.
#       The script uses a temporary file to store the scanned image before OCR.

import easyocr
import numpy as np
from collections import defaultdict
import glob
import os
import cv2

# Set the target folder path (change 'zoulaimi' to your Mac username if needed)
desktop_path = os.path.expanduser("~/Desktop")
target_folder = os.path.join(desktop_path, "JPG to TXT")

# Change working directory to the target folder
if os.path.exists(target_folder):
    os.chdir(target_folder)
else:
    print(f"Target folder '{target_folder}' does not exist.")
    exit()

# Create an EasyOCR Reader object (English only; add more language codes if needed)
reader = easyocr.Reader(['en'])

def order_points(pts):
    """Order points in a consistent way: top-left, top-right, bottom-right, bottom-left."""
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Perform a perspective transform to get a "bird's eye view" of the document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute width and height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    # Destination points for "bird's eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def scan_document(image_path):
    """Scan and warp the document in the image."""
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
    if docCnt is not None:
        warped = four_point_transform(orig, docCnt.reshape(4, 2))
        return warped
    else:
        # If no contour is found, return original image
        return orig

def group_text_by_line(results, y_threshold=15):
    """Group OCR results by lines based on vertical position."""
    lines = defaultdict(list)
    for bbox, text, conf in results:
        y_avg = np.mean([bbox[0][1], bbox[1][1]])
        found_line = False
        for key in lines:
            if abs(key - y_avg) < y_threshold:
                lines[key].append((bbox, text))
                found_line = True
                break
        if not found_line:
            lines[y_avg].append((bbox, text))
    sorted_lines = []
    for key in sorted(lines.keys()):
        line = sorted(lines[key], key=lambda x: x[0][0][0])
        line_text = " ".join([t[1] for t in line])
        sorted_lines.append(line_text)
    return "\n".join(sorted_lines)

# Create output directory if it doesn't exist
output_dir = "extracted_txt"
os.makedirs(output_dir, exist_ok=True)

# Find all .jpg files in the current directory, excluding those with 'extracted' in the name
jpg_files = [f for f in glob.glob("*.jpg") if "extracted" not in f.lower()]

# Check if any .jpg files were found
if not jpg_files:
    print("No new .jpg files found in the current directory.")
    exit()

for image_path in jpg_files:
    print(f"Processing {image_path}...")
    # Scan and warp the document
    scanned_image = scan_document(image_path)
    temp_path = "temp_scanned.jpg"
    cv2.imwrite(temp_path, scanned_image)
    # OCR on the scanned image
    results = reader.readtext(temp_path)
    extracted_text = group_text_by_line(results)
    # Create output filename based on image filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_path, 'w') as f:
        f.write(extracted_text)
    # Optionally, remove the temporary file
    os.remove(temp_path)
    # Rename the processed image to include 'extracted'
    new_image_path = f"{base_name}_extracted.jpg"
    os.rename(image_path, new_image_path)
    print(f"Text extracted and saved to: {output_path}")
print("Processing complete.")