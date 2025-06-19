# JPG to TXT - Automated Image Text Extraction

This project automatically scans `.jpg` images in a folder, detects and flattens documents (like CamScanner), extracts text using OCR, and saves the results as `.txt` files.

## Features

- Processes all new `.jpg` files in the `~/Desktop/JPG to TXT` folder
- Ignores images already processed (those with `extracted` in the filename)
- Extracted text is saved in the `extracted_txt` folder
- Uses OpenCV for document detection and EasyOCR for text extraction

## Requirements

- Python 3.12+
- See `requirements.txt` for dependencies

## Usage

1. Clone this repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Place your `.jpg` images in `~/Desktop/JPG to TXT`.
4. Run the script:
   ```sh
   python ImageToTXT.py
   ```
5. Extracted text files will appear in the `extracted_txt` folder.

## Notes

- Only processes `.jpg` files that do **not** have `extracted` in their filename.
- Processed images are renamed to include `_extracted`.
- Make sure the `JPG to TXT` folder exists on your Desktop.

---

**Enjoy automated image-to-text extraction!**
