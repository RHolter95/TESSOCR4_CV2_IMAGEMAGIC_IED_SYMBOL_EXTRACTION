# TESSOCR4_CV2_IMAGEMAGIC_IED_SYMBOL_EXTRACTION
Extracts all required data from IED (Internal Execution Doc) using CV2 and ImageMagic for symbol manipulation
Replace your image path for IED on line 139 with "Image(filename='YOUR_IMAGE_PATH_HERE') as img:"
Run program. This will cleanup your image and process it to consol.

All images used in preparing data are custom generated under existing file path names.
To change these names alter lines 146,151,156 to your desired PATH and NAME.

Example:
Line 146
"img = cv2.imread('C:/Users/mrmug/PycharmProjects/OCR/MillsHSkewedRight.png')"
Change to 
"img = cv2.imread('YOUR_FILE_PATH/YOUR_FILE_NAME.png')"

This will generate documentation to your path under your desired name.

Example:
Line 151 & 156
"cv2.imwrite('MillsHSkewedRight.png',img_erosion)"
Change to
"cv2.imwrite('YOUR_FILE_NAME.png',img_erosion)"

This will allow the program to manipulate your document and save it to YOUR_FILE_NAME.png

