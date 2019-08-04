import tesserocr
from PIL import Image as ImagePIL
from wand.image import Image
import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, RIL, iterate_level

def deskew(im, max_skew=11):
    height, width,_ = im.shape

    # Create a grayscale image and denoise it
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return im
def DiverseSupplier():
    # Gets Bounds of Line on DiverseSupplier by cropping
    y = 510
    x = 1800
    h = 80
    w = 175
    crop_img = img[y:y + h, x:x + w]
    #cv2.imshow('Image', crop_img)
    cv2.imwrite('Crop_DiverseSupplier.png', crop_img)
    cv2.waitKey(0)

    imageOne = 'C:/Users/mrmug/PycharmProjects/OCR/Crop_DiverseSupplier.png'
    text = symbolConfidenc(imageOne)

    # imageOne = ImagePIL.open('C:/Users/mrmug/PycharmProjects/OCR/Crop_DiverseSupplier.png')
    #text = tesserocr.image_to_text(imageOne)  # print ocr text from image
    print('Diverse Supplier : ' + text)

    return
def ProjectName():
    # Gets Bounds of Line on DiverseSupplier by cropping
    y = 250
    x = 760
    h = 135
    w = 525
    crop_img = img[y:y + h, x:x + w]
    #cv2.imshow('Image', crop_img)
    cv2.imwrite('Crop_ProjectName.png', crop_img)
    cv2.waitKey(0)

    imageOne = 'C:/Users/mrmug/PycharmProjects/OCR/Crop_ProjectName.png'
    text = symbolConfidenc(imageOne)

    #imageOne = ImagePIL.open('C:/Users/mrmug/PycharmProjects/OCR/Crop_ProjectName.png')
    #text = tesserocr.image_to_text(imageOne)  # print ocr text from image
    print('Project Name: ' + text)
    return
def FinanceComments():
    # Gets Bounds of Line on DiverseSupplier by cropping
    y = 600
    x = 1760
    h = 825
    w = 550
    crop_img = img[y:y + h, x:x + w]
    #cv2.imshow('Image', crop_img)
    cv2.imwrite('Crop_FinanceComments.png', crop_img)
    cv2.waitKey(0)

    imageOne = 'C:/Users/mrmug/PycharmProjects/OCR/Crop_FinanceComments.png'
    text = symbolConfidenc(imageOne)

    #imageOne = ImagePIL.open('C:/Users/mrmug/PycharmProjects/OCR/Crop_FinanceComments.png')
    #text = tesserocr.image_to_text(imageOne)  # print ocr text from image
    print('Finance Comments : ' + text)
    return
def symbolConfidenc(img):
    word = ''
    count = 0
    insertSpace = 'false'
    with PyTessBaseAPI() as api:
        api.SetImageFile(img)
        api.Recognize()

        ri = api.GetIterator()
        #levelTwo = RIL.TEXTLINE
        level = RIL.WORD
        for r in iterate_level(ri, level):
            #space = r.GetUTF8Text(levelTwo)#gets whole line includes everything unlike RIL.SYMBOL
            symbol = r.GetUTF8Text(level)  # r == ri
            conf = r.Confidence(level)

            if conf > 50:
                word = word + ' ' + symbol
    return word


#ENHANCES TEXT
with Image(filename='C:/Users/mrmug/PycharmProjects/OCR/MillsHSkewedRight.png') as img:
    img.enhance()
    img.despeckle()
    img.transform_colorspace("gray")
    img.save(filename='C:/Users/mrmug/PycharmProjects/OCR/MillsHRestructured.png'.format(1))

#ERODES TEXT (1,1) & cleans
img = cv2.imread('C:/Users/mrmug/PycharmProjects/OCR/MillsHSkewedRight.png')
h, w, _ = img.shape # assumes color image
# Taking a matrix of size 5 as the kernel
kernel = np.ones((1,1), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
cv2.imwrite('MillsHSkewedRight.png',img_erosion)



img = deskew(img);
cv2.imwrite('MillsHDeSkewed.png', img)

#Crops image for relevant information
DiverseSupplier()
ProjectName()
FinanceComments()



















'''
#Text from img to text
imageOne = ImagePIL.open('C:/Users/mrmug/PycharmProjects/OCR/MillsHRestructured.png')
text = tesserocr.image_to_text(imageOne) # print ocr text from image
print(text)
textfile = open('C:/Users/mrmug/PycharmProjects/OCR/', 'w')
textfile.write(text)
textfile.close()

'''











