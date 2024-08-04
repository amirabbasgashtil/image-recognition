import pytesseract
from pytesseract import Output
import PIL.Image
import cv2

"""
Page segmentation modes:
 0  Orientaion and script detection (OSD) only.
 1  Automatic page segmentation with  OSD.
 2  Automatic page segmentation, but no OSD, or OCR.
 3  Fully automatic page segmentation, but no OSD. (Default)
 4  Assume a single column of text of variable size.
 5  Assume a single uniform block of vertically aligned text.
 6  Assume a single uniform block of text.
 7  Treat the image as a single text line.
 8  Treat the image as a single word
 9  Treat the image as a single word in circle.
 10 Treat the image as a single character.
 11 Sparse text. Find as much text as possible in no particular order.
 12 Sparse text with OSD.
 13 raw line. Treat the image as a single text line,
                        bypassing hacks that are Tesseract-specific.
"""

"""
OCR Engine Mode
 0   Legacy engine only
 1   Neural nets LSTM engine only
 2   Legacy + LSTM engines
 3   Default, based on what is available
"""
myconfig = r"--psm 6 --oem 3"


text = pytesseract.image_to_string(PIL.Image.open("image.jpg"), config=myconfig)
print(text)

img = cv2.imread("image.jpg")
height, width, _ = img.shape

'''
boxes = pytesseract.image_to_boxes(img, config=myconfig)
for box in boxes.splitlines():
    box = box.split(" ")
    img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0))'''

data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
print(data['conf'])
for d in data:
    if d['conf'] > 80:
        print(d['text'])
        
'''
print(data['text'])
'''

print(data['conf'])

'''
cv2.imshow('image', img)
cv2.waitKey(0)'''