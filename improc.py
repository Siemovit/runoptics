import cv2
import imutils
import skimage
from skimage import measure
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt

img = cv2.imread('vis3.JPG',0)
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

for label in np.unique(labels):
	if label == 0:
		continue
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	if numPixels > 500:
		mask = cv2.add(mask, labelMask)

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.rectangle(img, (int(cX-w/2) , int(cY-h/2)), (int(cX+w/2) , int(cY+h/2)),0, 2)

titles = ['Original Image','BINARY','MASK']

images = [img,thresh,mask]


for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])


if __name__ == '__main__':
    plt.show()
