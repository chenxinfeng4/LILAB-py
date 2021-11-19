import sys

import cv2  # Opencv ver 3.1.0 used
import numpy as np

# Set recursion limit
sys.setrecursionlimit(10 ** 5)

from opencvdragrect import selectinwindow

# paras
videofile = r"E:\cxf\2021-06-03 15-14-45 sub1p3.mp4"
MAXWIDTH, MAXHEIGHT = 1024, 800

cap = cv2.VideoCapture(videofile)
ret, frame = cap.read()
assert ret
cap.release()
frameHeight, frameWidth, _ = frame.shape
scale = max([frameWidth/MAXWIDTH, frameHeight/MAXHEIGHT, 1])



# Initialize the  drag object
wName = "select region"
imageWidth = int(frameWidth / scale)
imageHeight = int(frameHeight / scale)
image = frame

# Define the drag object
rectI = selectinwindow.DragRectangle(image, wName, frameWidth, frameHeight)

cv2.namedWindow(rectI.wname)
cv2.resizeWindow(rectI.wname, imageWidth, imageHeight);
cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)

# keep looping until rectangle finalized
while True:
    # display the image
    cv2.imshow(rectI.wname, rectI.image)
    if cv2.waitKey(1) in (ord('q'), 27): break
    # if returnflag is True, break from the loop
    if rectI.returnflag: break

print("Dragged rectangle coordinates")
print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
      str(rectI.outRect.w) + ',' + str(rectI.outRect.h))

# close all open windows
cv2.destroyAllWindows()

