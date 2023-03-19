import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap=cmap)
    plt.show()

def createRgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

road = cv2.imread('../DATA/road_image.jpg')
roadCopy = np.copy(road)

# create empty space to draw too
markerImage = np.zeros(road.shape[:2], dtype=np.int32)
segments = np.zeros(road.shape, dtype=np.int8)

colors = []
for i in range(10):
    colors.append(createRgb(i))

print(colors)

"""Global variables"""
numMarkers = 10
currentMarker = 1
marksUpdated = False

"""Callback function"""
def mouseCallback(event,x,y,flags,param):
    global marksUpdated
    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passed to watershed algorithm
        cv2.circle(markerImage, (x,y), 10, (currentMarker), -1)

        # user sees on the road image
        cv2.circle(roadCopy, (x,y), 10, colors[currentMarker], -1)

        marksUpdated = True

"""While true"""
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouseCallback)

while True:
    cv2.imshow('Watershed Segments', segments)
    cv2.imshow('Road Image', roadCopy)

    # close all windows
    k = cv2.waitKey(1)
    if k == 27:
        break

    # clear all the colors presses 'c' key
    elif k == ord('c'):
        roadCopy = road.copy()
        markerImage = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.int8)

    # update color choice
    elif k > 0 and chr(k).isdigit():
        currentMarker = int(chr(k))

    # update the markings
    if marksUpdated:
        markerImageCopy = markerImage.copy()
        cv2.watershed(road,markerImageCopy)
        segments = np.zeros(road.shape, dtype=np.uint8)

        for colorIndex in range(numMarkers):
            segments[markerImageCopy == (colorIndex)] = colors[colorIndex]

cv2.destroyAllWindows()