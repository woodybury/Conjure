import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color I am looking for
    bgr = [0, 56, 43]
    #threshold
    thresh = 40
    # bgr to hsv
    hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

    minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    mask = cv2.inRange(hsvFrame, minHSV, maxHSV)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    median = cv2.medianBlur(res,15)
    cv2.imshow('Median Blur',median)

    # create small image (24x16) and edges
    smFrame = cv2.resize(mask, (24,16), interpolation = cv2.INTER_NEAREST)

    # small
    cv2.namedWindow('smColor',cv2.WINDOW_NORMAL)

    smColor = cv2.resize (smFrame, (480,320), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('smColor', smColor)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()