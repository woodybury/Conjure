import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color I am looking for
    bgr = [58, 66, 23]
    # bgr = [39, 33, 138]
    #threshold
    thresh = 50
    # bgr to hsv
    hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

    minHSV = np.array([hsv[0] - (thresh / 2), hsv[1] - thresh, hsv[2] - thresh])
    maxHSV = np.array([hsv[0] + (thresh / 2), hsv[1] + thresh, hsv[2] + thresh])

    mask = cv2.inRange(hsvFrame, minHSV, maxHSV)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # run a median blur
    median = cv2.medianBlur(mask,15)

    # create small image (24x16) for pixels
    smFrame = cv2.resize(median, (24,16), interpolation = cv2.INTER_NEAREST)

    # resize it so we can see
    pixels = cv2.resize (smFrame, (480,320), interpolation = cv2.INTER_NEAREST)
    # make other windows same size
    medianBlur = cv2.resize (median, (480,320))
    original = cv2.resize (frame, (480,320))

    cv2.imshow('pixels', pixels)
    cv2.imshow ('median', medianBlur)
    cv2.imshow ('original', original)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()