import cv2
import numpy as np
import connect

channel = connect.join()

cap = cv2.VideoCapture(0)

def colorMask (bgr, thresh, grey):
    # bgr to hsv
    hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

    minHSV = np.array([hsv[0] - (thresh / 2), hsv[1] - thresh, hsv[2] - thresh])
    maxHSV = np.array([hsv[0] + (thresh / 2), hsv[1] + thresh, hsv[2] + thresh])

    mask = cv2.inRange(hsvFrame, minHSV, maxHSV)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    grey_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # run a median blur
    median = cv2.medianBlur(grey_image,15)

    # create small image (24x16) for pixels
    pixels = cv2.resize(median, (24,16), interpolation = cv2.INTER_NEAREST)

    pixels[pixels > 0] = grey

    return pixels

while(1):

    # colors we are looking for
    bgr1 = [58, 66, 23]
    bgr2 = [96, 37, 27]

    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    original = cv2.resize (frame, (480,320))

    # color1
    color1 = colorMask(bgr1, 40, 195)
    # color2
    color2 = colorMask(bgr2, 40, 120)
    # total
    total = cv2.add( color1, color2)

    grandTotalRev = np.concatenate((total, total, total), axis=0)

    grandTotal = np.swapaxes(grandTotalRev, 0, 1 )

    cv2.imshow ('original', original)
    cv2.imshow ('grand total', grandTotal)

    grandTotalFlat = grandTotal.flatten()
    transform = grandTotalFlat

    count = 0
    transformSend = ""
    for ele in transform:
        transformSend+=(" "+str(ele))
        count += 1

    channel.push("input",{"body": transformSend})


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()