import cv2
import numpy as np
from occamy import Socket

# socket = Socket("ws://dlevs.me:4000/socket")
# socket.connect()

# channel = socket.channel("room:lobby", {})
# channel.on("connect", print ('Im in'))
# channel.on("new_msg", lambda msg, x: print("> {}".format(msg["body"])))

# channel.join()

# capture frames from a camera
cap = cv2.VideoCapture(0)


# auto canny using median
def auto_canny(vid, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(vid)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(vid, lower, upper)

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # return the edged image
    return edged


# loop runs if capturing has been initialized
while(1):

    # reads frames from a camera
    ret, frame = cap.read()

    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # filter
    filter = cv2.GaussianBlur(frame, (15,15), 0)

    # finds edges in the input image image and
    # marks them in the output map edges
    lgEdges = auto_canny(filter)

    # create small image (24x16) and edges
    smFrame = cv2.resize(filter, (24,16), interpolation = cv2.INTER_NEAREST)
    smEdges = auto_canny(smFrame)

    # filter
    lgFilter = cv2.medianBlur(lgEdges,1)
    smFilter = cv2.medianBlur(smEdges,15)

    # resize windows
    # large
    lgEdges = cv2.resize (lgEdges, (480,320))
    # original
    original = cv2.resize (frame, (480,320))

    # Display edges in a frame
    # large
    cv2.imshow('lgEdges', lgEdges)
    # small
    cv2.imshow('smEdges', smEdges)
    # original
    cv2.imshow('Original',original)

    grandTotalRev = np.concatenate((smEdges, smEdges, smEdges), axis=0)

    grandTotal = np.swapaxes(grandTotalRev, 0, 1 )

    grandTotalFlat = grandTotal.flatten()
    transform = grandTotalFlat

    count = 0
    transformSend = ""
    for ele in transform:
        transformSend+=(" "+str(ele))
        count += 1

    # channel.push("input",{"body": transformSend})


    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()