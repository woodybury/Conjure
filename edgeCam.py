# OpenCV program to perform Edge detection in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2

# np is an alias pointing to numpy library
import numpy as np


# capture frames from a camera
cap = cv2.VideoCapture(0)


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

    # Display an original image
    #cv2.imshow('Original',frame)

    # finds edges in the input image image and
    # marks them in the output map edges
    lgEdges = cv2.Canny(frame,100,200)

    # create small image (24x16) and edges
    smFrame = cv2.resize(frame, (24,16))
    smEdges = cv2.Canny(smFrame,400,500)

    # resize windows
    # large
    cv2.namedWindow('lgEdges',cv2.WINDOW_NORMAL)
    lgEdges = cv2.resize (lgEdges, (480,320))
    # small
    cv2.namedWindow('smEdges',cv2.WINDOW_NORMAL)
    smEdges = cv2.resize (smEdges, (480,320))

    # Display edges in a frame
    # large
    cv2.imshow('lgEdges', lgEdges)
    # small
    cv2.imshow('smEdges', smEdges)


    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()