import numpy as np
import cv2
import time

# cascade for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def lookforfaces():
    # pi cam
    cap = cv2.VideoCapture(0)

    num = 0
    while (1):
        # frames
        ret, frame = cap.read()

        # detect faces in video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        blank = ()
        num += 1
        if faces != blank:
                cv2.imwrite('img/face/face'+str(num)+'.jpg',frame)

    # release capture
    cap.release()
    cv2.destroyAllWindows()

lookforfaces()