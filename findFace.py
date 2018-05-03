import numpy as np
import cv2
import time
import imutils

# cascade for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def lookforfaces():
    # pi cam
    cap = cv2.VideoCapture(0)
    num = 0
    blurthresh = 60
    changethresh = 5000000
    firstFrame = None

    while (1):
        # frames
        ret, frame = cap.read()

        # detect faces in video
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # create median blur for diff test
        # median = cv2.medianBlur(gray,15)
        # find faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        blank = ()
        num += 1
        if faces != blank:
            print ('face')
            # test blur
            '''
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur > blurthresh:
                print ('blur face')
            else:
                print ('good face')

                if firstFrame is None:
                    firstFrame = median
                    continue
                # test change
                delta = cv2.absdiff(firstFrame, median).sum()

                if delta > changethresh:
                    print('good different face')
                    cv2.imwrite('img/face/face'+str(num)+'.jpg',frame)
            
            '''
        # time.sleep(2)

    # release capture
    cap.release()
    cv2.destroyAllWindows()

lookforfaces()