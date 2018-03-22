import numpy as np
import cv2
import time

face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('data/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        smile = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=22, minSize=(25, 25))
        # find biggest face
        maxSmile = 0
        bigSmile = None
        for s in smile:
            if s[2] * s[3] > maxSmile:
                maxSmile = s[2] * s[3]
                bigSmile = s
        if bigSmile is None:
            print ('frown')
        else:
            # cv2.rectangle(roi_img, (bigSmile[0], bigSmile[1]), (bigSmile[0] + bigSmile[2], bigSmile[1] + bigSmile[3]), (0, 255, 0), 1)
            sm_ratio = round(bigSmile[2] / bigSmile[0], 3)
            if sm_ratio>2:
                print('smile')
            else:
                print ('neutral')

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()