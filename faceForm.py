import cv2
import numpy as np
import dlib,time

cap= cv2.VideoCapture(0)
time.sleep(2)

# cascade
faceCascade =  cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

# landmark predictor
predictor =  dlib.shape_predictor('xml/shape_predictor_68_face_landmarks.dat')

while (1):

    ret, image = cap.read()
    if ret:
        image = cv2.flip(image, 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        h, w = image.shape[:2]
        blank_image = np.zeros((h,w,3), np.uint8)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = predictor(image, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            posPrev = (0,0)
            for idx,point in enumerate(landmarks):
                pos=(point[0,0],point[0,1])

                cv2.circle(blank_image,pos,2,color=(255,255,255),thickness=30)

                lineLength = abs(posPrev[0] - pos[0]) + abs(posPrev[1] - pos[1])
                if lineLength < 75:
                    cv2.line(blank_image,posPrev,pos,(255,255,255),30)

                posPrev = pos


        sm_image = cv2.resize(blank_image, (24,16), interpolation = cv2.INTER_NEAREST)

        cv2.imshow('large image',blank_image)
        cv2.imshow('small image',sm_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()