import cv2
import numpy as np
import dlib,time

cap= cv2.VideoCapture(0)
time.sleep(2)

# cascade
faceCascade =  cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

# landmark predictor
predictor =  dlib.shape_predictor('xml/shape_predictor_68_face_landmarks.dat')

# pick apart the face
jaw = list(range(0, 17))
browR = list(range(17, 22))
browL = list(range(22, 27))
nose = list(range(27, 36))
eyeR = list(range(36, 42))
eyeL = list(range(42, 48))
mouthOut = list(range(48, 61))
mouthIn = list(range(61, 68))

while (1):

    ret, image = cap.read()
    if ret:
        image = cv2.flip(image, 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        h, w = image.shape[:2]
        blank_image = np.zeros((h,w,3), np.uint8)

        for (x, y, w, h) in faces:
            cv2.rectangle(blank_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            blank_image = blank_image[y:y+h, x:x+w]
            image = image[y:y+h, x:x+w]

        faces = faceCascade.detectMultiScale(image, 1.3, 5)

        for (x, y, w, h) in faces:

            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = predictor(image, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

            landmarks_each = landmarks[mouthOut + eyeR + eyeL + nose]

            posPrev = (0,0)
            for idx,point in enumerate(landmarks_each):
                pos=(point[0,0],point[0,1])

                cv2.circle(blank_image,pos,2,color=(255,255,255),thickness=5)

                lineLength = abs(posPrev[0] - pos[0]) + abs(posPrev[1] - pos[1])
                if lineLength < 50:
                    cv2.line(blank_image,posPrev,pos,(255,255,255),20)

                posPrev = pos

        lg_image = cv2.resize(blank_image, (240,160), interpolation = cv2.INTER_NEAREST)

        sm_image = cv2.resize(blank_image, (24,16), interpolation = cv2.INTER_NEAREST)

        cv2.imshow('large',lg_image)
        cv2.imshow('small',sm_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()