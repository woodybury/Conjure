import cv2
import numpy as np
import dlib,time
from threading import Thread
from occamy import Socket
from listening import recognition


# global is paused? not good practice but w/e
ispaused = False

def connect():
    socket = Socket("ws://dlevs.me:4000/socket")
    socket.connect()

    channel = socket.channel("room:lobby", {})
    channel.on("connect", print ('Im in'))
    channel.on("new_msg", lambda msg, x: print("> {}".format(msg["body"])))

    channel.join()

def faceform():

    cap= cv2.VideoCapture(0)
    time.sleep(2)

    # cascade
    faceCascade =  cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # landmark predictor
    predictor =  dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

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
            blank_image = np.zeros((h,w,1), np.uint8)

            '''
            for (x, y, w, h) in faces:
                blank_image = blank_image[y:y+h,x:x+w]
                gray = gray[y:y+h, x:x+w]
            '''

            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detected_landmarks = predictor(image, dlib_rect).parts()

                landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

                landmarks_each = landmarks[mouthOut + eyeR + eyeL + nose]

                posPrev = (0,0)
                for i,point in enumerate(landmarks_each):
                    pos=(point[0,0],point[0,1])
                    # cv2.circle(blank_image,pos,2,color=255,thickness=10)
                    # wish python had a switch
                    if i in range(1,13):
                        cv2.line(blank_image,posPrev,pos,135,10)
                    if i in range(14,19):
                        cv2.line(blank_image,posPrev,pos,145,10)
                    if i in range(20,25):
                        cv2.line(blank_image,posPrev,pos,145,10)
                    if i in range(26,28):
                        cv2.line(blank_image,posPrev,pos,155,10)
                    if i in range(28,29):
                        cv2.line(blank_image,posPrev,pos,185,10)
                    posPrev = pos

            blank_image = blank_image[240:480, 360:840]

            sm_image = cv2.resize(blank_image, (48,24), interpolation = cv2.INTER_NEAREST)
            # lg_image = cv2.resize(sm_image, (480,240), interpolation = cv2.INTER_NEAREST)
            # cv2.imshow('large',lg_image)

        # flatten array
        sm_image = sm_image.flatten()

        # stringify for server
        transformSend = ""
        for ele in sm_image:
            transformSend+=(" "+str(ele))

        if not ispaused:
            # if you want to look at the numbers :)
            print (transformSend)
            # uncomment this to send to server
            # channel.push("input",{"body": transformSend})

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def pause():
    global ispaused
    ispaused = True
    print ('stop')

def start():
    global ispaused
    ispaused = False
    print ('stop')

def voicecontrolstop ():
    recognition(pause, 'stop')
def voicecontrolstart ():
    recognition(start, 'start')

if __name__ == "__main__":

    #connect()

    t1 = Thread(target = voicecontrolstop)
    t2 = Thread(target = voicecontrolstart)
    t3 = Thread(target = faceform)

    t1.start()
    time.sleep(2)
    t2.start()
    time.sleep(2)
    t3.start()