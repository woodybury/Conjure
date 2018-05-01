from imutils import face_utils
import numpy as np
import imutils
import dlib,time
import cv2

import connect

connected = True
try:
    channel = connect.join()
except ConnectionRefusedError:
    connected = False
    print ('failed to connect')


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# pick apart the face
# jaw = list(range(0, 17))
# brow_right = list(range(17, 22))
# brow_left = list(range(22, 27))
nose_down = list(range(27, 31))
nose_accross = list(range(31, 36))
eye_right = list(range(36, 42))
eye_left = list(range(42, 48))
mouth_out = list(range(48, 61))
mouth_in = list(range(61, 68))


def transform_image( image ):
    img = cv2.imread(image, 0)
    # resize img for transform
    img = cv2.resize(img, (16,24), interpolation = cv2.INTER_NEAREST)

    return img

def paint_line(image, landmarks, color, weight, offset=0):
    pos_prev = None
    for point in landmarks:
        pos=(point[0,0],point[0,1] + offset)
        if pos_prev:
            cv2.line(image,pos_prev,pos,color,weight)
        pos_prev = pos

def paint_circle(image, landmark, color, radius, weight, offset=0):
    cv2.circle(image, (landmark[0,0],landmark[0,1] - offset), radius, color, weight)


# 'centered' creates the transform size rectangle around the center of the nose. The rectangle uses a scalar that is the distance between the eyes
# 'bounded' uses the dlib rectangle from the detecting faces and maps it to transform size without loosing face aspect ratio
def crop_aspect (image, landmarks, type='center', rect=None):
    try:
        if type == 'center':
            scalar = landmarks[44][0,0] - landmarks[37][0,0]

            aspect_x = int(landmarks[33][0,0] - (scalar*(2/3)))
            aspect_width = int(landmarks[33][0,0] + (scalar*(2/3)))
            aspect_y = int(landmarks[33][0,1] - scalar)
            aspect_height = int(landmarks[33][0,1] + scalar)

        if type == 'bounded':
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            aspect_width = x + int(h*(4/6) + w/5)
            aspect_x = x + int(w/5)

            aspect_height = y + h
            aspect_y = y

        return image[ aspect_y:aspect_height, aspect_x: aspect_width ]
    except ValueError:
        return transform_image('img/tm.jpg')


def transform(image_1, image_2, image_3):

    trans_image = np.concatenate((image_1, image_2, image_3), axis=1)

    # for dev
    lg_image = cv2.resize(trans_image, (960,480), interpolation = cv2.INTER_NEAREST)
    cv2.imshow("Large", lg_image)

    if connected:
        # flatten array
        flat_image = trans_image.flatten()

        # stringify for server
        transform_send = ""
        for ele in flat_image:
            transform_send+=(" "+str(ele))

        # if you want to look at the numbers :)
        # print (transform_send)
        # uncomment this to send to server
        channel.push("input",{"body": transform_send})


def facial_landmark_stuff (rect, gray, h, w):
    blank_image = np.zeros((h,w,1), np.uint8)

    # determine the facial landmarks for the face region, then
    detected_landmarks = predictor(gray, rect).parts()
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

    weight_scalar = (landmarks[44][0,0] - landmarks[37][0,0])
    nose_offset = int((landmarks[30][0,1] - landmarks[32][0,1])/2.0)

    if weight_scalar > 30:
        #nose
        paint_line (blank_image, landmarks[nose_down], 155, 2)
        paint_line (blank_image, landmarks[nose_accross], 155, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 155, 3, 3, nose_offset)
        # eyes
        paint_line (blank_image, landmarks[eye_left], 145, 2)
        paint_line (blank_image, landmarks[eye_right], 145, 2)
        # mouth
        paint_line (blank_image, landmarks[mouth_out], 155, 2)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)
    elif weight_scalar > 15:
        #nose
        paint_line (blank_image, landmarks[nose_down], 200, 1)
        paint_line (blank_image, landmarks[nose_accross], 200, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 175, 2, 2, nose_offset)
        # eyes
        paint_line (blank_image, landmarks[eye_left], 190, 1)
        paint_line (blank_image, landmarks[eye_right], 190, 1)
        # mouth
        paint_line (blank_image, landmarks[mouth_out], 200, 1)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)

    else:
        print ('small')
        #nose
        paint_line (blank_image, landmarks[nose_down], 175, 1)
        paint_line (blank_image, landmarks[nose_accross], 175, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 150, 1, 1, nose_offset)
        # eyes
        paint_line (blank_image, landmarks[eye_left], 165, 1)
        paint_line (blank_image, landmarks[eye_right], 165, 1)
        # mouth
        paint_line (blank_image, landmarks[mouth_out], 175, 1)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)

    # use my helper functions to track and crop face using either the 'center' method or the 'bounded' method.
    # 'centered' creates the transform size rectangle around the center of the nose. The rectangle uses a scalar that is the distance between the eyes
    # 'bounded' uses the dlib rectangle from the detecting faces and maps it to transform while maintaining face aspect ratio
    crop_image = crop_aspect(blank_image, landmarks, 'center', rect)

    # resize for transform using INTER_AREA => interpolation using pixel area relation
    trans_image = cv2.resize(crop_image, (16,24), interpolation = cv2.INTER_AREA)

    # various useful blurs
    '''
    blur = cv2.medianBlur(trans_image,1)
    blur = cv2.bilateralFilter(trans_image,5,75,75)
    blur = cv2.GaussianBlur(trans_image,(5,5),0)
    kernel = np.ones((2,2),np.float32)/4
    blur = cv2.filter2D(trans_image,-1,kernel)
    '''

    return trans_image


def faceform():
    cap= cv2.VideoCapture(0)
    time.sleep(1)

    empty_screen = transform_image('img/tm.jpg')
    # empty_screen = np.zeros((24,16), np.uint8)

    while (1):
        ret, image = cap.read()
        if ret:
            # load the image, resize it (helps with speed!), and convert it to grayscale
            image = imutils.resize(image, width=400)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]

            # detect faces in the grayscale image
            rects = detector(gray, 1)

            face_number = len(rects)

            # loop over the face detections
            if face_number != 0:
                if face_number == 1:
                    screen_1 = facial_landmark_stuff(rects[0], gray, h, w)
                    transform(screen_1, empty_screen, empty_screen)
                elif face_number == 2:
                    screen_1 = facial_landmark_stuff(rects[0], gray, h, w)
                    screen_2 = facial_landmark_stuff(rects[1], gray, h, w)
                    transform(screen_1, screen_2, empty_screen)
                else:
                    screen_1 = facial_landmark_stuff(rects[0], gray, h, w)
                    screen_2 = facial_landmark_stuff(rects[1], gray, h, w)
                    screen_3 = facial_landmark_stuff(rects[2], gray, h, w)
                    transform(screen_1, screen_2, screen_3)


            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    faceform()