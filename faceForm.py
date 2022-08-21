import argparse
# pip install imutils
from imutils import face_utils
import numpy as np
import imutils
# pip install dlib
import dlib,time
# pip install opencv-python
import cv2
# pip install https://github.com/jhosteny/occamy/zipball/master
from occamy import Socket

# initialize dlib HOG-based face detector
detector = dlib.get_frontal_face_detector()
# initialize opencv face cascades - xml in repo data dir
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# create facial landmark predictor - dat in repo data dir
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# pick landmarks from face
nose_down = list(range(27, 31))
nose_across = list(range(31, 36))
eye_right = list(range(36, 42))
eye_left = list(range(42, 48))
mouth_out = list(range(48, 61))
mouth_in = list(range(61, 68))
'''
Other available landmarks:
jaw = list(range(0, 17))
brow_right = list(range(17, 22))
brow_left = list(range(22, 27))
'''

# img resize for transform screens (16/24)
def transform_image( image ):
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (16,24), interpolation = cv2.INTER_NEAREST)
    return img

# paint lines on image from list of landmarks points
def paint_line(image, landmarks, color, weight, offset=0):
    pos_prev = None
    for point in landmarks:
        pos=(point[0,0],point[0,1] + offset)
        if pos_prev:
            cv2.line(image,pos_prev,pos,color,weight)
        pos_prev = pos

# paint circles on image from list of landmarks points
def paint_circle(image, landmark, color, radius, weight, offset=0):
    cv2.circle(image, (landmark[0,0],landmark[0,1] - offset), radius, color, weight)

# extract facial landmark features from image
def get_features(rect, gray, sm_scale, h, w):
    (rect_x,rect_y,rect_w,rect_h) = rect
    screen_rect = dlib.rectangle(rect_x*sm_scale, rect_y*sm_scale, (rect_x + rect_w)*sm_scale, (rect_y + rect_h)*sm_scale)
    screen_stuff = facial_landmark_predictor_painter(screen_rect, gray, h, w)
    return screen_stuff


'''
'centered' creates the transform size rectangle around the center of the nose.
The rectangle uses a scalar that is the distance between the eyes
'bounded' uses the dlib rectangle from the detecting faces and maps it to
transform size without loosing face aspect ratio
'''
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
    except cv2.error:
        print ('crop error')
        return transform_image('img/tm.jpg')

# concatenate the 3 screens, flatten and send to transform if connected
def transform_send(screens, empty_screen, connected, streaming, monitor):
    screens_count = len(screens)
    '''
    Displaying up to 3 faces on transform, if less than three faces are detected
    use the tangible group logo as placeholder. First face detected is in middle position.
    '''
    if screens_count == 1:
        screen_1 = empty_screen
        screen_2 = screens[0]
        screen_3 = empty_screen
    elif screens_count == 2:
        screen_1 = empty_screen
        screen_2 = screens[0]
        screen_3 = screens[1]
    else:
        screen_1 = screens[2]
        screen_2 = screens[0]
        screen_3 = screens[1]

    total_image = np.concatenate((screen_1, screen_2, screen_3), axis=1)

    if monitor:
        # for dev you can watch the transform size video here
        demo_image = cv2.resize(total_image, (960,480), interpolation = cv2.INTER_NEAREST)
        cv2.imshow("Large", demo_image)

    if connected and streaming:
        # flatten array
        flat_image = total_image.flatten()

        # stringify for server
        transform_data = ""
        for ele in flat_image:
            transform_data+=(" "+str(ele))

        # if you want to look at the matrix :)
        # print (transform_data)
        channel.push("input",{"body": transform_data})

# run the landmark model and paint based on b/w eye scalar
def facial_landmark_predictor_painter (rect, gray, h, w):

    # create same size blank img for painting canvas
    blank_image = np.zeros((h,w,1), np.uint8)

    # determine the facial landmarks for the face region
    detected_landmarks = predictor(gray, rect).parts()
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

    weight_scalar = (landmarks[44][0,0] - landmarks[37][0,0])
    nose_offset = int((landmarks[30][0,1] - landmarks[32][0,1])/2.0)

    # three thresholds for weight and contrast
    if weight_scalar > 30:
        #nose
        paint_line(blank_image, landmarks[nose_down], 155, 2)
        paint_line(blank_image, landmarks[nose_across], 155, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 155, 3, 3, nose_offset)
        # eyes
        paint_line(blank_image, landmarks[eye_left], 145, 2)
        paint_line(blank_image, landmarks[eye_right], 145, 2)
        # mouth
        paint_line(blank_image, landmarks[mouth_out], 155, 2)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)
    elif weight_scalar > 15:
        #nose
        paint_line(blank_image, landmarks[nose_down], 200, 1)
        paint_line(blank_image, landmarks[nose_across], 200, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 175, 2, 2, nose_offset)
        # eyes
        paint_line(blank_image, landmarks[eye_left], 190, 1)
        paint_line(blank_image, landmarks[eye_right], 190, 1)
        # mouth
        paint_line(blank_image, landmarks[mouth_out], 200, 1)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)

    else:
        #nose
        paint_line(blank_image, landmarks[nose_down], 175, 1)
        paint_line(blank_image, landmarks[nose_across], 175, 1, nose_offset )
        paint_circle(blank_image, landmarks[30], 150, 1, 1, nose_offset)
        # eyes
        paint_line(blank_image, landmarks[eye_left], 165, 1)
        paint_line(blank_image, landmarks[eye_right], 165, 1)
        # mouth
        paint_line(blank_image, landmarks[mouth_out], 175, 1)
        # paint_line (blank_image, landmarks[mouth_in], 155, r2)

    try:
        '''
        use helper function to track and crop face using either the 'center' method or the 'bounded' method.
        'centered' creates the transform size rectangle around the center of the nose. The rectangle uses a scalar that is the distance between the eyes
        'bounded' uses the dlib rectangle from the detecting faces and maps it to transform while maintaining face aspect ratio
        '''
        crop_image = crop_aspect(blank_image, landmarks, 'center', rect)
        # resize for transform using INTER_AREA => interpolation using pixel area relation
        trans_image = cv2.resize(crop_image, (16,24), interpolation = cv2.INTER_AREA)
    except cv2.error:
        print ('crop error')
        trans_image = transform_image('img/tm.jpg')

    # various other useful blurs
    '''
    blur = cv2.medianBlur(trans_image,1)
    blur = cv2.bilateralFilter(trans_image,5,75,75)
    blur = cv2.GaussianBlur(trans_image,(5,5),0)
    kernel = np.ones((2,2),np.float32)/4
    blur = cv2.filter2D(trans_image,-1,kernel)
    '''
    return [trans_image, landmarks]


def face_form(face_detector, sm_scale, sm_width, connected, monitor):
    streaming = connected # toggle sending to transform
    run = True # kill flag
    cap = cv2.VideoCapture(0)
    time.sleep(1)

    # using TMG logo for empty states
    empty_screen = transform_image('img/tm.jpg')

    count = 0
    rects = None

    while (run):
        ret, image = cap.read()
        if ret:
            # load the image, resize it (helps with speed!), and convert it to grayscale
            image = imutils.resize(image, width=sm_width)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            monitor_landmarks = []

            # get new h and w
            h, w = gray.shape[:2]

            # scale for face detection
            sm_image = imutils.resize(gray, width=int(w/sm_scale))

            # detect faces in the scaled grayscale image every nth frame (for speed)
            count = count + 1
            if count % 5:
                if face_detector == 'cv2':
                    rects = face_cascade.detectMultiScale(sm_image, 1.3, 5)
                elif face_detector == 'dlib':
                    rects = detector(sm_image, 1)
            # using cv2 cascades
            if face_detector == 'cv2' and len(rects) != 0 or face_detector == 'dlib' and rects:
                screens = []
                for i, rect in enumerate(rects):
                    if i < 3: # max three faces can be displayed on transform
                        if face_detector == 'cv2':
                            screen = get_features(rect, gray, sm_scale, h, w)
                            monitor_landmarks.append(screen[1])
                            screens.append(screen[0])
                        # for using dlib HOG-based face detector (slower but more accurate?)
                        elif face_detector == 'dlib':
                            (rect_x, rect_y, rect_w, rect_h) = face_utils.rect_to_bb(rects[0])
                            screen_rect = dlib.rectangle(rect_x*sm_scale, rect_y*sm_scale, (rect_x + rect_w)*sm_scale, (rect_y + rect_h)*sm_scale)
                            screen = facial_landmark_predictor_painter(screen_rect, gray, h, w)
                            monitor_landmarks.append(screen[1])
                            screens.append(screen[0])
                # send to transform
                transform_send(screens, empty_screen, connected, streaming, monitor)

            if monitor:
                # paint facial features on original video for demo
                for i, landmarks in enumerate(monitor_landmarks):
                    if i == 0:
                        color = (255,0,0)
                    elif i == 1:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    paint_line (image, landmarks[nose_down], color, 2)
                    paint_line (image, landmarks[nose_across], color, 2)
                    paint_circle(image, landmarks[30], color, 1, 1)
                    # eyes
                    paint_line (image, landmarks[eye_left], color, 2)
                    paint_line (image, landmarks[eye_right], color, 2)
                    # mouth
                    paint_line (image, landmarks[mouth_out], color, 2)

                # show original image w/ facial features for demo
                demo_image = cv2.resize(image, (1110,int((1110/w)*h)))
                cv2.imshow('original', demo_image)

            k = cv2.waitKey(5) & 0xF
            # start / stop streaming to transform using the space bar
            if k == 0: # space bar
                if streaming:
                    print ('stopped streaming to shape display')
                    streaming = False
                else:
                    print ('started streaming to shape display')
                    streaming = True
            if k == 11: # escape
                run = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    Init the faceform app. You can user 'cv2' or 'dlib' for the initial finding of faces. I think 'cv2' is faster but 'dlib' may be more accurate.
    second arg is for scaling the image for finding faces and third arg is for resizing the image for tracking facial features. Both effect speed, accuracy, and distance.
    For exmaple, faceform('cv2', 2, 800) will use Open CV's harr cascades to find faces, and will scale the face finding image to 400 width (800/2).
    Then within the coordinates of a found face it will track facial features using the 800 width image.

    CLI: e.g. python faceForm.py -f cv2 -s 1 -w 700 -ws 'ws://dlevs.me:4000/socket' -m yes
    '''
    parser = argparse.ArgumentParser(
        description='face form app'
    )
    parser.add_argument('-f','--face_detector', help="'cv2' or 'dlib' for the initial finding of faces", type=str, default='cv2')
    parser.add_argument('-s','--img_scale', help='scale of img', type=int, default=1)
    parser.add_argument('-w','--img_width', help='width of img', type=int, default=700)
    parser.add_argument('-ws','--websocket', help='websocket address', type=str, default='ws://dlevs.me:4000/socket')
    parser.add_argument('-m','--monitor', help='visual monitor', type=bool, default=False)
    args = parser.parse_args()

    def onConnect():
        print('connected')
        print ('starting face form')
        print ('press space bar to pause stream')
        print ('press escape to close program')
        face_form(args.face_detector, args.img_scale, args.img_width, True, args.monitor)
    try:
        socket = Socket(args.websocket)
        socket.connect()
        channel = socket.channel("room:lobby", {})
        channel.join()
        channel.on("connect", onConnect)
    except:
        # TODO retry logic
        print ('failed to connect')
        print ('starting face form in monitor mode')
        print ('press escape to exit program')
        face_form(args.face_detector, args.img_scale, args.img_width, False, args.monitor)