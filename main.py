import argparse
import datetime
import cv2
import dlib
import math
import time

from utils import put_text
from imutils import face_utils
#from action import translate
from detection import detect_head, detect_eyes
from action import HeadAction, EyeAction
#constants
CALIBRATE = True


""" Main """

# facial detection and real time processing credit goes to:
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
# it's ok to use this since its open source and we are allowed to use facial detection packages

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="Path to facial landmark predictor")
args = vars(ap.parse_args())

def main():

    # uncomment here as well as in below if u wanna run opencv's face detection / eye detection code
    """
    # load eye and face classifiers
    face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_eye.xml')
    right_eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml')
    left_eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.1/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml')
    if face_cascade.empty() or eye_cascade.empty() or right_eye_cascade.empty() or left_eye_cascade.empty():
        print("[ERROR] could not load xml, make sure path to xml is correct")
    """



    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    camera = cv2.VideoCapture(0)

    # frame counter stored counts of how many frames have passed for a certain detection
    frame_counters = {
        "both_eyes_closed" : 0,
        "left_blink" : 0,
        "right_blink" : 0    
    }

    # eye status counter
    total_blinks = 0
    total_left_winks = 0
    total_right_winks = 0

    # head state counter
    total_left = 0
    total_right = 0
    total_up = 0
    total_down = 0
    total_zoomed = 0

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        h, w, _ = frame.shape

        resize_ratio = 800 / w
        frame = cv2.resize(frame,  (0,0), fx=resize_ratio, fy=resize_ratio)
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # optionally instruct user to position head in specific spot
            if CALIBRATE:
                line_height = int(h*0.7)
                align_x = int(w/2-100)
                cv2.line(frame, (int(w/2-30), line_height), (int(w/2+30), line_height), (0, 0, 0), 2)
                put_text(frame, "make chin tangent to black line", (align_x, line_height + 20))
                put_text(frame, "make hair tangent to top of frame", (align_x, line_height + 2*20))
                put_text(frame, "for best results", (align_x, line_height + 3*20))

            # get the status of areas we are interested in
            head_state, zoom_state = detect_head(shape, frame)
            eye_status = detect_eyes(shape, frame, frame_counters)

            # display decisions
            print("=================================")
            align_x = int(w*0.1)
            align_y = int(h*0.1)
            print(head_state)
            print(zoom_state)
            print(eye_status)
            put_text(frame, str(head_state)[11:], (align_x, align_y + 30), scale=0.5, thickness=2)
            put_text(frame, str(zoom_state)[11:], (align_x, align_y + 2*30), scale=0.5, thickness=2)
            put_text(frame, str(eye_status)[10:], (align_x, align_y + 3*30), scale=0.5, thickness=2)


            # record and display count of blinks and winks for accuracy purposes
            if eye_status == EyeAction.BOTH_BLINK:
                total_blinks += 1
            if eye_status == EyeAction.LEFT_WINK:
                total_left_winks += 1
            if eye_status == EyeAction.RIGHT_WINK:
                total_right_winks += 1
            if head_state == HeadAction.LEFT:
                total_left += 1
            if head_state == HeadAction.RIGHT:
                total_right += 1
            if head_state == HeadAction.UP:
                total_up += 1
            if head_state == HeadAction.DOWN:
                total_down += 1
            if zoom_state == HeadAction.ZOOMED:
                total_zoomed += 1     

            align_x = int(w*0.68)
            put_text(frame, "      Total", (align_x, 20))
            put_text(frame, "     blinks: " + str(total_blinks), (align_x, 40))
            put_text(frame, " left winks: " + str(total_left_winks), (align_x, 55))
            put_text(frame, "right winks: " + str(total_right_winks), (align_x, 70))
            #put_text(frame, "       left: " + str(total_left), (align_x, 85))
            #put_text(frame, "      right: " + str(total_right), (align_x, 100))
            #put_text(frame, "         up: " + str(total_up), (align_x, 115))
            #put_text(frame, "       down: " + str(total_down), (align_x, 130))
            #put_text(frame, "      zoomed: " + str(total_zoomed), (align_x, 145))

        """
        # trying out open cv's eye detection
        # source: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #print("eyes: ", eyes)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

            right_eye = right_eye_cascade.detectMultiScale(roi_gray)
            print("right_eye: ", right_eye)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
            #ex,ey,ew,eh = right_eye[0]
            #cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

            left_eye =  left_eye_cascade.detectMultiScale(roi_gray)
        """


        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
