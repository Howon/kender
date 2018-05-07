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

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        h, w, _ = frame.shape

        resize_ratio = 400 / w
        frame = cv2.resize(frame,  (0,0), fx=resize_ratio, fy=resize_ratio)
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
                cv2.line(frame, (177, 175), (237, 175), (0, 0, 0), 2)
                put_text(frame, "make chin tangent to black line", (75, 190))
                put_text(frame, "make hair tangent to top of frame", (75, 205))
                put_text(frame, "for best results", (75, 220))

            # get the status of areas we are interested in
            head_state, zoom_state = detect_head(shape, frame)
            eye_status = detect_eyes(shape, frame, frame_counters)

            # display decisions
            print("=================================")
            print(head_state)
            print(zoom_state)
            print(eye_status)
            put_text(frame, str(head_state)[11:], (30, 30), scale=0.5, thickness=2)
            put_text(frame, str(zoom_state)[11:], (30, 60), scale=0.5, thickness=2)
            put_text(frame, str(eye_status)[10:], (30, 90), scale=0.5, thickness=2)


            # translate_head(head_status)
            # translate_zoom(head_status)
            # translate_eye(head_status)
            #
            # if head_status == "RIGHT":
                # move window to the right
            # if eye_status = "EYES BLINKED"
                # do something....
            # etc...

            # record and display count of blinks and winks for accuracy purposes
            if eye_status == EyeAction.BOTH_BLINK:
                total_blinks += 1
            if eye_status == EyeAction.LEFT_WINK:
                total_left_winks += 1
            if eye_status == EyeAction.RIGHT_WINK:
                total_right_winks += 1

            put_text(frame, "total blinks: " + str(total_blinks), (230, 40))
            put_text(frame, "total left winks: " + str(total_left_winks), (230, 55))
            put_text(frame, "total right winks: " + str(total_right_winks), (230, 70))

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
