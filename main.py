import argparse
import datetime
import cv2
import dlib
import math
import time

from imutils import face_utils

from utils import put_text, resize_frame
from detection import detect_head, detect_eyes
from action import HeadAction, EyeAction

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

    head_action_log = {
        HeadAction.LEFT: 0,
        HeadAction.RIGHT: 0,
        HeadAction.UP: 0,
        HeadAction.DOWN: 0,
        HeadAction.CENTER: 0,
        HeadAction.ZOOMED: 0,
        HeadAction.NOT_ZOOMED: 0
    }

    eye_action_log = {
        EyeAction.BOTH_OPEN: 0,
        EyeAction.BOTH_BLINK: 0,
        EyeAction.BOTH_CLOSED: 0,
        EyeAction.RIGHT_WINK: 0,
        EyeAction.RIGHT_CLOSED: 0,
        EyeAction.LEFT_WINK: 0,
        EyeAction.LEFT_CLOSED: 0
    }

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        frame = resize_frame(frame)
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # get the status of areas we are interested in
            head_action, zoom_action = detect_head(shape, frame)
            eye_action = detect_eyes(shape, frame, frame_counters)

            # display decisions
            print("=================================")
            align_x, align_y = int(w * 0.1), int(h * 0.1)
            print(head_action)
            print(zoom_action)
            print(eye_action)

            put_text(frame, str(head_action)[11:], (align_x, align_y + 30))
            put_text(frame, str(zoom_action)[11:], (align_x, align_y + 2 * 30))
            put_text(frame, str(eye_action)[10:], (align_x, align_y + 3 * 30))

            eye_action_log[eye_action] += 1
            head_action_log[head_action] += 1
            head_action_log[zoom_action] += 1

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
