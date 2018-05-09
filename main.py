import argparse
import datetime
import cv2
import dlib
import math
import time

from imutils import face_utils

from display import *
from utils import COUNTER_LOG, put_text, resize_frame
from detection import detect_head, detect_eyes
from action import ActionHandler, HEAD_REST_STATE
from macro import MacroHandler, translate_action

CALIBRATE = True
__SPECTACLE_MACROS = '~/Library/Application Support/Spectacle/Shortcuts.json'

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

    action_handler = ActionHandler()
    macro_handler = MacroHandler(__SPECTACLE_MACROS)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    camera = cv2.VideoCapture(0)

    # frame counter stored counts of how many frames have passed for a certain detection
    frame_counters = {
        "both_eyes_closed" : 0,
        "left_blink" : 0,
        "right_blink" : 0
    }

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        h_original, w_original, _ = frame.shape
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
            head_action = detect_head(shape, frame, w_original)
            eye_action = detect_eyes(shape, frame, frame_counters)

            COUNTER_LOG[eye_action] += 1
            COUNTER_LOG[head_action] += 1

            # perform, action = action_handler.get_next(
            #     eye_action if head_action == HEAD_REST_STATE else head_action)

            perform, action = action_handler.get_next(head_action)

            display_decisions(frame, head_action, eye_action)
            display_counters(frame, COUNTER_LOG)

            if perform:
                COUNTER_LOG[action] += 1
                macro = translate_action(action)
                macro_handler.execute(macro)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
