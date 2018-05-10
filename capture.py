import datetime
import cv2
import dlib
import math
import time

from imutils import face_utils

from head import Head
from eyes import Eyes
from display import *
from utils import COUNTER_LOG, put_text, resize_frame
from detection import detect_head, detect_eyes
from action import ActionHandler, HEAD_REST_STATE

CALIBRATE = True

def capture_action(pred_path, cb, debug=False, log=False):
    """Facial detection and real time processing credit goes to:

    https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
    """
    action_handler = ActionHandler()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_path)

    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        if frame is None:
            continue

        _, w_original, _ = frame.shape
        frame = resize_frame(frame)

        h, w, _ = frame.shape

        display_bounds(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            cur_head = Head(shape, w)
            cur_eyes = Eyes(shape, frame)

            eye_action = detect_eyes(shape, cur_eyes)
            head_action = detect_head(shape, cur_head)

            COUNTER_LOG[eye_action] += 1
            COUNTER_LOG[head_action] += 1

            perform, action = action_handler.get_next(eye_action, head_action)

            if log:
                display_decisions(frame, head_action, eye_action)
                display_counters(frame, COUNTER_LOG)

            if perform:
                COUNTER_LOG[action] += 1
                cb(action)

            if debug:
                cur_head.debug(frame)
                cur_eyes.debug(frame)

                cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
