# import the necessary packages
import cv2
import math
import statistics
import numpy as np

from utils import *

from head import Head
from eyes import Eyes
from action import HeadAction, EyeAction
from scipy.spatial import ConvexHull

DEBUG_HEAD = True
DEBUG_EYES = True

""" Decision Thresholds """
# eventually might want to make these calibrated or dynamic according to face ratio
# rather than absolute values regarding the frame dimensions
                       # to make detection more sensitive:

# returns what position the head is in
def detect_head(shape, frame, original_frame_width):
    if frame is None:
        return HeadAction.CENTER

    cur_head = Head(shape, original_frame_width)

    head_state = HeadAction.CENTER

    if cur_head.zoom():
        head_state = HeadAction.ZOOM
    elif cur_head.turned_left():
        head_state = HeadAction.LEFT
    elif cur_head.turned_right():
        head_state = HeadAction.RIGHT
    elif cur_head.turned_up():
        head_state = HeadAction.UP
    elif cur_head.turned_down():
        head_state = HeadAction.DOWN
    else:
        head_state = HeadAction.CENTER

    if DEBUG_HEAD:
        cur_head.debug(frame)

    return head_state

def detect_eyes(shape, frame, frame_counters):
    if frame is None:
        return EyeAction.BOTH_OPEN

    cur_eyes = Eyes(shape, frame)
    eye_action = EyeAction.BOTH_OPEN

    if cur_eyes.both_closed():
        frame_counters["both_eyes_closed"] += 1
        eye_action = EyeAction.BOTH_CLOSED
    else:
        frame_counters["both_eyes_closed"] = 0

    if cur_eyes.right_blink():
        frame_counters["right_blink"] += 1
        eye_action = EyeAction.RIGHT_CLOSED
    else:
        frame_counters["right_blink"] = 0

    if cur_eyes.left_blink():
        frame_counters["left_blink"] += 1
        eye_action = EyeAction.LEFT_CLOSED
    else:
        frame_counters["left_blink"] = 0

    if DEBUG_EYES:
        cur_eyes.debug(frame)

    return eye_action
