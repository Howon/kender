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

DEBUG = True

""" Decision Thresholds """
# eventually might want to make these calibrated or dynamic according to face ratio
# rather than absolute values regarding the frame dimensions
                       # to make detection more sensitive:

# timing constants
CLOSED_CONSEC_FRAMES = 5  #this is how many frames the signal is required to be consistent for

# returns what position the head is in
def detect_head(shape, frame):
    # first get all our points of interest
    cur_head = Head(shape)

    head_state = HeadAction.CENTER
    zoom_state = cur_head.zoomed_in()

    # check what position the head is in
    if cur_head.turned_left():
        head_state = HeadAction.LEFT
    elif cur_head.turned_right():
        head_state = HeadAction.RIGHT
    elif cur_head.turned_up():
        head_state = HeadAction.UP
    elif cur_head.turned_down():
        head_state = HeadAction.DOWN

    # draw some useful information
    if DEBUG:
        cur_head.debug(frame)

    return head_state, zoom_state

def detect_eyes(shape, frame, frame_counters):
    cur_eyes = Eyes(shape)
    status = EyeAction.BOTH_OPEN

    if cur_eyes.is_both_closed():
        frame_counters["both_eyes_closed"] += 1
        status = EyeAction.BOTH_CLOSED
        put_text(frame, str(status), (30, 90), thickness=0.7, scale=2)
    else:
        if frame_counters["both_eyes_closed"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.BOTH_BLINK
        frame_counters["both_eyes_closed"] = 0

    if cur_eyes.right_blink():
        frame_counters["right_blink"] += 1
        status = EyeAction.RIGHT_CLOSED
        put_text(frame, str(status), (30, 90), thickness=0.7, scale=2)
    else:
        if frame_counters["right_blink"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.RIGHT_WINK
        frame_counters["right_blink"] = 0

    if cur_eyes.left_blink():
        frame_counters["left_blink"] += 1
        status = EyeAction.LEFT_CLOSED
        put_text(frame, str(status), (30, 90), thickness=0.7, scale=2)
    else:
        if frame_counters["left_blink"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.LEFT_WINK
        frame_counters["left_blink"] = 0

    return status
