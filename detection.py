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

# timing constants
CLOSED_CONSEC_FRAMES = 5  #this is how many frames the signal is required to be consistent for

# returns what position the head is in
def detect_head(shape, frame, original_frame_width):
    cur_head = Head(shape, original_frame_width)

    head_state = HeadAction.CENTER
    zoom_state = HeadAction.ZOOMED if cur_head.zoom() else HeadAction.NOT_ZOOMED

    if cur_head.turned_left():
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

    return head_state, zoom_state

def detect_eyes(shape, frame, frame_counters):
    cur_eyes = Eyes(shape)
    status = EyeAction.BOTH_OPEN

    if cur_eyes.is_both_closed():
        frame_counters["both_eyes_closed"] += 1
        status = EyeAction.BOTH_CLOSED
    else:
        if frame_counters["both_eyes_closed"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.BOTH_BLINK
        frame_counters["both_eyes_closed"] = 0

    if cur_eyes.right_blink():
        frame_counters["right_blink"] += 1
        status = EyeAction.RIGHT_CLOSED
    else:
        if frame_counters["right_blink"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.RIGHT_WINK
        frame_counters["right_blink"] = 0

    if cur_eyes.left_blink():
        frame_counters["left_blink"] += 1
        status = EyeAction.LEFT_CLOSED
    else:
        if frame_counters["left_blink"] > CLOSED_CONSEC_FRAMES:
            status = EyeAction.LEFT_WINK
        frame_counters["left_blink"] = 0

    # draw some useful information
    if DEBUG_EYES:
        cur_eyes.debug(frame)

    return status
