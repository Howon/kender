# import the necessary packages
import cv2
import math
import statistics
import numpy as np

from utils import *

from action import HeadAction, EyeAction
from scipy.spatial import ConvexHull

DEBUG_HEAD = True
DEBUG_EYES = True

""" Decision Thresholds """
# eventually might want to make these calibrated or dynamic according to face ratio
# rather than absolute values regarding the frame dimensions
# to make detection more sensitive:

# returns what position the head is in
def detect_head(shape, cur_head):
    head_state = HeadAction.CENTER

    if cur_head.turned_left():
        head_state = HeadAction.LEFT
    elif cur_head.turned_right():
        head_state = HeadAction.RIGHT
    elif cur_head.zoom():
        head_state = HeadAction.ZOOM
    elif cur_head.turned_up():
        head_state = HeadAction.UP
    elif cur_head.turned_down():
        head_state = HeadAction.DOWN

    return head_state

def detect_eyes(shape, cur_eyes):
    eye_action = EyeAction.BOTH_OPEN

    if cur_eyes.leaning():
        return EyeAction.BOTH_OPEN

    if cur_eyes.both_closed():
        return EyeAction.BOTH_CLOSED

    if cur_eyes.left_blink():
        return EyeAction.LEFT_CLOSED

    if cur_eyes.right_blink():
        return EyeAction.RIGHT_CLOSED

    return EyeAction.BOTH_OPEN
