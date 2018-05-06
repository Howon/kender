# import the necessary packages
import cv2
import math
import statistics
import numpy as np

from utils import *

from head import Head
from action import HeadAction
from scipy.spatial import ConvexHull

DEBUG = True

""" Decision Thresholds """
# eventually might want to make these calibrated or dynamic according to face ratio
# rather than absolute values regarding the frame dimensions
                       # to make detection more sensitive:

CLOSE_THRESHOLD = 4   # increase
EAR_CLOSE_THRESHOLD = 0.25   # increase
WINK_THRESHOLD = 0.02   # decrease

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

""" Eye Status Detection """
# for all eye functions the distance between the eyelid and the bottom of
# the eye for each eye is compared to a threshold

# source: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# source: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist(eye[0], eye[3])

    # compute the eye aspect ratio
    EAR = (A + B) / (2.0 * C)

    # return the eye aspect ratio (E.A.R.)
    return EAR

def right_blink(left_EAR, right_EAR):
    return (left_EAR - right_EAR) > WINK_THRESHOLD and right_EAR < EAR_CLOSE_THRESHOLD

def left_blink(left_EAR, right_EAR):
    return (right_EAR - left_EAR) > WINK_THRESHOLD and left_EAR < EAR_CLOSE_THRESHOLD

#checks if left and right eye have low EAR (i.e. eyes are closed)
def is_both_closed(left_EAR, right_EAR):
    return left_EAR < EAR_CLOSE_THRESHOLD and right_EAR < EAR_CLOSE_THRESHOLD

# checks the status of the eyes
def detect_eyes(shape, frame, frame_counters):
    # https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
    # this is how the eyes are indexed
    #
    #           *37 *38              *43 *44
    #        *36       *39        *42       *45
    #           *41 *40              *47 *46
    # first get all our points of interest
    left_eye_points = []
    for i in range(36, 42):
        left_eye_points.append(shape_coord(shape, i))

    right_eye_points = []
    for i in range(42, 48):
        right_eye_points.append(shape_coord(shape, i))


    leftEye = shape[36:42]
    rightEye = shape[42:48]


    # calculate the Eye Aspect Ratio (EAR)
    left_EAR = eye_aspect_ratio(left_eye_points)
    right_EAR = eye_aspect_ratio(right_eye_points)


    # uncomment if you want some other possibly useful
    # features that I had been experimenting with...
    """
    # the following may also come in handy at some point:
    center_head_point = head_center(shape)
    left_eye_top_left_point = shape_coord(shape, 37)
    left_eye_top_right_point = shape_coord(shape, 38)
    left_eye_bottom_left_point = shape_coord(shape, 41)
    left_eye_bottom_right_point = shape_coord(shape, 40)

    right_eye_top_left_point = shape_coord(shape, 43)
    right_eye_top_right_point = shape_coord(shape, 44)
    right_eye_bottom_left_point = shape_coord(shape, 47)
    right_eye_bottom_right_point = shape_coord(shape, 46)

    # we want to calculate the distance of these lines
    #
    #           *37 *38              *43 *44
    #        *36  |   | *39       *42 |   | *45
    #           *41 *40              *47 *46
    left_eye_left_dist = dist(left_eye_top_left_point, left_eye_bottom_left_point)
    left_eye_right_dist = dist(left_eye_top_right_point, left_eye_bottom_right_point)

    right_eye_left_dist = dist(right_eye_top_left_point, right_eye_bottom_left_point)
    right_eye_right_dist = dist(right_eye_top_right_point, right_eye_bottom_right_point)

    # then we take the average distane of the two lines on each eye
    left_eye_close_dist = statistics.mean([left_eye_left_dist, left_eye_right_dist])
    right_eye_close_dist = statistics.mean([right_eye_left_dist, right_eye_right_dist])

    # We want to calculate center of the eye's landmarks.
    # The center will move down when the eye blinks
    #
    #           *37 *38               *43 *44
    #         *36   []   *39       *42   []  *45
    #           *41 *40               *47 *46

    left_eye_points = np.zeros((6, 2))
    for i, j in zip(range(36,42), range(0,6)):
        left_eye_points[j,0] = shape_coord(shape, i)[0]
        left_eye_points[j,1] = shape_coord(shape, i)[1]
    left_eye_center = np.mean(left_eye_points, axis=0)
    left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))

    right_eye_points = np.zeros((6, 2))
    for i, j in zip(range(42,48), range(0,6)):
        right_eye_points[j,0] = shape_coord(shape, i)[0]
        r3ight_eye_points[j,1] = shape_coord(shape, i)[1]
    right_eye_center = np.mean(right_eye_points, axis=0)
    right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))



    # get the area of the convex hull of each eye
    left_eye_hull = ConvexHull(left_eye_points)
    right_eye_hull = ConvexHull(right_eye_points)
    print(" left_eye_hull.area: ", left_eye_hull.area)
    print("right_eye_hull.area: ", right_eye_hull.area)
    print("difference: ", left_eye_hull.area - right_eye_hull.area)


    # another possibility...
    # analyze the pixels within the bounding box of the eye,
    # then threshold on white, and then compare the area
    # could also threshold on other things as well...
    """


    # draw some useful information
    if DEBUG:
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, " LEFT EAR: " + str(round(left_EAR,2)), (230, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "RIGHT EAR: " + str(round(right_EAR,2)), (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # this is for major debugging. uncomment if the eye blinking is completely wack
        # each index might be off by 1 which would cause everything to mess up...
        # you can check that the eye indexes are what you think they are this way
        """
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        i = 30
        for (x, y) in shape[30:50]:
            #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            #uncomment to check indexes
            cv2.putText(frame, "*"+str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            i+=1
        """
    status = "EYES OPEN"

    # blink
    if is_both_closed(left_EAR, right_EAR):
        frame_counters["both_eyes_closed"] += 1
        status = "EYES CLOSED"
        cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        if frame_counters["both_eyes_closed"] > CLOSED_CONSEC_FRAMES:
            status = "EYES BLINKED"
        frame_counters["both_eyes_closed"] = 0

    # right wink
    if right_blink(left_EAR, right_EAR):
        frame_counters["right_blink"] += 1
        status = "RIGHT EYE CLOSED"
        cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        if frame_counters["right_blink"] > CLOSED_CONSEC_FRAMES:
            status = "RIGHT WINK"
        frame_counters["right_blink"] = 0

    # left wink
    if left_blink(left_EAR, right_EAR):
        frame_counters["left_blink"] += 1
        status = "LEFT EYE CLOSED"
        cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        if frame_counters["left_blink"] > CLOSED_CONSEC_FRAMES:
            status = "LEFT WINK"
        frame_counters["left_blink"] = 0

    return status
