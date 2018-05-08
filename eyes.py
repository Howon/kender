import cv2
from utils import *

WINK_THRESH = 0.02   # decrease
CLOSED_THRESH = 0.30   # increase
EYE_RECT_SCALE = 5

def eye_rect(eye_points):
    """Get the bounding rectangle for a given eye.
    Args:
        eye_points:
               * *
            *       *
               * *
    Returns:
        Top left and bottom right coordinates for the rectangle that encapsulates
        eye coordinates.
            -----------
            |   * *   |
            |*       *|
            |   * *   |
            -----------
    """
    tlx = eye_points[0][0]
    tly = eye_points[1][1]

    brx = eye_points[3][0]
    bry = eye_points[4][1]

    return (tlx, tly), (brx, bry)

class Eyes():
    """Eye Status Detection
    For all eye functions the distance between the eyelid and the bottom of
    the eye for each eye is compared to a threshold

    source: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    source: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    """
    def __init__(self, shape):
        """
        Eye indices:
                *37 *38              *43 *44
            *36         *39      *42         *45
                *41 *40              *47 *46
        https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        """

        left_eye = [shape_coord(shape, i) for i in range(36, 42)]
        right_eye = [shape_coord(shape, i + 6) for i in range(36, 42)]

        # calculate the Eye Aspect Ratio (EAR)
        self.__ear_l = self.__eye_aspect_ratio(left_eye)
        self.__ear_r = self.__eye_aspect_ratio(right_eye)

        self.__l_closed = self.__ear_l < CLOSED_THRESH
        self.__r_closed = self.__ear_r < CLOSED_THRESH

        # necessary for drawing the eye in the debug function
        self.__left_eye_indices = shape[36:42]
        self.__right_eye_indices = shape[42:48]

        self.__l_rect = eye_rect(self.__left_eye_indices)
        self.__r_rect = eye_rect(self.__right_eye_indices)

    def __eye_aspect_ratio(self, eye):
        """Compute the distances between the two sets of vertical eye landmarks.

        Eye indices:
               *1 *2
            *0       *6
               *5 *4
        """
        # Vertical distance ratios.
        A, B = dist(eye[1], eye[5]), dist(eye[2], eye[4])

        # Horizontal distance ratios.
        C = dist(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def right_blink(self):
        # Compare to see how different they are (look for whites of eyes)
        # if histograms very different and more white in left eye then return true
        return (self.__ear_l - self.__ear_r) > WINK_THRESH and self.__r_closed

    def left_blink(self):
        return (self.__ear_r - self.__ear_l) > WINK_THRESH and self.__l_closed

    def is_both_closed(self):
        return self.__l_closed and self.__r_closed

    def debug(self, frame):
        h, w, _ = frame.shape
        l_tl, l_br = self.__l_rect
        r_tl, r_br = self.__r_rect

        cv2.rectangle(frame, l_tl, l_br, (0,255,0), 3)
        cv2.rectangle(frame, r_tl, r_br, (0,255,0), 3)

        # leftEyeHull = cv2.convexHull(self.__left_eye_indices)
        # rightEyeHull = cv2.convexHull(self.__right_eye_indices)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        put_text(frame, " LEFT EAR: " + str(round(self.__ear_l, 2)), (int(w/2), 20))
        put_text(frame, "RIGHT EAR: " + str(round(self.__ear_r, 2)), (int(w/2), 2*20))
