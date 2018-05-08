import cv2
from utils import *

     # to make more sensitive:
UP_THRESH = 0.5     # decrease
DOWN_THRESH = 0.5    # decrease

ZOOM_THRESH = 60    # decrease

""" Drawing functions """
# draws the threshold line for left turn
def draw_left_line(frame, left_midpoint):
    ptA = ((left_midpoint[0]), left_midpoint[1] + 50)
    ptB = ((left_midpoint[0]), left_midpoint[1] - 50)
    cv2.line(frame, ptA, ptB, (255, 255, 255), 2)

    return frame

# draws the threshold line for right turn
def draw_right_line(frame, right_midpoint):
    ptA = ((right_midpoint[0]), right_midpoint[1] + 50)
    ptB = ((right_midpoint[0]), right_midpoint[1] - 50)
    cv2.line(frame, ptA, ptB, (255, 255, 255), 2)

    return frame

# draws the threshold line for right turn
def draw_up_line(frame, head_center):
    ptA = (170, (head_center[1] + UP_THRESH))
    ptB = (230, (head_center[1] + UP_THRESH))
    cv2.line(frame, ptA, ptB, (255, 0, 0), 2)

    return frame

# draws the threshold line for right turn
def draw_down_line(frame, head_center):
    ptA = (170, (head_center[1] + DOWN_THRESH))
    ptB = (230, (head_center[1] + DOWN_THRESH))
    cv2.line(frame, ptA, ptB, (255, 255, 255), 2)

    return frame

class Head():
    """Head Position Detection

    https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
    """

    def __init__(self, shape):
        """

        Left Ear Index: 2.
        Right Ear Index: 16.
        Chin Index: 9.
        Nose Index: 30.
        """
        self.__left_ear = shape_coord(shape, 2)
        self.__right_ear = shape_coord(shape, 16)

        self.__chin = shape_coord(shape, 8)
        self.__nose_tip = shape_coord(shape, 30)
        self.__head = midpoint(self.__left_ear, self.__right_ear)
        self.__between_eyes = midpoint(shape_coord(shape, 39), shape_coord(shape, 42))
        self.__nose_chin_ratio = dist(self.__nose_tip, self.__head) / dist(self.__chin, self.__head)
        self.__nose_eye_ratio = dist(self.__between_eyes, self.__head) / dist(self.__nose_tip, self.__head) 


    def turned_left(self):
        """Detects if the head is turned left.

        Compares the nose point against the head center.
        """
        nx, _ = self.__nose_tip
        mx, _ = midpoint(self.__left_ear, self.__head)

        return nx < mx

    def turned_right(self):
        """Detects if the head is turned right.

        Compares the nose point against the head center.
        """
        nx, _ = self.__nose_tip
        mx, _ = midpoint(self.__right_ear, self.__head)

        return nx > mx

    def turned_up(self):
        """Detects if the head is nodding up.

        Checks if the chin point is above the head center.
        """ 
        _, ny = self.__nose_tip
        _, hy = self.__head

        return ny < hy and self.__nose_chin_ratio > UP_THRESH

    def turned_down(self):
        """Detects if the head is nodding up.

        Checks the chin point is below the head center.
        """
        _, ny = self.__nose_tip
        _, hy = self.__head

        return ny > hy and self.__nose_eye_ratio < DOWN_THRESH

    def zoom(self):
        """Determines if the head is zommed in or not.

        Compares the distance between center of head and right ear.
        """
        return dist(self.__right_ear, self.__head) > ZOOM_THRESH

    def debug(self, frame):
        h, w, _ = frame.shape
        # Zoom debugging.
        cv2.circle(frame, self.__right_ear, 1, (255, 0, 0), -1)
        cv2.line(frame, self.__head, self.__right_ear, (255, 200, 255), 2)
        # White line needs to be larger than this pink line according to zoom threshold
        cv2.line(frame, (200, 105), (258, 87), (255, 200, 255), 2)

        # Head turn debugging.
        cv2.circle(frame, self.__nose_tip, 2, (255, 255, 255), -1)
        cv2.circle(frame, self.__head, 2, (0, 0, 255), -1)
        cv2.circle(frame, self.__chin, 1, (255, 0, 0), -1)
        cv2.circle(frame, self.__between_eyes, 1, (0, 100, 0), -1)
        frame = draw_left_line(frame, midpoint(self.__left_ear, self.__head))
        frame = draw_right_line(frame, midpoint(self.__right_ear, self.__head))
        put_text(frame, "nose_chin_ratio " + str(round(self.__nose_chin_ratio, 2)), (int(w/2), 4*20))
        put_text(frame, "nose_eye_ratio " + str(round(self.__nose_eye_ratio, 2)), (int(w/2), 5*20))
