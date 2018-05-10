import cv2
import math
from utils import *

UP_THRESH = 0.2      # decrease
DOWN_THRESH = 0.45    # increase
ZOOM_THRESH = 0.5     # decrease

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

class Head():
    """Head Position Detection

    https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
    """

    def __init__(self, shape, width):
        """
        Left Ear Index: 1.
        Right Ear Index: 16.
        Chin Index: 9.
        Nose Index: 30.
        Rightmost point in left eye Index: 39
        Leftmost point in left eye Index: 42
        """

        self.__left_ear = shape_coord(shape, 1)
        self.__right_ear = shape_coord(shape, 16)

        self.__chin = shape_coord(shape, 8)
        self.__nose_tip = shape_coord(shape, 30)
        self.__head = midpoint(self.__left_ear, self.__right_ear)
        self.__mid_eyes = midpoint(shape_coord(shape, 39), shape_coord(shape, 42))

        # Used in Down detection
        # Get the vertical distance between the center of the head and the nose tip
        # Get the vertical distance between the center of the head and the chin
        # Take the ratio of those two distances
        nose_head_dist = dist(self.__nose_tip, self.__head)
        nose_chin_dist = dist(self.__nose_tip, self.__chin)

        self.__nc_ratio = round(nose_head_dist / nose_chin_dist, 2)

        # Used in Up detection
        # compure ratio of head width over frame width
        self.__frame_width = width
        self.__head_zoom_ratio = dist(self.__left_ear, self.__right_ear) / width

    def turned_left(self):
        """Detects if the head is turned left.

        Compares the nose point against the midpoint
        of the leftmost head point and head center
        """
        nx, _ = self.__nose_tip
        mx, _ = midpoint(self.__left_ear, self.__head)

        return nx < mx

    def turned_right(self):
        """Detects if the head is turned right.

        Compares the nose point against the midpoint
        of the rightmost head point and head center
        """
        nx, _ = self.__nose_tip
        mx, _ = midpoint(self.__right_ear, self.__head)

        return nx > mx

    def turned_up(self):
        """Detects if the head is nodding up.

        Checks if nose tip is above center of head
        Checks if head's features have certain ratio implying up
        """
        _, ny = self.__nose_tip
        _, hy = self.__head

        return ny < hy and self.__nc_ratio > UP_THRESH

    def turned_down(self):
        """Detects if the head is nodding up.

        Checks if nose tip is below center of head
        Checks if head's features have certain ratio implying down
        """
        _, hy = self.__head
        _, ny = self.__nose_tip
        _, cy = self.__chin

        return hy < ny and ny < cy and self.__nc_ratio > DOWN_THRESH

    def zoom(self):
        """Determines if the head is zommed in or not.

        Compares one's head width / frame width ratio to threshold
        """
        return self.__head_zoom_ratio > ZOOM_THRESH

    def debug(self, frame):
        h, w, _ = frame.shape
        align_x = int(w * 0.63)
        align_y = int(h * 0.2)

        # Important features
        cv2.circle(frame, self.__nose_tip, 3, 255, -1)
        cv2.circle(frame, self.__head, 3, 255, -1)
        cv2.circle(frame, self.__chin, 3, 255, -1)
        cv2.circle(frame, self.__mid_eyes, 3, 255, -1)

        # Left debugging
        frame = draw_left_line(frame, midpoint(self.__left_ear, self.__head))
        frame = draw_right_line(frame, midpoint(self.__right_ear, self.__head))

        # Down debugging.
        put_text(frame, "DOWN: " + str(self.__nc_ratio),
                 (align_x, align_y + 20))
        cv2.line(frame, self.__head, self.__mid_eyes, (0, 100, 0), 2)
        cv2.line(frame, self.__head, self.__nose_tip, (255, 0, 255), 2)

        # Up debugging.
        put_text(frame, "UP: " + str(round(self.__nc_ratio, 2)),
                 (align_x, align_y + 40))
        cv2.line(frame, self.__nose_tip, self.__chin, (255, 0, 255), 2)

        # Zoom debugging.
        put_text(frame, "ZOOM: " + str(round(self.__head_zoom_ratio, 2)),
                 (align_x, align_y + 3 * 20))
