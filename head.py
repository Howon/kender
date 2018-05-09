import cv2
from utils import *

     # to make more sensitive:
UP_THRESH = 0.5     # decrease
DOWN_THRESH = 0.3    # increase
ZOOM_THRESH = 0.25    # decrease

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

    def __init__(self, shape, original_frame_width):
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
        self.__between_eyes = midpoint(shape_coord(shape, 39), shape_coord(shape, 42))

        # Used in Down detection
        # Get the vertical distance between the center of the head and the nose tip
        # Get the vertical distance between the center of the head and the chin
        # Take the ratio of those two distances
        chin_head_dist = max(y_dist(self.__chin, self.__head), 0.01)
        self.__nose_chin_ratio = y_dist(self.__nose_tip, self.__head) / chin_head_dist

        # Used in Up detection
        # Get the vertical distance between between the center of the head and the nose tip.
        # Get the vertical distance between between the center of the head and the midpoint of the eyes.
        # Take the ratio of those two distances
        nose_head_dist = max(y_dist(self.__nose_tip, self.__head), 0.01)
        self.__nose_eye_ratio = y_dist(self.__between_eyes, self.__head) / nose_head_dist

        # Used in Up detection
        # compure ratio of head width over frame width
        self.__head_zoom_ratio = dist(self.__left_ear, self.__right_ear) / original_frame_width

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

        return ny < hy and self.__nose_chin_ratio > UP_THRESH

    def turned_down(self):
        """Detects if the head is nodding up.

        Checks if nose tip is below center of head
        Checks if head's features have certain ratio implying down
        """
        _, ny = self.__nose_tip
        _, hy = self.__head

        return ny > hy and self.__nose_eye_ratio < DOWN_THRESH

    def zoom(self):
        """Determines if the head is zommed in or not.

        Compares one's head width / frame width ratio to threshold
        """
        return self.__head_zoom_ratio > ZOOM_THRESH

    def debug(self, frame):
        h, w, _ = frame.shape
        align_x = int(w*0.1)
        align_y = int(h*0.4)

        # Important features
        cv2.circle(frame, self.__nose_tip, 3, 255, -1)
        cv2.circle(frame, self.__head, 3, 255, -1)
        cv2.circle(frame, self.__chin, 3, 255, -1)
        cv2.circle(frame, self.__between_eyes, 3, 255, -1)

        # Left debugging
        frame = draw_left_line(frame, midpoint(self.__left_ear, self.__head))

        # Right debugging
        frame = draw_right_line(frame, midpoint(self.__right_ear, self.__head))

        # Down debugging.
        put_text(frame, "DOWN: nose_eye_ratio " + str(round(self.__nose_eye_ratio, 2)), (align_x, align_y+20))
        #cv2.line(frame, self.__head, self.__between_eyes, (0, 100, 0), 2)
        #cv2.line(frame, self.__head, self.__nose_tip, (255, 0, 255), 2)

        # Up debugging.
        put_text(frame, "UP: nose_chin_ratio " + str(round(self.__nose_chin_ratio, 2)), (align_x, align_y+20))
        #cv2.line(frame, self.__head, self.__nose_tip, (255, 0, 255), 2)
        #cv2.line(frame, self.__head, self.__chin, (255, 0, 0), 2)

        # Zoom debugging.
        put_text(frame, "ZOOM: head_zoom_ratio " + str(round(self.__head_zoom_ratio, 2)), (align_x, align_y+2*20))
        #cv2.line(frame, self.__left_ear, self.__right_ear, (255, 0, 255), 2)
