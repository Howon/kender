import cv2
import math
import numpy as np

__DEF_FONT = cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1, p2):
    """Midpoint between two points.

    Args:
        p1: Starting point.
        p2: Ending point.
    Returns:
        Midpoint between p1 and p2.
    """
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

def dist(p1, p2):
    """Euclidean distance between two points.

    Args:
        p1: Starting point.
        p2: Ending point.
    Returns:
        Euclidean Distance.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def y_dist(p1, p2):
    """Euclidean distance between two points.

    Args:
        p1: Starting point.
        p2: Ending point.
    Returns:
        Absolute Y Distance.
    """
    return abs(p2[1] - p1[1])

# gets the point at a given index in the shape feature array of points
def shape_coord(shape, i):
    return shape[i][0], shape[i][1]

def put_text(frame, text, loc, scale=0.5, color=(0, 0, 255), thickness=1):
    cv2.putText(frame, text, loc, __DEF_FONT, scale, color, thickness)

def resize_frame(frame):
    h, w, _ = frame.shape

    x = frame.copy()
    right_removed = np.delete(x, range(3 * w // 4, w), axis=1)
    left_removed = np.delete(right_removed, range(0, w//4), axis=1)

    return cv2.flip(left_removed, 1)
