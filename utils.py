import cv2
import math

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

# gets the point at a given index in the shape feature array of points
def shape_coord(shape, i):
    return shape[i][0], shape[i][1]

def put_text(frame, text, loc, thickness=0.5, color=(0, 0, 255)):
    cv2.putText(frame, text, loc, __DEF_FONT, thickness, color, 1)