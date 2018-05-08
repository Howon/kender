import cv2
from utils import *

WINK_THRESH = 0.02   # decrease
CLOSED_THRESH = 0.30   # increase

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
        #get histogram for right eye
        #get histogram for left eye
        #compare to see how different they are (look for whites of eyes)
        # if histograms very different and more white in left eye then return true
        return (self.__ear_l - self.__ear_r) > WINK_THRESH and self.__r_closed

    def left_blink(self):
        return (self.__ear_r - self.__ear_l) > WINK_THRESH and self.__l_closed

    def is_both_closed(self):
        return self.__l_closed and self.__r_closed

    def debug(self, frame):
        h, w, _ = frame.shape
        leftEyeHull = cv2.convexHull(self.__left_eye_indices)
        rightEyeHull = cv2.convexHull(self.__right_eye_indices)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        put_text(frame, " LEFT EAR: " + str(round(self.__ear_l, 2)), (int(w/2), 20))
        put_text(frame, "RIGHT EAR: " + str(round(self.__ear_r, 2)), (int(w/2), 2*20))

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

left_eye = np.zeros((6, 2))
for i, j in zip(range(36,42), range(0,6)):
    left_eye[j,0] = shape_coord(shape, i)[0]
    left_eye[j,1] = shape_coord(shape, i)[1]
left_eye_center = np.mean(left_eye, axis=0)
left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))

right_eye = np.zeros((6, 2))
for i, j in zip(range(42,48), range(0,6)):
    right_eye[j,0] = shape_coord(shape, i)[0]
    r3ight_eye[j,1] = shape_coord(shape, i)[1]
right_eye_center = np.mean(right_eye, axis=0)
right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))



# get the area of the convex hull of each eye
left_eye_hull = ConvexHull(left_eye)
right_eye_hull = ConvexHull(right_eye)
print(" left_eye_hull.area: ", left_eye_hull.area)
print("right_eye_hull.area: ", right_eye_hull.area)
print("difference: ", left_eye_hull.area - right_eye_hull.area)


# another possibility...
# analyze the pixels within the bounding box of the eye,
# then threshold on white, and then compare the area
# could also threshold on other things as well...
"""
