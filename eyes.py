import cv2
from utils import *

WINK_THRESH = 0.015   # decrease
CLOSED_THRESH = 0.30   # increase
EYE_HIST_THRESH = 0.15
EYE_RECT_MODIFIER = 10
EYE_RECT_MIN_HEIGHT = 25

DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
class Eyes():
    """Eye Status Detection
    For all eye functions the distance between the eyelid and the bottom of
    the eye for each eye is compared to a threshold

    source: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    source: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    """
    def __init__(self, shape, frame):
        """
        Eye indices:
                *37 *38              *43 *44
            *36         *39      *42         *45
                *41 *40              *47 *46
        https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        """
        left_eye = [shape_coord(shape, i) for i in range(36, 42)]
        right_eye = [shape_coord(shape, i + 6) for i in range(36, 42)]

        # # calculate the Eye Aspect Ratio (EAR)
        # self.__ear_l = self.__eye_aspect_ratio(left_eye)
        # self.__ear_r = self.__eye_aspect_ratio(right_eye)

        # necessary for drawing the eye in the debug function
        self.__left_eye_indices = shape[36:42]
        self.__right_eye_indices = shape[42:48]

        self.__l_rect, l_disp = self.__eye_rect(self.__left_eye_indices, frame)
        self.__r_rect, r_disp = self.__eye_rect(self.__right_eye_indices, frame)

        self.__l_disp = l_disp
        self.__r_disp = r_disp
        self.__l_hist = self.__check_hist(self.__l_rect[2])
        self.__r_hist = self.__check_hist(self.__r_rect[2])

        # self.__l_closed = self.__l_hist < EYE_HIST_THRES and self.__ear_l < CLOSED_THRESH
        # self.__r_closed = self.__r_hist < EYE_HIST_THRES and self.__ear_r < CLOSED_THRESH

        self.__l_closed = self.__l_hist < EYE_HIST_THRESH
        self.__r_closed = self.__r_hist < EYE_HIST_THRESH

    def __eye_rect(self, eye_points, frame):
        """Get the bounding rectangle for a given eye.
        Args:
            eye_points:
                   * *
                *       *
                   * *
            frame: Current captured frame.
        Returns:
            Top left and bottom right coordinates for the rectangle that encapsulates
            eye coordinates.
                -----------
                |   * *   |
                |*       *|
                |   * *   |
                -----------
        """
        tlx = eye_points[0][0] - EYE_RECT_MODIFIER
        tly = eye_points[1][1] - EYE_RECT_MODIFIER

        brx = eye_points[3][0] + EYE_RECT_MODIFIER
        bry = eye_points[4][1] + EYE_RECT_MODIFIER

        tl, br = (tlx, tly), (brx, bry)
        try:
            eye = frame[tly:bry, tlx:brx].copy()
            eye_copy = eye.copy()

            gray_eye = cv2.cvtColor(eye.copy(), cv2.COLOR_BGR2GRAY)
            _, thresh_eye = cv2.threshold(gray_eye, 80, 255, cv2.THRESH_BINARY)

            gray_eye_disp = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2BGR)
            thresh_eye_disp = cv2.cvtColor(thresh_eye, cv2.COLOR_GRAY2BGR)

            dilation = cv2.dilate(thresh_eye, DILATE_KERNEL, iterations=1)
            thresh_eye = cv2.erode(dilation, DILATE_KERNEL, iterations=1)

            eroded = cv2.cvtColor(thresh_eye, cv2.COLOR_GRAY2BGR)
            disp = np.hstack((eye_copy, gray_eye_disp, thresh_eye_disp, eroded))

            return (tl, br, thresh_eye), disp
        except:
            pass

        return (tl, br, None), None

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

    def __check_hist(self, eye):
        if eye is None:
            return 1

        h, w = eye.shape
        h = h if h > EYE_RECT_MIN_HEIGHT else EYE_RECT_MIN_HEIGHT

        return 1 - (len(eye.nonzero()[0]) / (h * w))

    def left_blink(self):
        #return (self.__ear_r - self.__ear_l) > WINK_THRESH and self.__l_closed
        return self.__l_closed

    def right_blink(self):
        # return (self.__ear_l - self.__ear_r) > WINK_THRESH and self.__r_closed
        return self.__r_closed

    def both_closed(self):
        return self.__l_closed and self.__r_closed

    def debug(self, frame):
        h, w, _ = frame.shape
        l_tl, l_br = self.__l_rect[0:2]
        r_tl, r_br = self.__r_rect[0:2]

        cv2.rectangle(frame, l_tl, l_br, (0,255,0), 3)
        cv2.rectangle(frame, r_tl, r_br, (0,255,0), 3)

        l_cnt = cv2.convexHull(self.__left_eye_indices)
        r_cnt = cv2.convexHull(self.__right_eye_indices)

        cv2.drawContours(frame, [l_cnt], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [r_cnt], -1, (0, 255, 0), 1)

        put_text(frame, "LEFT HIST: " + str(round(self.__l_hist, 2)),
                 (int(w/2), 2 * 20))
        put_text(frame, "RIGHT HIST: " + str(round(self.__r_hist, 2)),
                 (int(w/2), 4 * 20))

        l_disp_y = 0
        if self.__l_disp is not None:
            l_disp_y, l_disp_x, _ = self.__l_disp.shape
            frame[0:l_disp_y, 0:l_disp_x] = self.__l_disp

        if self.__r_disp is not None:
            r_disp_y, r_disp_x, _ = self.__r_disp.shape
            frame[l_disp_y:(l_disp_y + r_disp_y), 0:r_disp_x] = self.__r_disp
