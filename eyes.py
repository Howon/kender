import cv2
from utils import *

_EYE_HIST_THRESH = 0.02
_EYE_DIFF_THRESH = 1.7

_EYE_RECT_MODIFIER = 7

_LEAN_THRESH = 10
_C_FLOOR = 70
_ELLIPSE_SCALE = 0.45
_MASK_THICKNESS = 10
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

class Eyes():
    """Eye Status Detection
    For all eye functions the distance between the eyelid and the bottom of
    the eye for each eye is compared to a threshold

    source: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    """
    def __init__(self, shape, frame):
        """
        Eye indices:
                *37 *38              *43 *44
            *36         *39      *42         *45
                *41 *40              *47 *46
        """
        left_eye = [shape_coord(shape, i) for i in range(36, 42)]
        right_eye = [shape_coord(shape, i + 6) for i in range(36, 42)]

        self.__left_eye = shape[36:42]
        self.__right_eye = shape[42:48]

        self.__l_rect, l_disp = self.__find_eye_roi(self.__left_eye, frame)
        self.__r_rect, r_disp = self.__find_eye_roi(self.__right_eye, frame)

        self.__l_disp = l_disp
        self.__r_disp = r_disp
        self.__l_hist = self.__check_hist(self.__l_rect[2])
        self.__r_hist = self.__check_hist(self.__r_rect[2])

        self.__l_closed = self.__is_closed(self.__l_hist, self.__r_hist)
        self.__r_closed = self.__is_closed(self.__r_hist, self.__l_hist)

    def __mask_eyelash(self, eye, ellipse):
        eye_cpy = eye.copy()
        center, size, angle = ellipse

        c = (int(center[0]), int(center[1]))
        axes = (int(2.2 * _ELLIPSE_SCALE * size[0]),
                int(0.7 * _ELLIPSE_SCALE * size[1]))
        angle = int(angle)

        stencil = np.zeros(eye_cpy.shape)
        cv2.ellipse(stencil, ellipse, 255, -1)
        cv2.ellipse(eye_cpy, c, axes, angle, 0, 360, 255, _MASK_THICKNESS)
        cv2.ellipse(eye_cpy, c, axes, angle, 0, 360, 255, _MASK_THICKNESS)

        eye_cpy[stencil == 0] = 255

        return eye_cpy

    def __find_eye_roi(self, eye_points, frame):
        """Finds the ROI for eye action parsing.
        Args:
            eye_points:
                   * *
                *       *
                   * *
            frame: Current captured frame.
        Returns:
            1. Top left and bottom right coordinates for the rectangle that
               encapsulates the eye coordinates.
                -----------
                |   * *   |
                |*       *|
                |   * *   |
                -----------
            2. Thresholded Eye.
            3. Various stages of the eye transformation stacked horizontally.
        """
        tlx = eye_points[0][0] - _EYE_RECT_MODIFIER
        tly = eye_points[1][1] - _EYE_RECT_MODIFIER

        brx = eye_points[3][0] + _EYE_RECT_MODIFIER
        bry = eye_points[4][1] + _EYE_RECT_MODIFIER

        tl, br = (tlx, tly), (brx, bry)

        rel_eye_points = np.asarray([(x - tlx, y - tly) for x, y in eye_points])

        try:
            eye = frame[tly:bry, tlx:brx].copy()
            h, w,_ = eye.shape

            eye_copy = eye.copy()

            gray_eye = cv2.cvtColor(eye.copy(), cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_eye, _C_FLOOR, 255,
                                      cv2.THRESH_BINARY)

            gray_eye_disp = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2BGR)
            thresh_disp = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            ellipse = cv2.fitEllipse(rel_eye_points)

            thresh = self.__mask_eyelash(thresh, ellipse)

            no_eyelash = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
            dilation = cv2.dilate(thresh, _DILATE_KERNEL, iterations=1)
            thresh = cv2.erode(dilation, _DILATE_KERNEL, iterations=1)
            cv2.GaussianBlur(thresh, (3, 3), 0)

            eroded = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

            disp = np.hstack((eye_copy, gray_eye_disp, thresh_disp,
                              no_eyelash, eroded))

            return (tl, br, thresh), disp
        except:
            pass

        return (tl, br, None), None

    def __check_hist(self, eye):
        if eye is None:
            return 1

        h, w = eye.shape
        return 1 - (len(eye.nonzero()[0]) / (h * w))

    def __is_closed(self, hist, other):
        return _EYE_DIFF_THRESH * hist < other or hist < _EYE_HIST_THRESH

    def leaning(self):
        l_tl, l_br, __ = self.__l_rect
        r_tl, r_br, __ = self.__r_rect

        ly = l_br[1] - 0.5 * l_tl[1]
        ry = r_br[1] - 0.5 * r_tl[1]

        return abs(ly - ry) > _LEAN_THRESH

    def left_blink(self):
        return self.__l_closed

    def right_blink(self):
        return self.__r_closed

    def both_closed(self):
        return self.__l_closed and self.__r_closed

    def debug(self, frame):
        h, w, _ = frame.shape

        align_x = int(w * 0.63)
        align_y = int(h * 0.1)

        l_tl, l_br = self.__l_rect[0:2]
        r_tl, r_br = self.__r_rect[0:2]

        cv2.rectangle(frame, l_tl, l_br, (0, 255 ,0), 2)
        cv2.rectangle(frame, r_tl, r_br, (0, 255, 0), 2)

        l_cnt = cv2.convexHull(self.__left_eye)
        r_cnt = cv2.convexHull(self.__right_eye)

        cv2.drawContours(frame, [l_cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [r_cnt], -1, (0, 255, 0), 2)

        put_text(frame, "LEFT HIST: " + str(round(self.__l_hist, 2)),
                 (align_x, align_y + 20))
        put_text(frame, "RIGHT HIST: " + str(round(self.__r_hist, 2)),
                 (align_x, align_y + 2 * 20))

        l_disp_y = 0
        if self.__l_disp is not None:
            l_disp_y, l_disp_x, _ = self.__l_disp.shape

            if l_disp_x < w:
                frame[0:l_disp_y, 0:l_disp_x] = self.__l_disp

        if self.__r_disp is not None:
            r_disp_y, r_disp_x, _ = self.__r_disp.shape
            if l_disp_y >= w:
                return
            frame[l_disp_y:(l_disp_y + r_disp_y), 0:r_disp_x] = self.__r_disp
