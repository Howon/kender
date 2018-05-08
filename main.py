import argparse
import datetime
import cv2
import dlib
import math
import time

from utils import put_text, resize_frame
from imutils import face_utils
#from action import translate
from detection import detect_head, detect_eyes
from action import HeadAction, EyeAction
#constants
CALIBRATE = True


""" Main """

# facial detection and real time processing credit goes to:
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
# it's ok to use this since its open source and we are allowed to use facial detection packages

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="Path to facial landmark predictor")
args = vars(ap.parse_args())

def main():
    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    camera = cv2.VideoCapture(0)

    # frame counter stored counts of how many frames have passed for a certain detection
    frame_counters = {
        "both_eyes_closed" : 0,
        "left_blink" : 0,
        "right_blink" : 0
    }

    # eye status counter
    total_blinks = 0
    total_left_winks = 0
    total_right_winks = 0

    # head state counter
    total_left = 0
    total_right = 0
    total_up = 0
    total_down = 0
    total_zoomed = 0

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        frame = resize_frame(frame)
        print(frame.shape)
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # get the status of areas we are interested in
            head_state, zoom_state = detect_head(shape, frame)
            eye_status = detect_eyes(shape, frame, frame_counters)

            # display decisions
            print("=================================")
            align_x = int(w*0.1)
            align_y = int(h*0.1)
            print(head_state)
            print(zoom_state)
            print(eye_status)
            #
            put_text(frame, str(head_state)[11:], (align_x, align_y + 30))
            put_text(frame, str(zoom_state)[11:], (align_x, align_y + 2*30))
            put_text(frame, str(eye_status)[10:], (align_x, align_y + 3*30))

            # record and display count of blinks and winks for accuracy purposes
            # if eye_status == EyeAction.BOTH_BLINK:
            #     total_blinks += 1
            # if eye_status == EyeAction.LEFT_WINK:
            #     total_left_winks += 1
            # if eye_status == EyeAction.RIGHT_WINK:
            #     total_right_winks += 1
            # if head_state == HeadAction.LEFT:
            #     total_left += 1
            # if head_state == HeadAction.RIGHT:
            #     total_right += 1
            # if head_state == HeadAction.UP:
            #     total_up += 1
            # if head_state == HeadAction.DOWN:
            #     total_down += 1
            # if zoom_state == HeadAction.ZOOMED:
            #     total_zoomed += 1
            #
            # align_x = int(w*0.68)
            # put_text(frame, "      Total", (align_x, 20))
            # put_text(frame, "     blinks: " + str(total_blinks), (align_x, 40))
            # put_text(frame, " left winks: " + str(total_left_winks), (align_x, 55))
            # put_text(frame, "right winks: " + str(total_right_winks), (align_x, 70))

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
