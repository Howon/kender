# import the necessary packages
# therefore im just making a list... ¯\_(ツ)_/¯
from imutils import face_utils
import argparse
import datetime
import cv2
import detection
import dlib
import imutils
import math
import statistics
import time

#constants
CALIBRATE = True


""" Main """

# facial detection and real time processing credit goes to:
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
# it's ok to use this since its open source and we are allowed to use facial detection packages

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

def main():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    camera = cv2.VideoCapture(0)


    time.sleep(2.0)

# I'd prefer to have this dictionary but
# it's really odd but python kept throwing an error for this dictionary
# once it was passed into check_eyes()

# frame counter
    frame_counters = {
        "both_eyes_closed" : 0,
        "left_eye_closed" : 0,
        "right_eye_closed" : 0
    }

    # eye status counter
    total_blinks = 0
    total_left_winks = 0
    total_right_winks = 0

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        _, frame = camera.read()
        frame = cv2.flip( frame, 1 )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # optionally instruct user to position head in specific spot
            if CALIBRATE:
                cv2.line(frame, (177, 175), (237, 175), (0, 0, 0), 2)
                cv2.putText(frame, "make chin tangent to black line", (75, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "make hair tangent to top of frame", (75, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "for best results", (75, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # get the status of areas we are interested in
            head_status = detection.check_position(shape, frame)
            zoom_status = detection.check_zoom(shape, frame)
            eye_status, frame_counters = detection.check_eyes(shape, frame, frame_counters)

            print("=================================")
            print("head_status: ", head_status )
            print("zoom_status: ", zoom_status)
            print(" eye_status: ", eye_status)

            # if head_status == "RIGHT":
                # move window to the right
            # if eye_status = "EYES BLINKED"
                # do something....
            # etc...

            # record and display count of blinks and winks for accuracy purposes
            if eye_status == "EYES BLINKED":
                total_blinks +=1
            if eye_status == "LEFT WINK":
                total_left_winks +=1
            if eye_status == "RIGHT WINK":
                total_right_winks +=1
            cv2.putText(frame, "total blinks: " + str(total_blinks), (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "total left winks: " + str(total_left_winks), (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "total right winks: " + str(total_right_winks), (230, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            """
            if is_closed(left_EAR, right_EAR):
                COUNTER_CLOSED += 1
            else:
                if COUNTER_CLOSED >= CLOSED_CONSEC_FRAMES:
                    cv2.putText(frame, "EYES CLOSED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return "EYES CLOSED"
                COUNTER_CLOSED = 0
            """

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
