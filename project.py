# import the necessary packages
from imutils.video import VideoStream
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
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = cv2.flip( frame, 1 )
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		if CALIBRATE:
			cv2.line(frame, (177, 175), (237, 175), (0, 0, 0), 2)
			cv2.putText(frame, "make chin tangent to black line", (75, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(frame, "make hair tangent to top of frame", (75, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(frame, "for best results", (75, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

		#print( detection.check_position(shape, frame) )
		#print( detection.check_zoom(shape, frame) )
		print( detection.check_eyes(shape, frame) )
		

	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()