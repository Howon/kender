# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

#constants 
HEIGHT = 255
WIDTH = 400

LEFT_THRESHOLD = 20
RIGHT_THRESHOLD = 20
UP_THRESHOLD = 35
DOWN_THRESHOLD = 20
CENTER_LINE = WIDTH//2

DEBUG = True


""" Decision functions and their helper functions """

#computes average of two points
def compute_average(p1, p2):
	return ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)


# gets the point at a given index in the shape feature array of points
def get_shape_point(shape, i):
	return (shape[i][0], shape[i][1] )


# returns the point which is the tip of the nose
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/ 
# see ^ for how facial landmark mapping works. Just look up the index.
def get_nose_tip_point(shape):
	return get_shape_point(shape, 30) # 30 is the index which maps to nose point


# computes the center of the head which is the average of the left and right ear
def get_center_head_point(shape):
	left_ear_point = get_shape_point(shape, 2) # 2 is the index which maps to left ear
	right_ear_point = get_shape_point(shape, 16) # 16 is the index which maps to right ear
	return compute_average(left_ear_point, right_ear_point)


# computes the center of the head which is the average of the left and right ear
def get_chin_point(shape):
	return get_shape_point(shape, 9) # 9 is the index which maps to chin


# detects if the nose point has distanced itself from the center of the head.
def is_turned_left(nose_tip_point, center_head_point):
	return nose_tip_point[0] < (center_head_point[0] - LEFT_THRESHOLD)


# detects if the nose point has distanced itself from the center of the head.
def is_turned_right(nose_tip_point, center_head_point):
	return nose_tip_point[0] > (center_head_point[0] + RIGHT_THRESHOLD)


# detects if the nose point has distanced itself from the center of the head.
def is_turned_up(chin_point, center_head_point):
	return chin_point[1] < (center_head_point[1] + UP_THRESHOLD)


# detects if the nose point has distanced itself from the center of the head.
def is_turned_down(nose_tip_point, center_head_point):
	return nose_tip_point[1] > (center_head_point[1] + DOWN_THRESHOLD)


# returns what position the head is in
def check_position(shape, frame):
	# first get all our points of interest
	center_head_point = get_center_head_point(shape)
	nose_tip_point = get_nose_tip_point(shape)
	chin_point = get_chin_point(shape)

	# draw some useful information
	if DEBUG:
		cv2.circle(frame, nose_tip_point, 2, (255, 255, 255), -1)
		cv2.circle(frame, center_head_point, 2, (0, 0, 255), -1)
		cv2.circle(frame, chin_point, 1, (255, 0, 0), -1)
		frame = draw_left_line(frame, center_head_point)
		frame = draw_right_line(frame, center_head_point)
		frame = draw_up_line(frame, center_head_point)
		frame = draw_down_line(frame, center_head_point)

	# check what position the head is in
	if is_turned_left(nose_tip_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("LEFT")
		return "LEFT"
	
	elif is_turned_right(nose_tip_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("RIGHT")
		return "RIGHT"
	
	elif is_turned_up(chin_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("UP")
		return "UP"
	
	elif is_turned_down(nose_tip_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("DOWN")
		return "DOWN"
	
	else:
		if DEBUG:
			cv2.putText(frame, "CENTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("CENTER")
		return "CENTER"




""" Drawing functions """
# draws the threshold line for left turn
def draw_left_line(frame, center_head_point):
	ptA = ((center_head_point[0] - LEFT_THRESHOLD), 0)
	ptB = ((center_head_point[0] - LEFT_THRESHOLD), HEIGHT)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame


# draws the threshold line for right turn
def draw_right_line(frame, center_head_point):
	ptA = ((center_head_point[0] + RIGHT_THRESHOLD), 0)
	ptB = ((center_head_point[0] + RIGHT_THRESHOLD), HEIGHT)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame


# draws the threshold line for right turn
def draw_up_line(frame, center_head_point):
	ptA = (0, (center_head_point[1] + UP_THRESHOLD))
	ptB = (HEIGHT, (center_head_point[1] + UP_THRESHOLD))
	cv2.line(frame, ptA, ptB, (255, 0, 0), 2)
	return frame


# draws the threshold line for right turn
def draw_down_line(frame, center_head_point):
	ptA = (0, (center_head_point[1] + DOWN_THRESHOLD))
	ptB = (HEIGHT, (center_head_point[1] + DOWN_THRESHOLD))
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame


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


		print( check_position(shape, frame) )
		

	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()