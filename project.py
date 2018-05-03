# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import datetime
import cv2
import dlib
import imutils
import math
import statistics
import time



#constants 
HEIGHT = 255
WIDTH = 400
CENTER_LINE = WIDTH//2

""" Decision Thresholds """
# thresholds work best for person whose head is 1.5-2.0 ft away from camera
# eventually might want to make these calibrated or dynamic according to face ratio
                       # to make it more sensitive:
LEFT_THRESHOLD = 20    # decrease
RIGHT_THRESHOLD = 20   # decrease
UP_THRESHOLD = 45      # increase
DOWN_THRESHOLD = 20    # decrease
ZOOM_THRESHOLD = 60    # decrease
CLOSE_THRESHOLD = 4.5   # increase

DEBUG = True


""" Decision functions and their helper functions """

#computes average of two points
def compute_average(p1, p2):
	return ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)

#gets the distance
def dist(p1,p2):
	return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

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
			cv2.putText(frame, "DOWN", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("DOWN")
		return "DOWN"
	
	else:
		if DEBUG:
			cv2.putText(frame, "CENTER", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#print("CENTER")
		return "CENTER"

# returns whether the head is zoomed in or not
# checks distance between center of head and right ear
def is_zoomed_in(right_ear_point, center_head_point):
	distance = dist(right_ear_point, center_head_point)
	#print("distance: ", distance) #uncomment if u need to figure out threshold
	return distance > ZOOM_THRESHOLD 


# checks the zoom position of the head
def check_zoom(shape, frame):
	# first get all our points of interest
	center_head_point = get_center_head_point(shape)
	right_ear_point = get_shape_point(shape, 16)

	# draw some useful information
	if DEBUG:
		cv2.circle(frame, center_head_point, 2, (0, 0, 255), -1)
		cv2.circle(frame, right_ear_point, 1, (255, 0, 0), -1)
		cv2.line(frame, center_head_point, right_ear_point, (255, 255, 255), 2)
		# white line needs to be larger than this pink line according to zoom threshold
		cv2.line(frame, (200, 105), (258, 87), (255, 200, 255), 2)  


	# check what position the head is in
	if is_zoomed_in(right_ear_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "ZOOMED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "ZOOMED"
	else:
		if DEBUG:
			cv2.putText(frame, "NOT ZOOMED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "NOT ZOOMED"



# for all eye functions the distance between the eyelid and the bottom of 
# the eye for each eye is compared to a threshold

#checks if left eye is not shut but right eye is shut i.e. blink
def is_right_blinking(left_eye_close_dist, right_eye_close_dist):
	return left_eye_close_dist > CLOSE_THRESHOLD and right_eye_close_dist < CLOSE_THRESHOLD


#checks if left eye is shut but right eye is not shut i.e. blink
def is_left_blinking(left_eye_close_dist, right_eye_close_dist):
	return left_eye_close_dist < CLOSE_THRESHOLD and right_eye_close_dist > CLOSE_THRESHOLD


#checks if left eye is shut and right eye is shut i.e. eyes are closed
def is_closed(left_eye_close_dist, right_eye_close_dist):
	return left_eye_close_dist < CLOSE_THRESHOLD and right_eye_close_dist < CLOSE_THRESHOLD


# checks the status of the eyes
def check_eyes(shape, frame):
	# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
	# this is how the eyes are indexed
	#
	#   		*37 *38          	*43 *44
	#		*36       *39        *42       *45
	#   		*41 *40          	*47 *46

	# first get all our points of interest
	center_head_point = get_center_head_point(shape)
	left_eye_top_left_point = get_shape_point(shape, 37)
	left_eye_top_right_point = get_shape_point(shape, 38)
	left_eye_bottom_left_point = get_shape_point(shape, 41)
	left_eye_bottom_right_point = get_shape_point(shape, 40)

	right_eye_top_left_point = get_shape_point(shape, 43)
	right_eye_top_right_point = get_shape_point(shape, 44)
	right_eye_bottom_left_point = get_shape_point(shape, 47)
	right_eye_bottom_right_point = get_shape_point(shape, 46)

	# calculate useful information
	left_eye_left_dist = dist(left_eye_top_left_point, left_eye_bottom_left_point)
	left_eye_right_dist = dist(left_eye_top_right_point, left_eye_bottom_right_point)

	right_eye_left_dist = dist(right_eye_top_left_point, right_eye_bottom_left_point)
	right_eye_right_dist = dist(right_eye_top_right_point, right_eye_bottom_right_point)

	left_eye_close_dist = statistics.mean([left_eye_left_dist, left_eye_right_dist])
	right_eye_close_dist = statistics.mean([right_eye_left_dist, right_eye_right_dist])

	print(" left_eye_close_dist: ", round(left_eye_close_dist,2))
	print("right_eye_close_dist: ", round(right_eye_close_dist,2))
	print("")


	# draw some useful information
	if DEBUG:
		cv2.line(frame, left_eye_top_left_point, left_eye_bottom_left_point, (255, 255, 255), 2)
		cv2.line(frame, left_eye_top_right_point, left_eye_bottom_right_point, (255, 255, 255), 2)
		cv2.line(frame, right_eye_top_left_point, right_eye_bottom_left_point, (255, 255, 255), 2)
		cv2.line(frame, right_eye_top_right_point, right_eye_bottom_right_point, (255, 255, 255), 2)

		# this is for major debugging. uncomment if the eye blinking is completely wack
		# each index might be off by 1 which would cause everything to mess up...
		# you can check that the eye indexes are what you think they are this way
		"""
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		i = 30
		for (x, y) in shape[30:50]:
			#cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			#uncomment to check indexes
			cv2.putText(frame, "*"+str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
			i+=1
		"""

	# check the status of the eyes
	if is_right_blinking(left_eye_close_dist, right_eye_close_dist):
		if DEBUG:
			cv2.putText(frame, "RIGHT BLINK", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "RIGHT BINK"
	elif is_left_blinking(left_eye_close_dist, right_eye_close_dist):
		if DEBUG:
			cv2.putText(frame, "LEFT BLINK", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "LEFT BINK"
	elif is_closed(left_eye_close_dist, right_eye_close_dist):
		if DEBUG:
			cv2.putText(frame, "EYES CLOSED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "EYES CLOSED"
	else:
		if DEBUG:
			cv2.putText(frame, "EYES OPEN", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "EYES OPEN"


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

#simply draw a line between two points
#def draw_line(frame, pointA, pointB):
#	cv2.line(frame, pointA, pointB, (255, 255, 255), 2)
#	return frame


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


		#print( check_position(shape, frame) )
		#print( check_zoom(shape, frame) )
		print( check_eyes(shape, frame) )
		

	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()