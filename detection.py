# import the necessary packages
import cv2
import math
import statistics
import numpy as np
from scipy.spatial import ConvexHull

DEBUG = True

#constants 
HEIGHT = 255
WIDTH = 400
CENTER_LINE = WIDTH//2

""" Decision Thresholds """
# eventually might want to make these calibrated or dynamic according to face ratio
# rather than absolute values regarding the frame dimensions
                       # to make detection more sensitive:
LEFT_THRESHOLD = 20    # decrease
RIGHT_THRESHOLD = 20   # decrease
UP_THRESHOLD = 55      # increase
DOWN_THRESHOLD = 20    # decrease
ZOOM_THRESHOLD = 60    # decrease
CLOSE_THRESHOLD = 4   # increase
EAR_CLOSE_THRESHOLD = 0.25   # increase
WINK_THRESHOLD = 0.02   # decrease

# timing constants
CLOSED_CONSEC_FRAMES = 3  #this is how many frames the signal is required to be consistent for




""" Helper functions """

#computes average of two points
def compute_average(p1, p2):
	return ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)

#gets the distance
def dist(p1,p2):
	return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# gets the point at a given index in the shape feature array of points
def get_shape_point(shape, i):
	return (shape[i][0], shape[i][1] )



""" Head Position Detection """
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
		return "LEFT"
	
	elif is_turned_right(nose_tip_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "RIGHT"
	
	elif is_turned_up(chin_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "UP"
	
	elif is_turned_down(nose_tip_point, center_head_point):
		if DEBUG:
			cv2.putText(frame, "DOWN", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "DOWN"
	
	else:
		if DEBUG:
			cv2.putText(frame, "CENTER", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		return "CENTER"








""" Head Zoom Detection """
# returns whether the head is zoomed in or not
# checks distance between center of head and right ear
def is_zoomed_in(right_ear_point, center_head_point):
	distance = dist(right_ear_point, center_head_point)

	# THRESHOLD DEBUGGER 
	# uncomment if u need to determine good threshold
	#print("distance: ", distance)

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
		cv2.line(frame, center_head_point, right_ear_point, (255, 200, 255), 2)
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





""" Eye Status Detection """
# for all eye functions the distance between the eyelid and the bottom of 
# the eye for each eye is compared to a threshold

# source: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# source: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist(eye[1], eye[5])
	B = dist(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist(eye[0], eye[3])
 
	# compute the eye aspect ratio
	EAR = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio (E.A.R.)
	return EAR


# Note that EAR = eye aspect ratio (E.A.R.)
#checks if there is a difference between the EAR of the left and right eye
def is_right_closed(left_EAR, right_EAR):
	return (left_EAR - right_EAR) > WINK_THRESHOLD and right_EAR < EAR_CLOSE_THRESHOLD


#checks if there is a difference between the EAR of the left and right eye
def is_left_closed(left_EAR, right_EAR):
	return (right_EAR - left_EAR) > WINK_THRESHOLD and left_EAR < EAR_CLOSE_THRESHOLD


#checks if left and right eye have low EAR (i.e. eyes are closed)
def is_both_closed(left_EAR, right_EAR):
	return left_EAR < EAR_CLOSE_THRESHOLD and right_EAR < EAR_CLOSE_THRESHOLD


# checks the status of the eyes
def check_eyes(shape, frame, frame_counters):

	# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
	# this is how the eyes are indexed
	#
	#   		*37 *38          	*43 *44
	#		*36       *39        *42       *45
	#   		*41 *40          	*47 *46

	# first get all our points of interest
	left_eye_points = []
	for i in range(36,42):
		left_eye_points.append(get_shape_point(shape, i))

	right_eye_points = []
	for i in range(42,48):
		right_eye_points.append(get_shape_point(shape, i))


	leftEye = shape[36:42]
	rightEye = shape[42:48]


	# calculate the Eye Aspect Ratio (EAR)
	left_EAR = eye_aspect_ratio(left_eye_points)
	right_EAR = eye_aspect_ratio(right_eye_points)


	# uncomment if you want some other possibly useful 
	# features that I had been experimenting with...
	"""
	# the following may also come in handy at some point: 
	center_head_point = get_center_head_point(shape)
	left_eye_top_left_point = get_shape_point(shape, 37)
	left_eye_top_right_point = get_shape_point(shape, 38)
	left_eye_bottom_left_point = get_shape_point(shape, 41)
	left_eye_bottom_right_point = get_shape_point(shape, 40)

	right_eye_top_left_point = get_shape_point(shape, 43)
	right_eye_top_right_point = get_shape_point(shape, 44)
	right_eye_bottom_left_point = get_shape_point(shape, 47)
	right_eye_bottom_right_point = get_shape_point(shape, 46)

	# we want to calculate the distance of these lines
	#
	#   		*37 *38          	*43 *44
	#		*36  |   | *39       *42 |   | *45
	#   		*41 *40          	*47 *46
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
	#   		*37 *38          	 *43 *44
	#		 *36   []   *39       *42   []  *45
	#   		*41 *40          	 *47 *46

	left_eye_points = np.zeros((6, 2))
	for i, j in zip(range(36,42), range(0,6)):
		left_eye_points[j,0] = get_shape_point(shape, i)[0]
		left_eye_points[j,1] = get_shape_point(shape, i)[1]
	left_eye_center = np.mean(left_eye_points, axis=0)
	left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))

	right_eye_points = np.zeros((6, 2))
	for i, j in zip(range(42,48), range(0,6)):
		right_eye_points[j,0] = get_shape_point(shape, i)[0]
		right_eye_points[j,1] = get_shape_point(shape, i)[1]
	right_eye_center = np.mean(right_eye_points, axis=0)
	right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))


	
	# get the area of the convex hull of each eye
	left_eye_hull = ConvexHull(left_eye_points)
	right_eye_hull = ConvexHull(right_eye_points)
	print(" left_eye_hull.area: ", left_eye_hull.area)
	print("right_eye_hull.area: ", right_eye_hull.area)
	print("difference: ", left_eye_hull.area - right_eye_hull.area)
	

	# another possibility...
	# analyze the pixels within the bounding box of the eye, 
	# then threshold on white, and then compare the area
	# could also threshold on other things as well...
	"""


	# draw some useful information
	if DEBUG:
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, " LEFT EAR: " + str(round(left_EAR,2)), (230, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(frame, "RIGHT EAR: " + str(round(right_EAR,2)), (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		#cv2.putText(frame, "both eyes closed count: " + str(frame_counters["both_eyes_closed"]), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# uncomment if you want to visualize some of those other features I had calculated above
		#cv2.line(frame, left_eye_top_left_point, left_eye_bottom_left_point, (255, 255, 255), 2)
		#cv2.line(frame, left_eye_top_right_point, left_eye_bottom_right_point, (255, 255, 255), 2)
		#cv2.line(frame, right_eye_top_left_point, right_eye_bottom_left_point, (255, 255, 255), 2)
		#cv2.line(frame, right_eye_top_right_point, right_eye_bottom_right_point, (255, 255, 255), 2)
		#cv2.circle(frame, left_eye_center, 1, (0, 0, 255), -1)
		#cv2.circle(frame, right_eye_center, 1, (0, 0, 255), -1)
		#cv2.line(frame, left_eye_center, right_eye_center, (255, 255, 255), 2)


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



	status = "EYES OPEN"

	# blink
	if is_both_closed(left_EAR, right_EAR):
		frame_counters["both_eyes_closed"] += 1
		status = "EYES CLOSED"
		cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		if frame_counters["both_eyes_closed"] > CLOSED_CONSEC_FRAMES:
			status = "EYES BLINKED"
		frame_counters["both_eyes_closed"] = 0

	# right wink
	if is_right_closed(left_EAR, right_EAR):
		frame_counters["right_eye_closed"] += 1
		status = "RIGHT EYE CLOSED"
		cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		if frame_counters["right_eye_closed"] > CLOSED_CONSEC_FRAMES:
			status = "RIGHT WINK"
		frame_counters["right_eye_closed"] = 0

	# left wink
	if is_left_closed(left_EAR, right_EAR):
		frame_counters["left_eye_closed"] += 1
		status = "LEFT EYE CLOSED"
		cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		if frame_counters["left_eye_closed"] > CLOSED_CONSEC_FRAMES:
			status = "LEFT WINK"
		frame_counters["left_eye_closed"] = 0
		

	return status, frame_counters
	



""" Drawing functions """
# draws the threshold line for left turn
def draw_left_line(frame, center_head_point):
	ptA = ((center_head_point[0] - LEFT_THRESHOLD), 90)
	ptB = ((center_head_point[0] - LEFT_THRESHOLD), 115)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame


# draws the threshold line for right turn
def draw_right_line(frame, center_head_point):
	ptA = ((center_head_point[0] + RIGHT_THRESHOLD), 90)
	ptB = ((center_head_point[0] + RIGHT_THRESHOLD), 115)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame


# draws the threshold line for right turn
def draw_up_line(frame, center_head_point):
	ptA = (170, (center_head_point[1] + UP_THRESHOLD))
	ptB = (230, (center_head_point[1] + UP_THRESHOLD))
	cv2.line(frame, ptA, ptB, (255, 0, 0), 2)
	return frame


# draws the threshold line for right turn
def draw_down_line(frame, center_head_point):
	ptA = (170, (center_head_point[1] + DOWN_THRESHOLD))
	ptB = (230, (center_head_point[1] + DOWN_THRESHOLD))
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame