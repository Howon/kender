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
CENTER_LINE = WIDTH//2


# returns the point which is the tip of the nose
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/ 
# see ^ for landmark mapping

def compute_average(p1, p2):
	return ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)

def get_shape_point(shape, i):
	return (shape[i][0], shape[i][1] )


def get_nose_tip_point(shape):
	return get_shape_point(shape, 30)


def get_center_head_point(shape):
	left_ear_point = get_shape_point(shape, 2)
	right_ear_point = get_shape_point(shape, 16)
	return compute_average(left_ear_point, right_ear_point)

def turned_left(nose_tip_point, center_head_point):
	return nose_tip_point[0] < (center_head_point[0] - LEFT_THRESHOLD)

def turned_right(nose_tip_point, center_head_point):
	return nose_tip_point[0] > (center_head_point[0] + RIGHT_THRESHOLD)


def draw_left_line(frame, center_head_point):
	ptA = ((center_head_point[0] - LEFT_THRESHOLD), 0)
	ptB = ((center_head_point[0] - LEFT_THRESHOLD), HEIGHT)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame

def draw_right_line(frame, center_head_point):
	ptA = ((center_head_point[0] + RIGHT_THRESHOLD), 0)
	ptB = ((center_head_point[0] + RIGHT_THRESHOLD), HEIGHT)
	cv2.line(frame, ptA, ptB, (255, 255, 255), 2)
	return frame



# To Do
# should check if someone has actually *turned* their head left rather than merely moved left.
#def turned_left(point, point_data):





""" Main """
 
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

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		#for (x, y) in shape[30]:
		#	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		
		center_head_point = get_center_head_point(shape)
		nose_tip_point = get_nose_tip_point(shape)
		cv2.circle(frame, nose_tip_point, 1, (0, 0, 255), -1)
		cv2.circle(frame, center_head_point, 1, (0, 255, 255), -1)


		frame = draw_left_line(frame, center_head_point)
		frame = draw_right_line(frame, center_head_point)
		if turned_left(nose_tip_point, center_head_point):
			cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		elif turned_right(nose_tip_point, center_head_point):
			cv2.putText(frame, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)







		# if the nose tip was at the center but now it is at the left then print "left"

	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()