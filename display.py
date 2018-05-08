import cv2
from utils import put_text
from action import HeadAction, EyeAction

def display_decisions(frame, head_action, zoom_action, eye_action):
	h, w, _ = frame.shape
	print("=================================")
	align_x, align_y = int(w * 0.1), int(h * 0.1)
	print(head_action)
	print(zoom_action)
	print(eye_action)

	put_text(frame, str(head_action)[11:], (align_x, align_y + 30))
	put_text(frame, str(zoom_action)[11:], (align_x, align_y + 2 * 30))
	put_text(frame, str(eye_action)[10:], (align_x, align_y + 3 * 30))


def display_counters(frame, head_action_log, eye_action_log):
	h, w, _ = frame.shape
	align_x = int(w*0.68)
	align_y = 40
	put_text(frame, "      Total", (align_x, 20))
	put_text(frame, "     blinks: " + str(eye_action_log[EyeAction.BOTH_BLINK] ), (align_x, align_y))
	put_text(frame, " left winks: " + str(eye_action_log[EyeAction.LEFT_WINK]), (align_x, align_y+15))
	put_text(frame, "right winks: " + str(eye_action_log[EyeAction.RIGHT_WINK]), (align_x, align_y+2*15))
	put_text(frame, "       left: " + str(head_action_log[HeadAction.LEFT]), (align_x, align_y+3*15))
	put_text(frame, "      right: " + str(head_action_log[HeadAction.RIGHT]), (align_x, align_y+4*15))
	put_text(frame, "         up: " + str(head_action_log[HeadAction.UP]), (align_x, align_y+5*15))
	put_text(frame, "       down: " + str(head_action_log[HeadAction.DOWN]), (align_x, align_y+6*15))
	put_text(frame, "     zoomed: " + str(head_action_log[HeadAction.ZOOMED]), (align_x, align_y+7*15))
