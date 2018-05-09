import cv2
from utils import put_text
from action import HeadAction, EyeAction

def display_decisions(frame, head_action, eye_action):
	h, w, _ = frame.shape
	print("=================================")
	align_x, align_y = int(w * 0.1), int(h * 0.1)
	print(head_action)
	print(eye_action)

	put_text(frame, str(head_action)[11:], (align_x, align_y + 30))
	put_text(frame, str(eye_action)[10:], (align_x, align_y + 2 * 30))

def display_counters(frame, counter_logs):
    h, w, _ = frame.shape
    align_x = int(w * 0.6)
    align_y = int(h * 0.3)
    put_text(frame, "      Total", (align_x, 20))

    for i, (k, v) in enumerate(counter_logs.items()):
        text = "{}: {}".format(k, v).ljust(11).lower()
        put_text(frame, text, (align_x, 15 * (i + 1) + align_y))
