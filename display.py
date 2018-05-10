import cv2
from utils import put_text
from action import HeadAction, EyeAction

def display_bounds(frame):
    h, w, _ = frame.shape

    xs = [int(w / 4), int(3 * w / 4)]
    ys = [int(1 * h / 9), int(7.8 * h / 9)]

    for y in ys:
        [left, right] = xs
        cv2.line(frame, (left, y), (right, y), (255, 255, 255), 2)

def display_decisions(frame, head_action, eye_action):
	h, w, _ = frame.shape
	print("=================================")
	align_x, align_y = int(w * 0.05), int(h * 0.8)
	print(head_action)
	print(eye_action)

	put_text(frame, str(head_action)[11:],
          (align_x, align_y + 30), scale=1.5, thickness=2)
	put_text(frame, str(eye_action)[10:],
          (align_x, align_y + 3 * 30), scale=1.5, thickness=2)

def display_counters(frame, counter_logs):
    h, w, _ = frame.shape
    align_x = int(w * 0.63)
    align_y = int(h * 0.3)
    for i, (k, v) in enumerate(counter_logs.items()):
        text = "{}: {}".format(k, v).ljust(11).lower()
        put_text(frame, text, (align_x, 15 * (i + 1) + align_y))
