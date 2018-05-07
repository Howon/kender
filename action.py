from enum import Enum

class HeadAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CENTER = 4
    ZOOMED = 5
    NOT_ZOOMED = 6

class EyeAction(Enum):
    BOTH_OPEN = 0
    BOTH_CLOSED = 1
    BOTH_BLINK = 2
    RIGHT_WINK = 3
    RIGHT_CLOSED = 4
    LEFT_WINK = 5
    LEFT_CLOSED = 6

# def translate(action):
#     MOVE_LEFT = 1
#     MOVE_RIGHT = 2
#     EXPOSE = 3
#     FULLSCREEN = 4
#     MINIMIZE = 5
#     TAB_FORWARD = 6
#     TAB_LEFT = 7
#     COPY = 8
#     PASTE = 9
