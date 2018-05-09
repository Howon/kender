from enum import Enum
from collections import OrderedDict

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
    LEFT_WINK = 5
    LEFT_CLOSED = 6
    RIGHT_WINK = 3
    RIGHT_CLOSED = 4

head_action_log = OrderedDict({
    HeadAction.LEFT: 0,
    HeadAction.RIGHT: 0,
    HeadAction.UP: 0,
    HeadAction.DOWN: 0,
    HeadAction.CENTER: 0,
    HeadAction.ZOOMED: 0,
    HeadAction.NOT_ZOOMED: 0
})

eye_action_log = OrderedDict({
    EyeAction.BOTH_OPEN: 0,
    EyeAction.BOTH_BLINK: 0,
    EyeAction.BOTH_CLOSED: 0,
    EyeAction.LEFT_WINK: 0,
    EyeAction.LEFT_CLOSED: 0,
    EyeAction.RIGHT_WINK: 0,
    EyeAction.RIGHT_CLOSED: 0
})
