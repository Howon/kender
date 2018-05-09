from enum import Enum

CONSEC_FRAMES = 2

class HeadAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CENTER = 4
    ZOOM = 5
    NOT_ZOOM = 6

class EyeAction(Enum):
    BOTH_OPEN = 0
    BOTH_CLOSED = 1
    BOTH_BLINK = 2
    LEFT_WINK = 5
    LEFT_CLOSED = 6
    RIGHT_WINK = 3
    RIGHT_CLOSED = 4

success = {
    EyeAction.LEFT_CLOSED: EyeAction.LEFT_WINK,
    EyeAction.RIGHT_CLOSED: EyeAction.RIGHT_WINK,
    EyeAction.BOTH_CLOSED: EyeAction.BOTH_BLINK,
    HeadAction.LEFT: HeadAction.LEFT,
    HeadAction.RIGHT: HeadAction.RIGHT,
    HeadAction.UP: HeadAction.UP,
    HeadAction.DOWN: HeadAction.DOWN,
    HeadAction.ZOOM: HeadAction.ZOOM
}

class ActionHandler():
    def __init__(self):
        self.__prev_action = None
        self.__counter = 0
        return

    def __consec(self, action):
        if self.__prev_action == action:
            self.__counter += 1
        else:
            counts = self.__counter
            self.__counter = 0

            if counts >= CONSEC_FRAMES:
                return True

    def get_next(self, action):
        if self.__consec(action):
            prev = self.__prev_action
            self.__prev_action = action
            return prev in success, success[prev] if prev in success else None

        self.__prev_action = action
        return False, None
