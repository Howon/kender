from enum import Enum

_HEAD_FRAMES = 1
_EYE_FRAMES = 2

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

HEAD_REST_STATE = HeadAction.CENTER
EYE_REST_STATE = EyeAction.BOTH_OPEN

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
        self.__awake = False
        self.__prev_action = None
        self.__counter = 0
        return

    def __consec(self, action):
        if self.__prev_action == action:
            self.__counter += 1
        else:
            counts = self.__counter
            self.__counter = 0
            a_type, h_type = type(action), type(HEAD_REST_STATE)

            if counts >= (_HEAD_FRAMES if a_type == h_type else _EYE_FRAMES):
                return True

        return False

    def __next_state(self, prev, action):
        if prev not in success:
            return False, None

        # Have to return either of the resting state.
        if not action == HEAD_REST_STATE or not action == EYE_REST_STATE:
            return False, None

        candidate = success[prev]
        c_type, a_type = type(candidate), type(action)

        return c_type == a_type, candidate

    def wake_up(self, action):
        if self.__awake:
            return True

        prev = self.__prev_action
        self.__awake = self.__consec(action) and __prev == EyeAction.BOTH_CLOSED
        return False

    def get_next(self, action):
        if self.__consec(action):
            prev = self.__prev_action
            self.__prev_action = action
            return self.__next_state(prev, action)

        self.__prev_action = action
        return False, None
