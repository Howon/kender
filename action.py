from enum import Enum

_HEAD_FRAMES = 2
_EYE_FRAMES = 3

class HeadAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    CENTER = 4
    ZOOM = 5
    UNZOOM = 6

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
    EyeAction.LEFT_CLOSED: ([EyeAction.BOTH_OPEN], EyeAction.LEFT_WINK),
    EyeAction.RIGHT_CLOSED: ([EyeAction.BOTH_OPEN], EyeAction.RIGHT_WINK),
    EyeAction.BOTH_CLOSED: ([EyeAction.BOTH_OPEN], EyeAction.BOTH_BLINK),
    HeadAction.LEFT: ([HeadAction.CENTER, HeadAction.ZOOM], HeadAction.LEFT),
    HeadAction.RIGHT: ([HeadAction.CENTER, HeadAction.ZOOM], HeadAction.RIGHT),
    HeadAction.UP: ([HeadAction.CENTER, HeadAction.ZOOM], HeadAction.UP),
    HeadAction.DOWN: ([HeadAction.CENTER, HeadAction.ZOOM], HeadAction.DOWN),
    HeadAction.CENTER: ([HeadAction.ZOOM], HeadAction.ZOOM),
    HeadAction.ZOOM: ([HeadAction.CENTER], HeadAction.UNZOOM)
}

h_type = type(HEAD_REST_STATE)

class ActionHandler():
    def __init__(self):
        self.__awake = False
        self.__prev_head = HEAD_REST_STATE
        self.__prev_eye = EYE_REST_STATE

        self.__eye_count = 0
        self.__head_count = 0

        self.__zoomed = False

        return

    def __consec(self, prev, action, count):
        if prev != action:
            a_type = type(action)

            if count >= (_HEAD_FRAMES if a_type == h_type else _EYE_FRAMES):
                return True, 0

        return False, count + 1

    def __next_state(self, prev, now):
        if prev not in success:
            return False, None

        req, action = success[prev]

        if now not in req:
            return False, None

        if action == HeadAction.UNZOOM:

            print("UNZOOM", self.__zoomed)
            # Don't unzoom if not zoomed.
            if not self.__zoomed:
                return False, None

            self.__zoomed = False

        if action == HeadAction.ZOOM:
            # Don't zoom twice.
            if self.__zoomed:
                return False, None
            self.__zoomed = True

        return True, action

    def get_next(self, e_action, h_action):
        e_prev, h_prev = self.__prev_eye, self.__prev_head

        if h_prev == HEAD_REST_STATE:
            e_consec, e_count = self.__consec(e_prev, e_action, self.__eye_count)
        else:
            e_consec, e_count = False, 0

        h_consec, h_count = self.__consec(h_prev, h_action, self.__head_count)

        self.__prev_eye = e_action
        self.__prev_head = h_action

        self.__eye_count = e_count
        self.__head_count = h_count

        if e_consec and h_consec or (not e_consec and h_consec):
            self.__eye_count = 0
            return self.__next_state(h_prev, h_action)
        elif e_consec:
            self.__head_count = 0
            return self.__next_state(e_prev, e_action)

        return False, None
