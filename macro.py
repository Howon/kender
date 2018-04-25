import pyautogui
from enum import Enum

class Macro(Enum):
    WINDOW_LEFT = 1
    WINDOW_RIGHT = 2
    EXPOSE = 3
    MINIMIZE = 4
    TAB_FORWARD = 5
    TAB_LEFT = 6
    COPY = 7
    PASTE = 8

__macros = {
    Macro.WINDOW_LEFT: [""],
    Macro.WINDOW_RIGHT: [""],
    Macro.EXPOSE: [],
    Macro.MINIMIZE: [],
    Macro.TAB_FORWARD: [],
    Macro.TAB_LEFT: [],
    Macro.COPY: [],
    Macro.PASTE: [],
}

def execute(macro):
    hotkeys = __macros[macro]

    for k in hotkeys:
        pyautogui.keyDown(k)

    for k in hotkeys[::-1]:
        pyautogui.keyUp(k)
