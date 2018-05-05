import os
import json
import pyautogui

from enum import Enum

__SPECTACLE_MACROS = '~/Library/Application Support/Spectacle/Shortcuts.json'

class Macro(Enum):
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    EXPOSE = 3
    FULLSCREEN = 4
    MINIMIZE = 5
    TAB_FORWARD = 6
    TAB_LEFT = 7
    COPY = 8
    PASTE = 9

class MacroHandler():
    __macros = {
        Macro.MOVE_LEFT: [],
        Macro.MOVE_RIGHT: [],
        Macro.EXPOSE: [],
        Macro.MINIMIZE: [],
        Macro.TAB_FORWARD: [],
        Macro.TAB_LEFT: [],
        Macro.COPY: [],
        Macro.PASTE: [],
    }

    def __init__(self, config_file):
        """Loads predefined Spectacle commands and populates __macros dictionary.
        """
        with open(os.path.expanduser(config_file)) as json_data:
            c = json.load(json_data)
            self.__macros[Macro.MOVE_LEFT] = c['MoveToLeftHalf'].split("+")
            self.__macros[Macro.MOVE_RIGHT] = c['MoveToRightHalf'].split("+")
            self.__macros[Macro.FULLSCREEN] = c['MoveToFullscreen'].split("+")

        print self.__macros

    def execute(macro):
        hotkeys = __macros[macro]

        for k in hotkeys:
            pyautogui.keyDown(k)

        for k in hotkeys[::-1]:
            pyautogui.keyUp(k)
