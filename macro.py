import os
import json
import time
import pyautogui

from enum import Enum
from action import HeadAction, EyeAction

_SLEEP_DURATION = 0.3

class Macro(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    FULLSCREEN = 2
    EXPOSE = 3
    MINIMIZE = 4
    TAB_FORWARD = 5
    TAB_BACKWARD = 6
    COPY = 7
    PASTE = 8
    NO_ACTION = 9

__ATM = {
    HeadAction.LEFT: Macro.TAB_BACKWARD,
    HeadAction.RIGHT: Macro.TAB_FORWARD,
    HeadAction.UP: Macro.COPY,
    HeadAction.DOWN: Macro.PASTE,
    HeadAction.ZOOM: Macro.EXPOSE,
    HeadAction.UNZOOM: Macro.EXPOSE,
    EyeAction.LEFT_WINK: Macro.MOVE_LEFT,
    EyeAction.RIGHT_WINK: Macro.MOVE_RIGHT,
    EyeAction.BOTH_BLINK: Macro.FULLSCREEN,
}

def translate_action(action):
    return __ATM[action] if action in __ATM else Macro.NO_ACTION

class MacroHandler():
    __macros = {
        Macro.MOVE_LEFT: [],
        Macro.MOVE_RIGHT: [],
        Macro.EXPOSE: [],
        Macro.MINIMIZE: [],
        Macro.TAB_FORWARD: [],
        Macro.TAB_BACKWARD: [],
        Macro.COPY: [],
        Macro.PASTE: [],
    }

    def __init__(self, config_file):
        """Loads predefined Spectacle commands and populates __macros dictionary.
        """
        with open(os.path.expanduser(config_file)) as json_data:
            commands = json.load(json_data)
            for c in commands:
                name, binding = c['shortcut_name'], c['shortcut_key_binding']
                if binding is not None:
                    binding = binding.replace("cmd", "command")
                    binding = binding.split('+')

                if name == 'MoveToLeftHalf':
                    self.__macros[Macro.MOVE_LEFT] = binding
                elif name == 'MoveToRightHalf':
                    self.__macros[Macro.MOVE_RIGHT] = binding
                elif name  == 'MoveToFullscreen':
                    self.__macros[Macro.FULLSCREEN] = binding

        self.__macros[Macro.EXPOSE] = ["ctrl", "up"]
        self.__macros[Macro.MINIMIZE] = ["command", "down"]
        self.__macros[Macro.TAB_FORWARD] = ["command", "tab"]
        self.__macros[Macro.TAB_BACKWARD] = ["command", "shiftleft", "tab"]
        self.__macros[Macro.COPY] = ["command", "c"]
        self.__macros[Macro.PASTE] = ["command", "v"]

    def execute(self, macro):
        if macro == Macro.NO_ACTION:
            return

        hotkeys = self.__macros[macro]

        pyautogui.hotkey(*hotkeys, interval=0.1)

        time.sleep(_SLEEP_DURATION)
