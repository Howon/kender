import argparse

from capture import capture_action
from macro import MacroHandler, translate_action

_MACROS = '~/Library/Application Support/Spectacle/Shortcuts.json'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--macros", action="store_true", required=False,
                help="Enables macros.")

ap.add_argument("-p", "--shape-predictor", required=True,
                help="Path to facial landmark predictor")
ap.add_argument("-d", "--debug", action="store_true", required=False,
                help="Displays debugging information.")
ap.add_argument("-l", "--log", action="store_true", required=False,
                help="Displays logging information.")
args = vars(ap.parse_args())


def main():
    pred_path = args["shape_predictor"]
    enable_macros = args["macros"]
    macro_handler = MacroHandler(_MACROS)

    def trigger_macro(action):
        nonlocal enable_macros

        if enable_macros:
            macro = translate_action(action)
            macro_handler.execute(macro)

    capture_action(pred_path, trigger_macro, args["debug"], args["log"])

if __name__ == "__main__":
    main()
