from capture import capture_action
from macro import MacroHandler, translate_action

_MACROS = '~/Library/Application Support/Spectacle/Shortcuts.json'

def main():
    macro_handler = MacroHandler(_MACROS)

    def trigger_macro(action):
        macro = translate_action(action)
        macro_handler.execute(macro)

    capture_action(trigger_macro)

if __name__ == "__main__":
    main()
