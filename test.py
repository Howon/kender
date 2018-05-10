import time
import argparse

from random import shuffle
from action import HeadAction, success
from capture import capture_action

COMMANDS_LOG = "test_commands.txt"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="Path to facial landmark predictor")
ap.add_argument("-d", "--debug", action="store_true", required=False,
                help="Displays debugging information.")

args = vars(ap.parse_args())


def load_test():
    test_actions = [v for _, v in success.values() if v != HeadAction.UNZOOM]

    test_cases = test_actions * 10
    shuffle(test_cases)

    return test_cases

def main():
    global COMMANDS_LOG

    test_cases = load_test()

    with open(COMMANDS_LOG, "w+") as test_commands:
        for t in test_cases:
            test_commands.write(str(t) + "\n")

    print("Open {} in a new window and begin the test.".format(COMMANDS_LOG))
    print("Read each line to the subject and see how they perform.")

    time.sleep(5)

    def test(action):
        print("============\n{}".format(action))

    capture_action(args["shape_predictor"], test)

if __name__ == "__main__":
    main()
