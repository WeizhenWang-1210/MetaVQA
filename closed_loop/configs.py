from collections import defaultdict

ACTION_STATISTICS = defaultdict(lambda: 0)

RECORD_BUFFER = defaultdict(lambda: defaultdict(lambda: dict()))

INTERVENED = {
    "a": [0.15, 0.8],  # turn_left
    "b": [-0.15, 0.8],  # turn_right
    "c": [0, -0.135],  # slow_down
    "d": [0, -0.26],  # brake_now
    "e": [0, 0.15],  # keep_straight
    "f": [0, 0.3],  # speed_up
    "g": [0.6, 0.2],  # big_left
    "h": [-0.6, 0.2]  # big_right
}

NON_INTERVENED = INTERVENED

ACTION2OPTION = {
    "TURN_LEFT": "A", "TURN_RIGHT": "B", "SLOW_DOWN": "C", "BRAKE": "D", "KEEP_STRAIGHT": "E",
    "SPEED_UP": "F", "BIG_LEFT": "G", "BIG_RIGHT": "H"
}

RECORD_FOLDER = "/home/weizhen/closed_loops"

def convert_action(action, intervened):
    if intervened:
        ACTION_MAPPING = INTERVENED
    else:
        ACTION_MAPPING = NON_INTERVENED
    if action in ACTION_MAPPING.keys():
        return ACTION_MAPPING[action]
    else:
        return ACTION_MAPPING["e"]
