import os, json, re
import glob
from som.visualize_closed_loop import control2string


def navi_action_match(question, answer, options):
    ACTION2NAVI = {
        "TURN_LEFT": "go left",
        "TURN_RIGHT": "go right",
        "SLOW_DOWN": "forward",
        "BRAKE": "forward",
        "KEEP_STRAIGHT": "forward",
        "SPEED_UP": "forward",
        "BIG_LEFT": "go left",
        "BIG_RIGHT": "go right"
    }
    action_code = options[answer]
    match = re.search(r'and your navigation command is \"(.*?)\".', question)
    print(question)
    print(f"({answer})")
    if match:
        extracted_text = match.group(1)
        print("Extracted navigation:", extracted_text)
        if ACTION2NAVI[action_code] == extracted_text:
            return True
        else:
            return False
    else:
        print("No match found")
        return None


def match(action, navigation):
    ACTION2NAVI = {
        "TURN_LEFT": "go left",
        "TURN_RIGHT": "go right",
        "SLOW_DOWN": "forward",
        "BRAKE": "forward",
        "KEEP_STRAIGHT": "forward",
        "SPEED_UP": "forward",
        "BIG_LEFT": "go left",
        "BIG_RIGHT": "go right"
    }
    if ACTION2NAVI[action] == navigation:
        return True
    return False


"""action_buffers = glob.glob("E:/Bolei/4b/open_loop/*/action_buffer.json")
counts = 0
total = 0
for action_buffer in action_buffers:
    buffer = json.load(open(action_buffer, "r"))
    for key, val in buffer.items():
        total += 1
        control = control2string(val["action"])
        navigation = val["navigation"]
        counts += 1 if match(control, navigation) else 0
print(f"open_loop: {counts / total}")

action_buffers = glob.glob("E:/Bolei/4b/naive/*/action_buffer.json")
counts = 0
total = 0
for action_buffer in action_buffers:
    buffer = json.load(open(action_buffer, "r"))
    for key, val in buffer.items():
        total += 1
        control = control2string(val["action"])
        navigation = val["navigation"]
        counts += 1 if match(control, navigation) else 0
print(f"naive: {counts / total}")"""

qas = json.load(open("E:/Bolei/4b/4b.json","r"))
counts = 0
total = 0
for key, value in qas.items():
    total += 1
    match = navi_action_match(value["question"], value["model_response"], value["options"])
    if match is None:
        raise ValueError
    else:
        counts += 1 if match else 0
print(counts/total, counts, total)
