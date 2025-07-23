import json, os, pickle
import numpy as np
import glob
from som.closed_loop_utils import computeADE

session2trail = {
    "/home/weizhen/hope/zeroshot":"InternVL-4B",
    "/home/weizhen/hope/waymonusc":"InternVL-4B-tuned",
    "/home/weizhen/hope/8b_zeroshot":"InternVL-8B ",
    "/home/weizhen/hope/8b_finetuned":"InternVL-8B-tuned ",
}


sessions = [
    "/home/weizhen/hope/zeroshot",
    "/home/weizhen/hope/waymonusc",
    "/home/weizhen/hope/8b_zeroshot",
    "/home/weizhen/hope/8b_finetuned",
]
data_dir = "/data_weizhen/scenarios"
logs = dict()

for session in sessions:
    action_buffers = f"{session}/*/action_buffer.json"
    action_buffers = glob.glob(action_buffers)
    log = dict()
    for action_buffer in action_buffers:
        action_buffer = json.load(open(action_buffer, "r"))
        agent_positions = []
        for key, value in action_buffer.items():
            position = value["state"][0]
            agent_positions.append((key, position))
            scene = os.path.join(data_dir, value["scene"])
        log[scene] = agent_positions
    logs[session2trail[session]] = log

print(len, logs.keys())

# {
#   trial: {
#       pkl_file:[
#           (t=0, [x,y]), (t=5, [x,y]), ....
#       ]
#   }
# }

json.dump(logs,
    open("/home/weizhen/logs.json","w"),
    indent=1
)






