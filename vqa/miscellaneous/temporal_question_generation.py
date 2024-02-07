import argparse
import os
import shutil
from typing import List
from vqa.question_generation import *
from vqa.object_node import ObjectNode
from vqa.miscellaneous.temporal_question_generator import TempQueryAnswerer, TemporalTracker

temporal_example = dict(
    format = "temporal",
    paths = [
        [
            dict(
                obj = "car",
                action = "passed by"
                ),
        ],
    ],
    end = "Describe"
)
change_term = {
    "visible": {
        (True, False): "passed by",
        (False, True): "showed up"
    },
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--step", type=str, default="verification/10_40/world_10_40")
    parser.add_argument("--step_begin", type=int, default= 20)
    parser.add_argument("--step_end", type=int, default=201)
    parser.add_argument("--step_jump", type=int, default=20)
    args = parser.parse_args()
    tempTracker = TemporalTracker(change_term=change_term, track_feature_list = ["visible"])
    previous_step_num = None
    for step_num in range(args.step_begin, args.step_end, args.step_jump):
        filename = "verification/10_{}/world_10_{}.json".format(step_num, step_num)
        with open(filename, 'r') as scene_file:
            scene_dict = json.load(scene_file)
            agent_id, nodes = nodify(scene_dict)
            graph = SceneGraph(agent_id, nodes)
            tempTracker.update(graph)
            test = QuestionSpecifier(temporal_example, graph)
            # test = QuestionSpecifier(logic_example, graph)
            prophet = TempQueryAnswerer(graph, [])
            question = test.translate_to_En()
            ans = prophet.ans(tempTracker=tempTracker, query=test.translate_temporal_to_Q())
            print(question)
            # print(prophet.ans(test.translate_to_Q()))
            print(ans)
            if previous_step_num is not None:
                new_folder = f"verification/temp_10_{previous_step_num}_{step_num}"
                os.makedirs(new_folder, exist_ok=True)

                # Copy images
                for num in [previous_step_num, step_num]:
                    src_image = f"verification/10_{num}/front_10_{num}.png"
                    if os.path.exists(src_image):
                        shutil.copy(src_image, new_folder)
                    src_image = f"verification/10_{num}/rgb_10_{num}.png"
                    if os.path.exists(src_image):
                        shutil.copy(src_image, new_folder)

                # Create QA text file
                with open(os.path.join(new_folder, "QA.txt"), 'w') as qa_file:
                    qa_file.write("Question: {}\n".format(question))
                    qa_file.write("Answer: {}".format(ans))
        previous_step_num = step_num
