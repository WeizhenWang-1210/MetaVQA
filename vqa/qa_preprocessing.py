import json

import numpy as np
from dataset_utils import order_points_clockwise, centroid, norm

# Grid x: 0,50
# Grid y: -50, 50
grid_points = []
num_space = list(range(0, 50))
bool_space = ["true", "false"]

# Outer loop for x values
for x in range(0, 51):  # 51 because range is exclusive at the end
    # Inner loop for y values
    for y in range(-50, 51):  # Similarly, 51 to include 50
        # Append tuple to list
        grid_points.append((x, y))
joint_space = grid_points + num_space + bool_space
# print(joint_space)

answer_space = {key: id for id, key in enumerate(joint_space)}
answer_space_reversed = {id: key for id, key in enumerate(joint_space)}


def open_to_close_vocab(path, converted_path):
    print(len(answer_space))
    int_offset = len(grid_points)
    bool_offset = int_offset + len(num_space)
    grid_array = np.array(grid_points)
    # print(grid_array)
    with open(path, "r") as file:
        qa_pairs = json.load(file)
    for id, info in qa_pairs.items():
        if info["question_type"] == "localization":
            converted_answers = []
            for box in info["answer"]:
                core_array = np.zeros(2)
                for point in box:
                    core_array += np.array(point)
                core_array /= 4
                distance_sq = np.sum(((grid_array - core_array) ** 2), axis=1)
                # print(distance_sq)
                closest_index = np.argmin(distance_sq)
                # print(closest_index, answer_space[tuple(grid_array[closest_index].tolist())] )
                converted_answers.append(int(closest_index))
            # print(converted_answers)
            qa_pairs[id]["answer"] = converted_answers
        elif info["question_type"] == "counting":
            # print(info["answer"][0], answer_space[info["answer"][0]])
            qa_pairs[id]["answer"] = [answer_space[info["answer"][0]]]
        else:
            str_rep = "true" if info["answer"] else "false"
            qa_pairs[id]["answer"] = [answer_space[str_rep]]
    new_qa = {}
    new_qa["qas"] = qa_pairs
    new_qa["answer_space"] = len(answer_space)
    with open(converted_path, "w") as f:
        json.dump(new_qa, f, indent=2)


def order_and_round(bboxes):
    """
    first sort by order
    then counterclockwise
    """
    final_boxes = []
    bboxes.sort(key=lambda bbox: norm(centroid(bbox)))  # since the coordinates are already in ego view
    for bbox in bboxes:
        ordered_box = order_points_clockwise(bbox)
        rounded_box = [(round(x, 1), round(y, 1)) for x, y in ordered_box]
        final_boxes.append(rounded_box)
    return final_boxes


def qa_cleaning(qa_records):
    """
    Order boxes/vertices
    For questions with too answers too much, ignore it.
    Convert floating to one decimal point
    #add a question that converts centerpooint to bbox?
    """
    count = 0
    processed_qa = {}
    for id, record in qa_records.items():
        if record["question_type"] == "localization":
            print("hree")
            if len(record["answer"]) <= 8:
                print("bruh")
                record["answer"] = order_and_round(record["answer"])
            else:
                continue
        processed_qa[count] = record
        count += 1
    return processed_qa


if __name__ == "__main__":
    # open_to_close_vocab("/bigdata/weizhen/metavqa/100k_export/exported.json", "/bigdata/weizhen/metavqa/100k_export/converted.json")
    qa_records = json.load(open("./verification_multiview_small/static.json", "r"))
    json.dump(qa_cleaning(qa_records), open("./verification_multiview_small/filterd.json", "w"), indent=2)