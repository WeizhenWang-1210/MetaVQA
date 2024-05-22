import json
import os.path

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
        rounded_box = [[round(x, 1), round(y, 1)] for x, y in ordered_box]
        final_boxes.append(rounded_box)
    return final_boxes

def round_trajectory(points):
    final_points = []
    for point in points:
        rounded_point = [round(point[0], 1), round(point[1], 1)]
        final_points.append(str(rounded_point))
    return ",".join(final_points)


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
        if record["question_type"] in ["localization"]:
            print("localization will order")
            if len(record["answer"]) <= 8:
                print("bruh")
                record["answer"] = order_and_round(record["answer"])
            else:
                continue
        elif record["question_type"] in ["predict_trajectory"]:
            print("trajectory_prediction will also order")
            print("bruh")
            record["answer"] = round_trajectory(record["answer"])
        elif record["question_type"] in ["identify_speed"]:
            print("round speed to one point after decimal.")
            record["answer"] = round(record["answer"], 1)
        #elif record[""]
        processed_qa[count] = record
        count += 1
    return processed_qa

def postprocess_localization(raw_answer):
    #ordered_boxes = order_and_round(raw_answer)
    final_answer = []
    raw_answer.sort(key=lambda bbox: norm(centroid(bbox)))
    for box in raw_answer:
        center = centroid(box)
        center = [round(center[0],1), round(center[1],1)]
        stringed = str(center)
        final_answer.append(stringed)
    final_answer = ",".join(final_answer)
    return final_answer

def postprocess_predict_trajectory(raw_answer):
    rounded_trajectory = round_trajectory(raw_answer)
    return rounded_trajectory

def postprocess_counting(raw_answer):
    return str(raw_answer[0])

def postprocess_predicates(raw_answer):
    return str(raw_answer).capitalize()

def postprocess_type_identification(raw_answer):
    final_answer = []
    for answer in raw_answer:
        if answer == "Suv":
            final_answer.append(answer.upper())
        else:
            final_answer.append(answer)
    return ";".join(final_answer)
def postprocess_color_identification(raw_answer):
    return ";".join(raw_answer)

def postprocess_speed(raw_answer):
    return str(round(raw_answer,1))

def postprocess_type_identification_unique(raw_answer):
    if raw_answer == "Suv":
        return raw_answer.upper()
    else:
        return raw_answer




def postprocess_qa(qa_records):
    """
    Order boxes/vertices
    For questions with too answers too much, ignore it.
    Convert floating to one decimal point
    #add a question that converts centerpooint to bbox?
    """
    count = 0
    processed_qa = {}
    processor_mapping = dict(
        localization=postprocess_localization,
        counting=postprocess_counting,
        count_equal_binary=postprocess_predicates,
        count_more_binary=postprocess_predicates,
        color_identification=postprocess_color_identification, color_identification_unique=lambda x: x,
        type_identification=postprocess_type_identification, type_identification_unique=postprocess_type_identification_unique,
        identify_stationary=postprocess_predicates, identify_turning=postprocess_predicates, identify_acceleration=postprocess_predicates,
        identify_speed=postprocess_speed, identify_heading=lambda x: str(x), identify_head_toward=postprocess_predicates,
        predict_trajectory=postprocess_predict_trajectory,
    )
    for id, record in qa_records.items():
        type_string = record["question_type"]
        concrete_type = type_string.split("|")[-1]
        if concrete_type == "localization":
            if len(record["answer"]) > 8:
                continue
        postprocessor = processor_mapping[concrete_type]
        record["answer"] = postprocessor(record["answer"])
        processed_qa[count] = record
        count += 1
    print(f"Processed {count} valid tuples.")
    return processed_qa

def process_files(qa_record_paths, destination_folder):
    for qa_record_path in qa_record_paths:
        name = os.path.basename(qa_record_path)
        final_destination = os.path.join(destination_folder, name)
        print(f"Postprocessing {qa_record_path} to {final_destination}")
        qa_record = json.load(open(qa_record_path, "r"))
        processed = postprocess_qa(qa_record)
        json.dump(processed, open(final_destination,"w"))


def process_session(session_folder, destination_folder):
    qas = os.listdir(session_folder)
    qa_paths = [os.path.join(session_folder,qa) for qa in qas]
    process_files(qa_paths, destination_folder)





if __name__ == "__main__":
    # open_to_close_vocab("/bigdata/weizhen/metavqa/100k_export/exported.json", "/bigdata/weizhen/metavqa/100k_export/converted.json")
    #qa_records = json.load(open("./verification_multiview_small/static.json", "r"))
    #json.dump(qa_cleaning(qa_records), open("./verification_multiview_small/filterd.json", "w"), indent=2)
    #import json
    #sample = json.load(open("/bigdata/weizhen/metavqa_final/vqa/validation/multi_frame/dynamic_qa0.json", "r"))
    #print()
    #print(sample["70"]["answer"])
    #final_answer = postprocess_predict_trajectory(sample["70"]["answer"])
    #print(final_answer)
    #processed_sample = postprocess_qa(sample)
    #json.dump(processed_sample,open("/bigdata/weizhen/metavqa_final/vqa/validation/multi_frame/processed_dynamic_qa0.json","w"), indent=2)
    process_session("/bigdata/weizhen/metavqa_final/vqa/training/single_frame/", "/bigdata/weizhen/metavqa_final/vqa/training/single_frame_processed/")