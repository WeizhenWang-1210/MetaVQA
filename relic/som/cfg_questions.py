import json
import os

from vqa.vqagen.utils.qa_utils import enumerate_frame_labels
from vqa.static_question_generation import generate_all_frame


def cfg_adapater(frame_path, question_type, perspective, verbose, id2label_path: str = None):
    identifier = os.path.basename(frame_path)
    world_path = os.path.join(frame_path, "world_{}.json".format(identifier))
    world = json.load(open(world_path, "r"))
    label2id, invalid_ids = enumerate_frame_labels(frame_path, perspective, id2label_path)

    records, counts = generate_all_frame(
        templates=
        {"counting": {
            "text": [
                "How many <o> can be observed at this moment?",
                "How many <o> are there in observation currently?",
                "Give me the number of <o> at this moment.",
                "Count the number of observed <o> at current moment.",
                "Determine how many <o> are observed at current moment.",
                "Record the quantity of <o> at current moment."
            ],
            "params": [
                "<o>"
            ],
            "end_filter": "count",
            "constraint": []}}
        , frame=world_path, attempts=100, max=4, id_start=0, verbose=verbose, multiview=False)
    for id, record in records.items():
        print(id, record)
