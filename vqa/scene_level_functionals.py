from vqa.scene_graph import TemporalGraph
from vqa.object_node import TemporalNode, transform, overlap
from typing import Tuple, Iterable


def predict_collision(graph: TemporalGraph) -> Tuple[bool, Iterable[TemporalNode]]:
    collided_objects = set()
    ego = graph.get_ego_node()
    for collision_record in ego.collisions:
        collided_objects.add(collision_record[1])
    return len(collided_objects) > 0, collided_objects

def move_around(ego: TemporalNode, other: TemporalNode, collision_step, offset):
    origin = ego.positions[0]
    positive_x = ego.headings[0]
    t0_bbox = transform(origin, positive_x, ego.bboxes[collision_step])
    offseted_bbox = [[point[0]+offset[0], point[1]+offset[1]] for point in t0_bbox]
    t0_other_bbox = transform(origin, positive_x, other.bboxes[collision_step])
    return overlap(offseted_bbox, t0_other_bbox)



def counterfactual_trajectory(ego, other, injected_trajectory, injection_step):
    pass




def counterfactual_stop(ego, other, stop_step):
    pass


def counterfactual():
    pass






def bboxified_trajectory(center_trajectory, bbox):
    pass





def supersample_trajectory():
    """
    This is needed so that we have better granularity.

    """


def try_predict_collision(episode):
    import glob
    frame_files = sorted(glob.glob(episode, recursive=True))
    graph = TemporalGraph(frame_files)
    return predict_collision(graph)



if __name__ == "__main__":
    #EPISODE = "C:/school/Bolei/Merging/MetaVQA/test_collision/0_40_69/**/world*.json"
    EPISODE = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    print(try_predict_collision(EPISODE))




