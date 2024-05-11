from vqa.scene_graph import TemporalGraph
from vqa.object_node import TemporalNode, transform_vec, box_overlap, box_trajectories_overlap
from typing import Tuple, Iterable
from vqa.dataset_utils import sample_keypoints
from vqa.object_node import extrapolate_bounding_boxes
import numpy as np

def predict_collision(graph: TemporalGraph) -> Tuple[bool, Iterable[TemporalNode]]:
    collided_objects = set()
    ego = graph.get_ego_node()
    for collision_record in ego.collisions:
        collided_objects.add(collision_record[1])
    return len(collided_objects) > 0, collided_objects
def move_around(original_trajectory, offset, inject_step):
    original_box = original_trajectory[inject_step]
    new_box = [(vertex[0] + offset[0], vertex[1] + offset[1]) for vertex in original_box]
    return new_box


def counterfactual_trajectory(graph, injected_trajectory):
    ego = graph.get_ego_node()
    nodes = [graph.get_node(id) for id in graph.nodes.keys() if id != ego.id]
    for other in nodes:
        if box_trajectories_overlap(injected_trajectory, other.bboxes):
            return False
    return True


from vqa.dataset_utils import generate_stopped_trajectory


def counterfactual_stop(graph, stop_step):
    # ego_trajectory = ego.positions
    ego = graph.get_ego_node()
    nodes = [graph.get_node(id) for id in graph.nodes.keys() if id != ego.id]
    ego_bboxes = ego.bboxes
    # stopped_ego_trajectory = generate_stopped_trajectory(stop_step, ego_trajectory)
    stopped_ego_bboxes = generate_stopped_trajectory(stop_step, ego_bboxes)
    for other in nodes:
        if box_trajectories_overlap(stopped_ego_bboxes, other.bboxes):
            return False
    return True
def try_counterfactual_stop(episode):
    import glob
    frame_files = sorted(glob.glob(episode, recursive=True))
    graph = TemporalGraph(frame_files)
    return counterfactual_stop(graph, 10)


def try_predict_collision(episode):
    import glob
    frame_files = sorted(glob.glob(episode, recursive=True))
    graph = TemporalGraph(frame_files)
    return predict_collision(graph)



def try_counterfactual_trajectory(episode):
    import glob
    frame_files = sorted(glob.glob(episode, recursive=True))
    graph = TemporalGraph(frame_files)
    ego = graph.get_ego_node()
    injection_step = 0

    ego_trajectory = ego.positions[injection_step:]
    ego_bboxes = ego.bboxes
    sampled_ego_trajectory = sample_keypoints(np.array(ego_trajectory), num_points=2, sqrt_std_max=16)
    sampled_ego_trajectory = [(x, y) for x, y in zip(sampled_ego_trajectory[0], sampled_ego_trajectory[1])]
    sampled_ego_bboxes = extrapolate_bounding_boxes(sampled_ego_trajectory,
                                                    np.arctan2(ego.headings[0][1], ego.headings[0][0]),
                                                    ego.bboxes[0])
    injected_ego_bboxes = ego.bboxes[:injection_step] + sampled_ego_bboxes
    for box in injected_ego_bboxes:
        assert len(box) == 4, box
    return counterfactual_trajectory(graph, injected_ego_bboxes)

def locate_crash_timestamp(graph: TemporalGraph):
    ego = graph.get_ego_node()
    time = 1024
    for i, collision_record in enumerate(ego.collisions):
        time = min(time,collision_record[0])
    return time


def try_move_around(episode):
    import glob
    frame_files = sorted(glob.glob(episode, recursive=True))
    graph = TemporalGraph(frame_files)
    ego = graph.get_ego_node()
    first_impact_step = locate_crash_timestamp(graph)
    if first_impact_step != -1:
        std = 4
        offset = np.random.normal(0, std, (2)).tolist()
        print(first_impact_step)
        print(offset)
        new_box_at_impact = move_around(ego.bboxes, offset, first_impact_step)
        for id, node in graph.get_nodes().items():
            if id != ego.id and box_overlap(new_box_at_impact, node.bboxes[first_impact_step]):
                return False
    return True



if __name__ == "__main__":
    EPISODE = "C:/school/Bolei/Merging/MetaVQA/test_collision/0_40_69/**/world*.json"
    #EPISODE = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    # print(try_predict_collision(EPISODE))
    #print(try_counterfactual_stop(EPISODE))
    """count = 0
    while count < 100 and not try_counterfactual_trajectory(episode=EPISODE):
        print(count)
        count += 1"""
    print(try_move_around(episode=EPISODE))


