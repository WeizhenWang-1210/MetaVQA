from agent_node import AgentNode
from typing import Iterable
import numpy as np


def transform(ego:AgentNode,bbox:Iterable)->Iterable:
    def change_bases(x,y):
        relative_x, relative_y = x - ego.pos[0], y - ego.pos[1]
        new_x = ego.heading
        new_y = (-new_x[1], new_x[0])
        x = (relative_x*new_x[0] + relative_y*new_x[1])
        y = (relative_x*new_y[0] + relative_y*new_y[1])
        return x,y
    return [change_bases(*point) for point in bbox]

def distance(node1:AgentNode, node2:AgentNode)->float:
    dx,dy = node1.pos[0]-node2.pos[0], node1.pos[1]-node2.pos[1]
    return np.sqrt(dx**2 + dy**2)

def nodify(scene_dict:dict)->tuple[str,list[AgentNode]]:
    agent_dict = scene_dict['agent']
    agent_id = scene_dict['agent']['id']
    nodes = []
    for info in scene_dict['vehicles']:
        nodes.append(AgentNode(
                                        pos = info["pos"],
                                        color = info["color"],
                                        speed = info["speed"],
                                        heading = info["heading"],
                                        lane = info["lane"],
                                        id = info['id'],
                                        bbox = info['bbox'],
                                        height = info['height'],
                                        road_code=info['road_type'],
                                        type = info['type']
                                        )
                )
    nodes.append(
                AgentNode(
                            pos = agent_dict["pos"],
                            color = agent_dict["color"],
                            speed = agent_dict["speed"],
                            heading = agent_dict["heading"],
                            lane = agent_dict["lane"],
                            id = agent_dict['id'],
                            bbox = agent_dict['bbox'],
                            height = agent_dict['height'],
                            road_code=info['road_type'],
                            type = info['type'])
            )
    return agent_id, nodes

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def find_extremities(ref_heading, bboxes, center):
    recentered_bboxes = [(bbox[0] - center[0], bbox[1]-center[1]) for bbox in bboxes]
    max_dot, min_dot = float("-inf"), float("inf")
    left_bbox,right_bbox = bboxes[0], bboxes[1]
    for bbox in recentered_bboxes:
        dotp = dot(bbox, ref_heading)
        if  dotp > max_dot:
            left_bbox = bbox
            max_dot = dotp
        if dotp < min_dot:
            right_bbox = bbox
            min_dot = dotp
    return left_bbox, right_bbox
    