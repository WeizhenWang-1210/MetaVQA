from typing import Callable, Iterable
from scene_graph import SceneGraph
from agent_node import AgentNode
from collections import defaultdict
import json
import random
import numpy as np
import argparse
import glob
class QueryAnswerer:
    def __init__(self, scene_graph:SceneGraph) -> None:
        self.graph:SceneGraph = scene_graph
        self.ego_id:str = scene_graph.ego_id

    def filter(self, candidates: Iterable[str], filters:list[Callable])->Iterable[str]:
        result = candidates
        for filter in filters:
            result = filter(result)
        return result
    
    def counting(self, candidates: Iterable[str] = None, filter = list[Callable]):
        if not candidates:
            candidates = self.graph.get_nodes()
        return len(self.filter(candidates,filter))
        
        
def color_wrapper(colors:Iterable[str]):
    def color(candidates:Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            if candidate.color in colors:
                results.append(candidate)
        return results
    return color

def type_wrapper(types:Iterable[str]):
    #print(types)
    def type(candidates:Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            #print(candidate)
            for t in types:
                if candidate.type == t  or subclass(candidate.type, t):
                    results.append(candidate)
                    break
        return results
    return type

def pos_wrapper(ego: AgentNode, spatial_retionships: Iterable[str]):
    def pos(candidates: Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            #print(ego.id, candidate.id, ego.compute_relation(candidate,ego.heading))
            if ego.id != candidate.id and ego.compute_relation_string(candidate, ego.heading) in spatial_retionships:
                results.append(candidate)
        return results
    return pos

def target_color(colors:Iterable[str],candidates:Iterable[AgentNode]):
    results = []
    for candidate in candidates:
        if candidate.color in colors:
            results.append(candidate)
    return results

def target_type(candidates:Iterable[AgentNode], types:Iterable[str]):
    results = []
    for candidate in candidates:
        for type in types:
            if candidate.type == type  or subclass(candidate.type, type):
                results.append(candidate)
                break
    return results

def target_pos(candidates: Iterable[AgentNode], ego: AgentNode, spatial_retionships: Iterable[str]):
    results = []
    for candidate in candidates:
        if ego.compute_relation(candidate, ego.heading) in spatial_retionships:
            results.append(candidate)
    return results

def subclass(class1:str, class2:str)->bool:
    inheritance = get_inheritance()
    if class1 == class2:
        return True
    result = False
    for child in inheritance[class2]:
        result = result or subclass(class1, child)
    return result

def get_inheritance()->defaultdict:
    inheritance = defaultdict(lambda:[])
    inheritance["Vehicle"] = ["SUV", "Sedan", "Truck", "Sportscar","Jeep","Pickup","Compact Sedan"]
    inheritance["Traffic Obstacle"] = ["Traffic Cone", "Warning sign", "Planar Barrier"]
    inheritance["Traffic Participant"] = ["Vehicle", "Pedestrian"]
    return inheritance


"""dict(
    color = "white",
    type = "car",
    ...
)"""

class QuestionGenerator:
    def __init__(self, query_lists: list[dict], queryanswerer: QueryAnswerer):
        self.prophet: QueryAnswerer = queryanswerer
        self.paths: list[dict] = query_lists
        self.queries: Iterable[Iterable[Callable]] = self.create_queries(self.prophet.graph.get_ego_node())
        self.answers: Iterable = self.answer()

    def create_queries(self,ego) -> list[list[Callable]]:
        queries = []
        for path in self.paths:
            partial = []
            if path["color"]:
                partial.append(color_wrapper(path["color"]))
            if path["type"]:
                partial.append(type_wrapper(path["type"]))
            if path["pos"]:
                partial.append(pos_wrapper(ego,path["pos"]))
            queries.append(partial)
        return queries
    
    def answer(self)->list:
        answers = []
        for query in self.queries:
            answers.append(self.prophet.counting(filter = query))
        return answers

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default = None)
    args = parser.parse_args()
    try:
        print(args.step)
        with open('{}.json'.format(args.step),'r') as scene_file:
                scene_dict = json.load(scene_file)
    except:
        print("Wrong")
    agent_id,nodes = nodify(scene_dict)
    
    graph = SceneGraph(agent_id,nodes)
    prophet = QueryAnswerer(graph)
    query_list = [
        dict(color = None,
             type  = None,
             pos = ["rf"])
    ]
    test_generator = QuestionGenerator(query_list, prophet)
    #print(graph.spatial_graph[graph.ego_id])
    print(test_generator.answers)
   
            