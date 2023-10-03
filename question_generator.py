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
    def type(candidates:Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            for type in types:
                if candidate.type == type  or subclass(candidate.type, type):
                    results.append(candidate)
                    break
        return results
    return type

def pos_wrapper(ego: AgentNode, spatial_retionships: Iterable[str]):
    def pos(candidates: Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            if ego.compute_relation(candidate, ego.heading) in spatial_retionships:
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
    inheritance["vehicle"] = ["SUV", "Sedan", "Truck", "Sportscar"]
    inheritance["traffic obstacle"] = ["Traffic cone", "Warning sign"]
    inheritance["traffic participant"] = ["vehicle", "pedestrian"]
    return inheritance


dict(
    color = "white",
    type = "car",
    ...
)

class QuestionGenerator:
    def __init__(self, query_lists: list[dict], queryanswerer: QueryAnswerer):
        self.prophet: QueryAnswerer = queryanswerer
        self.paths: list[dict] = query_lists
        self.queries: Iterable[Iterable[Callable]] = self.create_queries()
        self.answers: Iterable = self.answer()

    def create_queries(self) -> list[list[Callable]]:
        queries = []
        for path in self.paths:
            partial = []
            if path["color"]:
                partial.append(color_wrapper(path["color"]))
            if path["type"]:
                partial.append(type_wrapper(path["type"]))
            if path["pos"]:
                partial.append(pos_wrapper(path["pos"]))
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
                            road_code=info['road_type'])
            )
    return agent_id, nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = bool, default= False)
    parser.add_argument("--step", type=str, default = None)
    parser.add_argument("--folder", type=str, default = None)
    args = parser.parse_args()
    if args.batch == True:
        assert args.folder is not None
        gts = glob.glob(args.folder+"/[0-9]*_[0-9]*/world*",recursive=True)
        for gt in gts:
            splitted = gt.split("\\")
            root = "/".join(splitted[:2])
            print(root)
            try:
                with open(gt,'r') as scene_file:
                    scene_dict = json.load(scene_file)
            except:
                print("Error in reading json file {}".format("gt"))
            if len(scene_dict["vehicles"]) == 0:
                continue
            agent_id, nodes = nodify(scene_dict)
            graph = scene_graph(agent_id,nodes)
            test_generator = Question_Generator(graph)
            datapoints = test_generator.generate_all()
            statistics =test_generator.get_stats()
            scene_data = {}
            for idx, datapoint in enumerate(datapoints):
                if len(datapoint)==2:
                    text, candidate = datapoint
                    qa_dict = {
                    "text":text,
                    "bbox":transform(graph.nodes[agent_id], graph.nodes[candidate].bbox),
                    "height":graph.nodes[candidate].height,
                    "id":candidate,
                    "ref":""
                }
                else:
                    text, candidate, compared = datapoint
                    qa_dict = {
                        "text":text,
                        "bbox":transform(graph.nodes[agent_id], graph.nodes[candidate].bbox),
                        "height":graph.nodes[candidate].height,
                        "id":candidate,
                        'ref':compared
                    }
                scene_data[idx] = qa_dict
            try:
                with open(root + '/qa_{}.json'.format(splitted[1]),'w') as file:
                    json.dump(scene_data,file)
            except:
                print("wtf")
            try:
                with open(root +"/stats_{}.json".format(splitted[1]),'w') as file:
                    json.dump(statistics, file)
            except:
                print("Error recording statistics")