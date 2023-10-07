from typing import Any, Callable, Iterable
from scene_graph import SceneGraph
from agent_node import AgentNode,transform, distance, nodify
from collections import defaultdict
import json
import random
import numpy as np
import argparse
import glob

class FilterConstructor:
    def __init__(self, type, plan: dict) -> None:
        self.plan = plan
        self.paths = None
    

    def readplan(self, plan):
        if not plan:
            plan = self.plan
        if plan['format'] == "logical":
            for path in plan["paths"]:
                subplan = (path["color"], path["type"], path["pos"])

class SubQuery:
    def __init__(self, color, type, pos, next, prev) -> None:
        self.color = color
        self.type = type
        self.pos = pos
        self.next = next
        self.prev = prev
        self.funcs = None
        self.ans = None
    
    def instantiate(self, ref_heading):
        color_func = color_wrapper(self.color) if self.color else None
        type_func = type_wrapper(self) if self.type else None
        pos_func = pos_wrapper(self.prev.ans, self.pos, ref_heading) if self.pos else None
        self.funcs = color_func, type_func, pos_func
    
    def __call__(self, candidates) -> Any:
        ans = candidates
        for func in self.funcs:
            if func:
                ans = func(ans)
        self.ans = ans
        return self.ans

class Query:
    def __init__(self, heads: Iterable[SubQuery], format, end_filter, ref_heading = None, candidates = None) -> None:
        self.candidates = candidates
        self.format = format
        self.heads = heads
        self.final = end_filter
        self.ref_heading = ref_heading
        self.ans = None

    def set_reference(self, heading):
        self.ref_heading = heading
    
    def set_searchspace(self, nodes):
        self.candidates = nodes
    
    def proceed(self):
        search_spaces = []
        for head in self.heads:
            traverser = head
            search_space = self.candidates
            while traverser:
                if not traverser.funcs:
                    traverser.instantiate(self.ref_heading)
                search_space = traverser(search_space)
                traverser = traverser.next
            search_spaces.append(search_space)
        if len(search_spaces) == 1:
            search_spaces = search_spaces[0]
        self.ans = self.final(search_spaces)
        return self.ans
        

        
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

def pos_wrapper(egos: [AgentNode], spatial_retionships: Iterable[str], ref_heading: tuple = None):
    def pos(candidates: Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            for ego in egos:
                if ego.id != candidate.id and ego.compute_relation_string(candidate, ref_heading) in spatial_retionships:
                    results.append((ego,candidate))
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

def target_pos(candidates: Iterable[AgentNode], egos: Iterable[AgentNode], ref_heading: tuple, spatial_retionships: Iterable[str]):
    results = []
    for candidate in candidates:
        for ego in egos:
            if ego.compute_relation(candidate, ref_heading) in spatial_retionships:
                results.append((ego,candidate))
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
        





class QueryAnswerer:
    def __init__(self, scene_graph:SceneGraph, queries: Iterable[Query]) -> None:
        self.graph:SceneGraph = scene_graph
        self.ego_id:str = scene_graph.ego_id
        self.queries: Iterable[Query] = queries
        self.log = None



    def ans(self, query: Query = None):
        answers = []
        if not query:
            for query in self.queries:
                query.set_reference(self.graph.get_ego_node().heading)
                query.set_searchspace(self.graph.get_nodes())
                answers.append(query.proceed())
        else:
            query.set_reference(self.graph.get_ego_node().heading)
            answers.append(query.proceed())
        return answers


    

    def filter(self, candidates: Iterable[str], filter_constructor)->Iterable[str]:
        result = candidates
        filter = filter_constructor.generate()
        while filter:
            result = filter(result)
            filter = filter_constructor.generate()
        return result
    
    def counting(self, candidates: Iterable[str] = None, filter = list[Callable]):
        if not candidates:
            candidates = self.graph.get_nodes()
        return len(self.filter(candidates,filter))
    
    def locating(self, candidates: Iterable[str] = None, filter = list[Callable]):
        ans = {}
        if not candidates:
            candidates = self.graph.get_nodes()
        agents = self.filter(candidates, filter)
        for agent in  agents:
            ans[agent.id] = agent.bbox
        return ans
        



class QuestionGenerator:
    def __init__(self, query_lists: list[dict], queryanswerer: QueryAnswerer):
        self.prophet: QueryAnswerer = queryanswerer
        self.paths: list[dict] = query_lists
        self.queries: Iterable[Iterable[Query]] = self.create_queries(self.prophet.graph.get_ego_node())
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
            query = Query(partial, path["format"])
            queries.append(query)
        return queries
    
    def answer(self)->list:
        answers = []
        for query in self.queries:
            if query.type == "counting":
                answers.append(self.prophet.counting(filter = query.filters))
            elif query.type == "locating":
                answers.append(self.prophet.locating(filter = query.filters))
            elif query.type == "logical":
                answer = None
        return answers








def greater(A,B):
    return A > B
def equal(A,B):
    return A==B
def less(A,B):
    return A<B

def count(stuff: Iterable):
    return len(count)
    







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


    q1 = SubQuery(None, ["Vehicle"], ["lf"])
    q2 = SubQuery(None, ["Compact Sedan"], ["rf"], next = None, prev = q1)
    q = Query([q1],"counting",count)

    result = prophet.ans(q)
    

    print(result)
    
    
    
    
    #test_generator = QuestionGenerator(query_list, prophet)
    #print(graph.spatial_graph[graph.ego_id])
    
   
            