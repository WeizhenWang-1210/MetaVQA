from typing import Any, Callable, Iterable
from scene_graph import SceneGraph
from agent_node import AgentNode,nodify,transform,extend_bbox
from collections import defaultdict
import json
import numpy as np
import argparse

class SubQuery:
    """
    A subquery is a single functional program. It can be stringed into a query. It has takes a search space and returns all
    AgentNode(from the search space) satisfy some conditions(color, type, position w.r.t. to user defined center).
    """
    def __init__(self, color:Iterable[str] = None, 
                 type:Iterable[str] = None,
                 pos:Iterable[str] = None,
                 next = None,
                 prev = None) -> None:
        self.color = color
        self.type = type
        self.pos = pos
        self.next = next
        self.prev = prev
        self.funcs = None
        self.ans = None
    
    def instantiate(self, egos: Iterable[AgentNode], 
                    ref_heading:tuple):
        color_func = color_wrapper(self.color) if self.color else None
        type_func = type_wrapper(self.type) if self.type else None
        """print(self.prev)
        print(self.type)"""
        if not self.prev:
            pos_func = pos_wrapper(egos, self.pos, ref_heading) if self.pos else None
        else:
            #print(self.prev.ans)
            pos_func = pos_wrapper(self.prev.ans, self.pos, ref_heading) if self.pos else None        
        self.funcs = color_func, type_func, pos_func
    
    def __call__(self, 
                 candidates:Iterable[AgentNode],
                 all_nodes: Iterable[AgentNode]) -> Any:
        #print(candidates)
        ans = candidates
        color_func, type_func,pos_func = self.funcs
        ans = pos_func(all_nodes) if pos_func else ans
        ans = type_func(ans) if type_func else ans
        ans = color_func(ans) if color_func else ans
        self.ans = ans
        return self.ans

class Query:
    """
    A query is a functional implentation of an English question. It can have a single-thread of subqueries(counting/referral),
    or it can have multiple lines of subqueries and aggregate the answers at the end.(In logical questions). 

    self.ans should either be None or bboxes (n, 4, 2), int, True/False

    """
    def __init__(self, heads: Iterable[SubQuery], format: str, end_filter: callable,
                  ref_heading: tuple = None, 
                  candidates:Iterable[AgentNode] = None) -> None:
        self.candidates = candidates
        self.format = format
        self.heads = heads
        self.final = end_filter
        self.ref_heading = ref_heading
        self.ans = None
        self.egos = None

    def set_reference(self, heading: tuple)->None:
        self.ref_heading = heading
    
    def set_searchspace(self, nodes: Iterable[AgentNode])->None:
        self.candidates = nodes

    def set_egos(self, nodes: Iterable[AgentNode]):
        self.egos = nodes
    
    def proceed(self):
        search_spaces = []
        for head in self.heads:
            traverser = head
            search_space = self.candidates
            while traverser:
                if not traverser.funcs:
                    traverser.instantiate(self.egos,self.ref_heading)
                search_space = traverser(search_space, self.candidates)
                traverser = traverser.next
            search_spaces.append(search_space)
        self.ans = self.final(search_spaces)
        return self.ans
    
    def __str__(self) -> str:
        print(self.format,[node.id.split("-")[0] for node in self.ans])
     
def color_wrapper(colors:Iterable[str])->Callable:
    def color(candidates:Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            if candidate.color in colors:
                results.append(candidate)
        return results
    return color

def type_wrapper(types:Iterable[str])->Callable:
    #print(types)
    def type(candidates:Iterable[AgentNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            #print(candidate)
            for t in types:
                #print(candidate.type, t)
                if candidate.type == t  or subclass(candidate.type, t):
                    #print(candidate.id)
                    results.append(candidate)
                    break
        return results
    return type

def pos_wrapper(egos: [AgentNode], spatial_retionships: Iterable[str], ref_heading: tuple = None)->Callable:
    def pos(candidates: Iterable[AgentNode]):
        results = []
        for candidate in candidates:
            for ego in egos:
                if ego.id != candidate.id and ego.compute_relation_string(candidate, ref_heading) in spatial_retionships:
                    results.append(candidate)
        return results
    return pos

def target_color(colors:Iterable[str],candidates:Iterable[AgentNode])->Iterable[AgentNode]:
    results = []
    for candidate in candidates:
        if candidate.color in colors:
            results.append(candidate)
    return results

def target_type(candidates:Iterable[AgentNode], types:Iterable[str])->Iterable[AgentNode]:
    results = []
    for candidate in candidates:
        for type in types:
            if candidate.type == type  or subclass(candidate.type, type):
                results.append(candidate)
                break
    return results

def target_pos(candidates: Iterable[AgentNode], egos: Iterable[AgentNode], ref_heading: tuple, spatial_retionships: Iterable[str])->Iterable[AgentNode]:
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
    """
    A queryanswerer is a "prophet". It has knowledge about a particular scene_graph, and it contains a list of queries
    to be answered. 

    self.ans() returns List[List[AgentNode]|None]
    
    """

    def __init__(self, scene_graph:SceneGraph, queries: list[Query]) -> None:
        self.graph:SceneGraph = scene_graph
        self.ego_id:str = scene_graph.ego_id
        self.queries: list[Query] = queries
        self.log = []

    def ans(self, query: Query = None):
        answers = []
        if not query:
            for query in self.queries:
                query.set_reference(self.graph.get_ego_node().heading)
                query.set_searchspace(self.graph.get_nodes())
                query.set_egos([self.graph.get_ego_node()])
                answers.append(query.proceed())
        else:
            query.set_reference(self.graph.get_ego_node().heading)
            query.set_searchspace(self.graph.get_nodes())
            query.set_egos([self.graph.get_ego_node()])
            return query.proceed()
        return answers
    

    def add_query(self, query: Query):
        self.queries.append(query)

    def display_queries(self):
        for query in self.queries:
            print(query)






def greater(A,B)->bool:
    return A > B
def equal(A,B)->bool:
    return A==B
def less(A,B)->bool:
    return A<B
def count(stuff: Iterable)->int:
    return [len(s) for s in stuff]
    
def locate(stuff: Iterable[AgentNode], origin: AgentNode)->Iterable:
    """
    Return the bbox of all AgentNodes in stuff.
    """
    result = []
    for s in stuff:
        for more_stuff in s:
            result.append(more_stuff.bbox)
            print(transform(origin,more_stuff.bbox))
    return result


def locate_wrapper(origin: AgentNode)->Callable:
    def locate(stuff: Iterable[AgentNode]):
        result = []
        for s in stuff:
            for more_stuff in s:
                ego_bbox = transform(origin,more_stuff.bbox)
                ego_3dbbox = extend_bbox(ego_bbox, more_stuff.height)
                result.append(ego_3dbbox)
        return result
    return locate



def CountGreater(search_spaces)->bool:
    """
    Return True if the first set in the search_spaces has greater length than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    return greater(nums[0],nums[1])

def CountEqual(search_spaces)->bool:
    """
    Return True if all sets in search_spaces have the same length.
    """
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    first = nums[0]
    for num in nums:
        if num != first:
            return False
    return True

def CountLess(search_spaces)->bool:
    """
    Return True if the first set in the search_spaces has greater smaller than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    return less(nums[0],nums[1])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default = "520/520_60/world_520_60")
    args = parser.parse_args()
    try:
        print(args.step)
        with open('{}.json'.format(args.step),'r') as scene_file:
                scene_dict = json.load(scene_file)
    except:
        print("Wrong")
    agent_id,nodes = nodify(scene_dict)
    graph = SceneGraph(agent_id,nodes)     
    q1 = SubQuery(None,["Sportscar"], ['rf'], None, None)
    q3 = SubQuery(None, ["Compact Sedan"], ['b'], None, q1)
    q1.next = q3
    q2 = SubQuery(None,["Compact Sedan"],None,None,None)
    q = Query([q1,q2],"counting",CountEqual)
    prophet = QueryAnswerer(graph,[q]) 
    result = prophet.ans(q)


            