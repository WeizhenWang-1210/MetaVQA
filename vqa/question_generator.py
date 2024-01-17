from typing import Any, Callable, Iterable, List
from vqa.scene_graph import SceneGraph
from vqa.object_node import ObjectNode,nodify,transform
from vqa.dataset_utils import extend_bbox
from collections import defaultdict
from vqa.dynamic_filter import follow, pass_by, collide_with, head_toward, drive_alongside
import json
import numpy as np
import argparse
import os
pwd = os.getcwd()
template_path = os.path.join(pwd, "vqa/question_templates.json")
with open(template_path,"r") as f:
    templates = json.load(f)



class SubQuery:
    """
    A subquery is a single functional program. It can be stringed into a query graph(abstracted in Query). It takes a search space and returns all
    ObjectNode(from the search space) satisfy some conditions(color, type, position w.r.t. to user defined center).
    """
    def __init__(self, color:Iterable[str] = None, 
                 type:Iterable[str] = None,
                 pos: Iterable[str] = None,
                 state: Iterable[str] = None,
                 action: Iterable[str] = None,
                 next = None,
                 prev = None) -> None:
        '''
        Initializer
        '''
        self.us = False
        self.state = state #The state we are looking for
        self.action = action #The action we are looking for, with or without respect to some objects
        self.color = color #The color we are looking for
        self.type = type   #The type we are looking for
        self.pos = pos     #The spatial relationship we are looking for
        self.next = next   #The previous subquery. It's used to retrieve the search space for the candidate
        self.prev = prev   #The next subquery
        self.funcs = None  #The actual functions used to do filtering.
        self.ans = None    #recording the answer in previous call
    
    def instantiate(self, egos: Iterable[ObjectNode],
                    ref_heading:tuple):
        '''
        Initialize the functions for filtering
        '''
        if self.us:
            #If the query is "us", just return the ego node. No actual search needs to be performed.
            self.funcs = [lambda x: egos]
        else: 
            #If this is, indeed, a valid subquery, then create the proper functions to narrow down the search space
            color_func = color_wrapper(self.color) if self.color else None
            type_func = type_wrapper(self.type) if self.type else None
            state_func = state_wrapper(self.state) if self.state else None
            pos_func = pos_wrapper(self.prev["pos"].ans, self.pos, ref_heading) if self.pos else None
            action_func = action_wrapper(self.prev["action"].ans, self.action) if self.action else None
            self.funcs = [color_func, type_func, pos_func, state_func, action_func]
        
    def __call__(self, 
                 candidates:Iterable[ObjectNode],
                 all_nodes: Iterable[ObjectNode]) -> Any:
        """
        candidates: the search space
        all_nodes: the node space
        Use subQuery as an actual function.
        """
        ans = candidates
        if self.us:
            self.ans = self.funcs[0](ans)
            return self.ans
        """
        The tricky part here is that when we keep the entire node space in mind when we do the filtering based on the spatial relationships,
        and consecutive filterings on types and colors are selected from all such nodes.
        """
        color_func, type_func,pos_func,state_func, action_func = self.funcs
        ans = pos_func(all_nodes) if pos_func else ans
        ans = type_func(ans) if type_func else ans
        ans = color_func(ans) if color_func else ans
        ans = state_func(ans) if state_func else ans
        ans = action_func(ans) if action_func else ans
        self.ans = ans
        return self.ans

class Query:
    """
    A query is a functional implentation of an English question. It can have a single-thread of subqueries(counting/referral),
    (in fact, better yet, we can abstrac the aggregat method of single-thread as the "identity" method: extracting the one-and-only answer)
    or it can have multiple lines of subqueries and aggregate the answers at the end.(In logical questions). 
    self.ans should either be None or bboxes (n, 4, 2), int, True/False
    """
    def __init__(self, heads: Iterable[SubQuery], format: str, end_filter: callable,
                  ref_heading: tuple = None, 
                  candidates:Iterable[ObjectNode] = None,
                  ) -> None:
        '''
        heads: the beginning subqueries of our query. Guaranteed linked-list-like structure. Reach the end and aggregate the result with end_filter
        format: a str for specifying what kind of question this is
        end_filter: a function: Iterable[Iterable] -> Iterable[]. Used to aggregate answers
        ref_heading: a vector that indicates the direction of positive x-coordinate, in world coordinate
        candidates: the search space of the Query
        '''
        self.candidates = candidates
        self.format = format
        self.heads = heads
        self.final = end_filter
        self.ref_heading = ref_heading
        self.ans = None
        self.egos = None

    def set_reference(self, heading: tuple)->None:
        '''
        set reference heading
        '''
        self.ref_heading = heading
    
    def set_searchspace(self, nodes: Iterable[ObjectNode])->None:
        '''
        set the allowed searchspace
        '''
        self.candidates = nodes

    def set_egos(self, nodes: Iterable[ObjectNode]):
        '''
        set the center vehicles
        '''
        self.egos = nodes
    
    def proceed(self):
        '''
        Go down the query, store partial answers in sub-query, store the final answer in query
        
        '''

        def postorder_traversal(subquery,egos, ref_heading, all_nodes):
            """
            Return: A list of nodes being the answer of the previous question.
            """
            
            if subquery.prev:
                for child_subquery in subquery.prev.values():
                    postorder_traversal(child_subquery, egos, ref_heading, all_nodes)
            if not subquery.funcs:
                subquery.instantiate(egos, ref_heading)
            result = all_nodes
            result = subquery(result,all_nodes)
        
        search_spaces = []
        for root in self.heads:
            postorder_traversal(root,self.egos,self.ref_heading,self.candidates)
            search_spaces.append(root.ans)
        self.ans = self.final(search_spaces)

        """ search_spaces = []
        for head in self.heads:
            traverser = head
            search_space = self.candidates
            while traverser:
                if not traverser.funcs:
                    traverser.instantiate(self.egos,self.ref_heading)
                search_space = traverser(search_space, self.candidates)
                traverser = traverser.next
            search_spaces.append(search_space)
        self.ans = self.final(search_spaces)"""
        return self.ans
    
    def __str__(self) -> str:
        '''
        get all answers
        '''
        print(self.format,[node.id.split("-")[0] for node in self.ans])
     
"""
Only return when candidates are visible by ego
"""


def color_wrapper(colors:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with color in colors
    '''
    def color(candidates:Iterable[ObjectNode]):
        results = []
        for candidate in candidates:
            if candidate.visible and candidate.color in colors:
                results.append(candidate)
        return results
    return color

def type_wrapper(types:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with type in types or is a subtype of type in types
    '''
    #print(types)
    def type(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            #print(candidate)
            for t in types:
                #print(candidate.type, t)
                if candidate.type == t  or subclass(candidate.type, t):
                    #print(candidate.id)
                    results.append(candidate)
                    break
        return results
    return type

def state_wrapper(states:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with one state in states
    '''
    #print(types)
    def state(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            #print(candidate)
            for s in states:
                #print(candidate.type, t)
                if s in candidate.state:
                    #print(candidate.id)
                    results.append(candidate)
                    break
        return results
    return state

def action_wrapper(egos: [ObjectNode], actions: Iterable[str], ref_heading: tuple = None)->Callable:
    def act(candidates: Iterable[ObjectNode]):
        mapping = {
            "follow": follow,
            "pass by": pass_by,
            "collide with":collide_with,
            "head toward": head_toward,
            "drive alongside": drive_alongside
        }
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            for ego in egos:
               for action in actions:
                   if ego.id != candidate.id and mapping[action](ego, candidate):
                       results.append(candidate)
        return results
    return act

def pos_wrapper(egos: [ObjectNode], spatial_retionships: Iterable[str], ref_heading: tuple = None)->Callable:
    '''
    A constructor for selecting all nodes that exhibit spatial_relationship with any ego in egos for spatial_relationship in spatial_relationships.
    ref_heading is provided to define what's left v.s. right
    '''
    def pos(candidates: Iterable[ObjectNode]):
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            for ego in egos:
                if ego.id != candidate.id and ego.compute_relation_string(candidate, ref_heading) in spatial_retionships:
                    results.append(candidate)
        return results
    return pos


def subclass(class1:str, class2:str)->bool:
    '''
    determine if class1 is the subclass of class2
    '''
    inheritance = get_inheritance() #inheritance is not a tree. But, it's a DAG from supertype to subtype(like your typing system in C++)
    if class1 == class2:
        return True
    result = False
    for child in inheritance[class2]:
        result = result or subclass(class1, child)
    return result

def get_inheritance()->defaultdict:
    '''
    Return a lineage tree as a dictionary
    '''
    import yaml
    with open("./asset_config.yaml","r") as stream:
        tree = yaml.safe_load(stream)["type"]

    inheritance = defaultdict(lambda:[])

    def get_non_leaf_nodes(d, inheritance, parent_key='', ):
        non_leaf_nodes = []
        for key, value in d.items():
            # Construct a full key path if you are in a nested dictionary
            full_key = parent_key + '.' + key if parent_key else key
            if isinstance(value, dict):
                inheritance[parent_key].append(key)
                non_leaf_nodes.append(full_key)
                # Recursively search for non-leaf nodes
                non_leaf_nodes.extend(get_non_leaf_nodes(value, inheritance, key))
        return non_leaf_nodes
    get_non_leaf_nodes(tree, inheritance)
    return inheritance
        
class QueryAnswerer:
    """
    A queryanswerer is a "prophet". It has knowledge about a particular scene_graph, and it contains a list of queries
    to be answered. 

    self.ans() returns List[List[AgentNode]|None]
    """
    def __init__(self, 
                 scene_graph:SceneGraph, 
                 queries: List[Query],
                ) -> None:
        self.graph:SceneGraph = scene_graph
        self.ego_id:str = scene_graph.ego_id
        self.queries: list[Query] = queries
        self.log = []

    def ans(self, query: Query = None):
        '''
        Generate answer
        '''
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
        '''
        add query to the prophet
        '''
        self.queries.append(query)

    def display_queries(self):
        '''
        list all queries
        '''

        for query in self.queries:
            print(query)



def greater(A,B)->bool:
    '''
    checker
    '''
    return A > B
def equal(A,B)->bool:
    '''
    checker
    '''
    return A==B
def less(A,B)->bool:
    '''
    checker
    '''
    return A<B
def count(stuff: Iterable)->int:
    '''
    checker
    '''
    return [len(s) for s in stuff]
    
'''
End filters
'''

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

def Describe(search_spaces)->str:
    """
    Return True if the first set in the search_spaces has greater smaller than the second set.
    """
    search_spaces = search_spaces[0]
    if len(search_spaces) == 0:
        return "No, there is not any item with specified action"
    result = "Yes, there is "
    result += search_spaces[0].color
    result += " "
    result += search_spaces[0].type
    if len(search_spaces) == 1:
        return result
    else:
        for node in search_spaces[1:]:
            result += " and "
            result += node.color
            result += " "
            result += node.type
        result += '.'
    return result

def Identity(search_spaces):
    '''
    Return the singleton answer in search spaces
    '''
    return search_spaces[0]

def locate(stuff: Iterable[ObjectNode], origin: ObjectNode)->Iterable:
    """
    Return the bbox of all AgentNodes in stuff.
    """
    result = []
    for s in stuff:
        for more_stuff in s:
            result.append(more_stuff.bbox)
            print(transform(origin,more_stuff.bbox))
    return result

def locate_wrapper(origin: ObjectNode)->Callable:
    """
    The returned function takes in an Iterable of ObjectNode and returns the 3d bounding boxes in ego's coordinate
    """
    def locate(stuff: Iterable[ObjectNode]):
        result = []
        for s in stuff:
            for more_stuff in s:
                ego_bbox = transform(origin,more_stuff.bbox)
                ego_3dbbox = extend_bbox(ego_bbox, more_stuff.height)
                result.append(ego_3dbbox)
        return result
    return locate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default = "verification/10_40/world_10_40")
    args = parser.parse_args()
    try:
        print(args.step)
        with open('{}.json'.format(args.step),'r') as scene_file:
                scene_dict = json.load(scene_file)
    except Exception as e:
        raise e
    agent_id,nodes = nodify(scene_dict)
    graph = SceneGraph(agent_id,nodes)    

    q1 = SubQuery(
        color= None,
        type = ["Bike"], 
        pos= ['rb'], 
        state= None, 
        action= None,
        next = None,
        prev = {})
    q2 = SubQuery(
        color= None,
        type = ["Caravan"], 
        pos= None, 
        state= None, 
        action= None,
        next = None,
        prev = {})
    
    q3 = SubQuery()
    q3.us = True
    q2.prev["pos"] = q3
    q1.prev["pos"] = q2

    #q3 = SubQuery(None, ["Truck"], ['r'], None, q1)
    #q1.next = q3
    #q2 = SubQuery(None,None,None,None,None)
    q = Query([q1],"counting",Identity)
    prophet = QueryAnswerer(graph,[q])
    result = prophet.ans(q)
    print(result[0].id)
    
    ids = [node.id for node in q.ans if node.id != agent_id]
    print(len(ids))
    from vqa.visualization import generate_highlighted
    generate_highlighted(path_to_mask =  "verification/10_40/mask_10_40.png",
                         path_to_mapping= "verification/10_40/metainformation_10_40.json",
                         folder = "verification/10_40",
                         ids = ids,
                         colors = [(1,1,1)]*len(ids))
    print(result)

            