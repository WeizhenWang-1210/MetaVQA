from typing import Any, Callable, Iterable, List
from vqa.scene_graph import SceneGraph
from vqa.object_node import ObjectNode, nodify, transform
from vqa.dataset_utils import extend_bbox
from vqa.functionals import subclass
from collections import defaultdict
import json
import numpy as np
import argparse
class TemporalTracker:
    def __init__(self, change_term, track_feature_list: List[str] = ["visible"]):
        self.track_feature_list = track_feature_list
        self.obj_feature_dict = dict()
        self.change_dict = defaultdict(dict)
        self.change_term = change_term
    def change_trigger(self, node_id, feature, old_value, new_value):
        print("{} change trigger for node id {}, from {} to {}".format(feature, node_id, old_value, new_value))
        feature_change_case = self.change_term[feature]
        self.change_dict[feature][node_id] = feature_change_case[(old_value, new_value)]
    def get_change_node(self):
        return self.change_dict
    def update_node(self, node: ObjectNode):
        if node.id not in self.obj_feature_dict:
            self.obj_feature_dict[node.id] = dict()
            for feature in self.track_feature_list:
                self.obj_feature_dict[node.id][feature] = getattr(node, feature)
        else:
            for feature in self.track_feature_list:
                if self.obj_feature_dict[node.id][feature] != getattr(node, feature):
                    self.change_trigger(feature=feature,
                                        node_id=node.id,
                                        old_value=self.obj_feature_dict[node.id][feature],
                                        new_value=getattr(node, feature))
                    self.obj_feature_dict[node.id][feature] = getattr(node, feature)

    def update(self, graph: SceneGraph):
        self.change_dict = defaultdict(dict)
        for node_id in graph.nodes.keys():
            self.update_node(graph.nodes[node_id])

class TempSubQuery:
    """
    A subquery is a single functional program. It can be stringed into a query graph(abstracted in Query). It takes a search space and returns all
    ObjectNode(from the search space) satisfy some conditions(color, type, position w.r.t. to user defined center).
    """

    def __init__(self,
                 obj: str,
                 action: str) -> None:
        '''
        Initializer
        '''
        self.obj = obj
        self.action = action
        self.funcs = None  # The actual functions used to do filtering.
        self.ans = None  # recording the answer in previous call
        # TODO: change this to config
        self.mapping_action_to_feature = {"passed by": "visible", "showed up": "visible"}
        self.next = None

    def instantiate(self, TemporalTracker):
        '''
        Initialize the functions for filtering
        '''
        obj_func = obj_wrapper(self.obj)
        change_dict = TemporalTracker.get_change_node()[self.mapping_action_to_feature[self.action]]
        action_func = action_wrapper(change_dict=change_dict, action=self.action)
        self.funcs = obj_func, action_func

    def __call__(self,
                 candidates: Iterable[ObjectNode],
                 all_nodes: Iterable[ObjectNode]) -> Any:
        """
        candidates: the search space
        all_nodes: the node space
        Use subQuery as an actual function.
        """
        ans = candidates
        obj_func, action_func = self.funcs
        """
        The tricky part here is that when we keep the entire node space in mind when we do the filtering based on the spatial relationships,
        and consecutive filterings on types and colors are selected from all such nodes.
        """
        ans = obj_func(all_nodes) if obj_func else ans
        ans = action_func(ans) if action_func else ans
        self.ans = ans

        return self.ans


class TempQuery:
    """
    A query is a functional implentation of an English question. It can have a single-thread of subqueries(counting/referral),
    (in fact, better yet, we can abstrac the aggregat method of single-thread as the "identity" method: extracting the one-and-only answer)
    or it can have multiple lines of subqueries and aggregate the answers at the end.(In logical questions).
    self.ans should either be None or bboxes (n, 4, 2), int, True/False
    """

    def __init__(self, heads: Iterable[TempSubQuery], format: str, end_filter: callable,
                 ref_heading: tuple = None,
                 candidates: Iterable[ObjectNode] = None,
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

    def set_reference(self, heading: tuple) -> None:
        '''
        set reference heading
        '''
        self.ref_heading = heading

    def set_searchspace(self, nodes: Iterable[ObjectNode]) -> None:
        '''
        set the allowed searchspace
        '''
        self.candidates = nodes

    def set_egos(self, nodes: Iterable[ObjectNode]):
        '''
        set the center vehicles
        '''
        self.egos = nodes

    def proceed(self, tempTracker: TemporalTracker):
        '''
        Go down the query, store partial answers in sub-query, store the final answer in query

        '''
        search_spaces = []
        for head in self.heads:
            traverser = head
            search_space = self.candidates
            while traverser:
                if not traverser.funcs:
                    traverser.instantiate(tempTracker)
                search_space = traverser(search_space, self.candidates)
                traverser = traverser.next
            search_spaces.append(search_space)
        self.ans = self.final(search_spaces)
        return self.ans

    def __str__(self) -> str:
        '''
        get all answers
        '''
        print(self.format, [node.id.split("-")[0] for node in self.ans])
class TempQueryAnswerer:
    """
    A queryanswerer is a "prophet". It has knowledge about a particular scene_graph, and it contains a list of queries
    to be answered.

    self.ans() returns List[List[AgentNode]|None]
    """
    def __init__(self,
                 scene_graph:SceneGraph,
                 queries: List[TempQuery],
                ) -> None:
        self.graph:SceneGraph = scene_graph
        self.ego_id:str = scene_graph.ego_id
        self.queries: list[TempQuery] = queries
        self.log = []

    def ans(self, tempTracker: TemporalTracker, query: TempQuery = None):
        '''
        Generate answer
        '''
        answers = []
        if not query:
            for query in self.queries:
                query.set_reference(self.graph.get_ego_node().heading)
                query.set_searchspace(self.graph.get_nodes())
                query.set_egos([self.graph.get_ego_node()])
                answers.append(query.proceed(tempTracker = tempTracker))
        else:
            query.set_reference(self.graph.get_ego_node().heading)
            query.set_searchspace(self.graph.get_nodes())
            query.set_egos([self.graph.get_ego_node()])
            return query.proceed(tempTracker = tempTracker)
        return answers

    def add_query(self, query: TempQuery):
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
def obj_wrapper(obj: str)->Callable:
    '''
    Constructor for a function that return all nodes with type in types or is a subtype of type in types
    '''
    #print(types)
    obj_names = []
    # TODO: Make this configurable
    if obj == "car":
        obj_names = ["SUV","Sedan", "Compact", "Compact Sedan", "Truck", "Bike", "Motorcycle", "SportCar", "Jeep", "Pickup"]
    def type(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            #print(candidate)
            existFlag = False
            for name in obj_names:
                #print(candidate.type, t)
                if candidate.type == name:
                    results.append(candidate)
                    existFlag = True
                    break
            if not existFlag:
                print("No, the type {} does not exist".format(candidate.type))
        return results
    return type
def action_wrapper(change_dict: dict, action)->Callable:
    '''
    Constructor for a function that return all nodes with type in types or is a subtype of type in types
    '''
    def type(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            #print(candidate)
            if candidate.id in change_dict.keys():
                if change_dict[candidate.id] == action:
                    results.append(candidate)
        return results
    return type