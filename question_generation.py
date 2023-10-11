import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
from collections import defaultdict
OPENAI_KEY = "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"
from question_generator import Query,SubQuery,count, QueryAnswerer,CountGreater, CountEqual, CountLess
from scene_graph import SceneGraph
from agent_node import nodify
#don't work when the scene is crowded.

"""
Note that MetaDrive uses a right-handed coordinate system with x being the heading direction
           
                
                | +x    
                |
    +y          |     (top-down view, +z coming out of the screen)
   ----------------------------------

"""



"""
User -> QuestionSpecifier -> Query         ------- Answer
                |             QueryAnswerer -------|
                |
            English Questioin

"""


class QuestionSpecifier:
    def __init__(self,
                 specification,
                 graph: SceneGraph) -> None:
        self.type = specification
        self.graph = graph


    def translate_to_En(self):
        format = self.type["format"]
        if format == "count":
            return counting_translator()(QtoEn(self.type["paths"]))
        elif format == "logical":
            translator = decode_predicate(self.type["end"])
            return translator(QtoEn(self.type["paths"]))
        return ""
        

    def translate_to_Q(self):
        def hookup(subqueries):
            prev = None
            for subquery in subqueries:
                if prev:
                    prev.next = subquery
                subquery.prev = prev
                prev = subquery

        def select_end_filter(type):
            if type == "count":
                return count
            elif type == "Count Greater":
                return CountGreater
            elif type == "Count Equal":
                return CountEqual
            elif type == "Count Less":
                return CountLess
            else:
                return lambda x: x
        
        paths = self.type["paths"]
        subqueries = []
        for path in paths:
            partial = []
            for sub in path:
                partial.append(SubQuery(sub["color"], sub["type"], sub["pos"]))
            hookup(partial)
            subqueries.append(partial)
        query = Query([sub[0] for sub in subqueries],self.type["format"],select_end_filter(self.type["end"]),
                      self.graph.get_ego_node().heading, 
                      self.graph.get_nodes())
        return query
                

def QtoEn(paths):
    result = []
    for path in paths[::-1]:
        string = ""
        for q_spec in path[::-1]:
            string += subQtoEn(q_spec)
        result.append(string)
    if len(result) == 1:
        result = result[0]
    return result

def subQtoEn(spec):
    #print(spec)
    pos_string = decode_pos(spec["pos"])
    color_string = decode_color(spec["color"])
    type_string = decode_type(spec["type"])
    return "{}{}{}".format(color_string, type_string, pos_string)

def decode_pos(pos_strings):
    if not pos_strings:
        return ""
    direction_string = pos_strings[0]
    prefix = "that is "
    modifier = ""
    if direction_string == 'l':
        modifier = 'directly to the left of '
    elif direction_string == 'lf':
        modifier = 'to the left and in front of '
    elif  direction_string == 'lb':
        modifier = 'to the left and behind '
    elif  direction_string == 'r':
        modifier = 'directly to the right of '
    elif  direction_string == 'rf':
        modifier = 'to the right and in front of '
    elif  direction_string == 'rb':
        modifier = 'to the right and behind '
    elif  direction_string == 'b':
        modifier = 'directly behind '
    else:
        modifier = 'directly in front of '
    return prefix + modifier

def decode_color(colors):
    if not colors:
        return ""
    return colors[0].lower() + " "

def decode_type(types):
    if not types:
        return ""
    return types[0].lower() + " "

def decode_predicate(type):
    if type == "Count Larger":
        return lambda x : "Are there more {}than {}?".format(x[0],x[1])
    elif type == "Count Equal":
        return lambda x : "Are there equal number of {} and {}?".format(x[0],x[1])
    else:
        return lambda x : "Are there less {}than {}?".format(x[0],x[1])






def counting_translator():
    def func(query_string):
        result = "How many {}are visible and lidar detectable?".format(query_string)
        return result
    return func




logic_example = dict(
    format = "logical",
    paths = [
        [
            dict(
                type = ["Vehicle"],
                color = ["White"],
                pos = None),
            dict(
                type = ["Vehicle"],
                color = ["Grey"],
                pos = ["l"]),

        ],
        [
            dict(
            type = ["Vehicle"],
            color = ["Red"],
            pos = None)
        ]
    ],
    end = "Count Less"
)
# Is there more grey sportscar that is in front of white vehicles than red vehicles
# <end>  <subq2,q1>                           <subq1,q1>           <end> <q2>
#                        <q1>


counting_example = dict(
    format = "count",
    paths =  [
        [
            dict(
                type = ["Vehicle"],
                color = ["White"],
                pos = ["lf"]),
            dict(
                type = ["Sportscar"],
                color = ["Grey"],
                pos = ["l"]
            )
        ]
    ],
    end = "count"
)

#How many grey sportscar that is in front of white vehicles are visible and lidar detectable?
#   <end>                     <q1>                                         <end>

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
    test = QuestionSpecifier(counting_example, graph)
    prophet = QueryAnswerer(graph, [test.translate_to_Q()])
    print(test.translate_to_En())
    print(prophet.ans())
    #print(test.translate_to_Q())



    
