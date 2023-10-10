import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
from collections import defaultdict
OPENAI_KEY = "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"
from question_generator import Query,SubQuery,count
from scene_graph import SceneGraph
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
        if self.type["format"] == "Counting":
            return counting_translator()("white cars")
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
            return lambda x: x
        
        paths = self.type["paths"]
        subqueries = []
        for path in paths:
            partial = []
            for sub in path:
                partial.append(SubQuery(sub["color"], sub["type"], sub["pos"]))
            hookup(partial)
            subqueries.append(partial)
        query = Query([sub[0] for sub in subqueries],self.type["format"],select_end_filter(self.type["format"]),
                      self.graph.get_ego_node().heading, 
                      self.graph.get_nodes())
        return query
            
            


def counting_translator():
    def func(query_string):
        result = "How many {} are visible and lidar detectable?".format(query_string)
        return result
    return func




dict(
    format = "Logical",
    paths = [
        [
            dict(
                type = ["vehicle"],
                color = ["White"],
                position = None),
            dict(
                type = ["sportscar"],
                color = ["Grey"],
                position = ["f"]
            )
        ],
        [
            dict(
            type = ["vehicle"],
            color = ["Red"],
            position = None)
        ]
    ],
    end = "Count Larger"
)
# Is there more grey sportscar that is in front of white vehicles than red vehicles
# <end>  <subq2,q1>                           <subq1,q1>           <end> <q2>
#                        <q1>


counting_example = dict(
    format = "Counting",
    paths =  [
        [
            dict(
                type = ["vehicle"],
                color = ["White"],
                position = None),
            dict(
                type = ["sportscar"],
                color = ["Grey"],
                position = ["f"]
            )
        ]
    ],
    end = "counting"
)

#How many grey sportscar that is in front of white vehicles are visible and lidar detectable?
#   <end>                     <q1>                                         <end>

if __name__ == "__main__":
    test = QuestionSpecifier(counting_example)
    print(test.translate_to_En())



    
