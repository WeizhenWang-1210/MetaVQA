import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
from collections import defaultdict
from vqa.question_generator import Query,SubQuery, QueryAnswerer, locate
from vqa.functionals import count, CountGreater, CountEqual, CountLess, Describe, locate_wrapper
from vqa.miscellaneous.temporal_question_generator import TempSubQuery, TempQuery, TempQueryAnswerer
from vqa.scene_graph import SceneGraph
from vqa.object_node import nodify
import os

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
"""
format:= count | logical | localize
end:= count | localize | Count Greater | Cout Less | Count Equal
"""

pwd = os.getcwd()
template_path = os.path.join(pwd, "vqa/question_templates.json")
try:
    with open(template_path,"r") as f:
        templates = json.load(f)["generic"]
except Exception as e:
    print(e)

templates = templates["general"]

class QuestionSpecifier: # 1-to-1 corresponding relationship with a particular query
    """
    The intermediate representation of a question. Used to intantiate the functional program associated with this representation 
    and to generate the English translation of this representation. Bridge English with functional program queries.
    
    """
    def __init__(self,
                 config,
                 graph: SceneGraph) -> None:
        
        """
        In the current English translation implementation, only can't generate fluent English like "Green Car or Red Car". In 
        addition, can only generate logical questions with two threads.
        """
        self.config= config
        self.template = config["format"]
        self.graph = graph

    def translate_to_En(self):
        """
        Generate the corresponding English Question.
        """
        #TODO Add more templates to the same question for diversity
        #The translator is a kind of function that takes in an str and supply it to a predefined question template
        format = self.config["format"]


        text_template = self.template['text']
        params = {param:"" for param in self.template["params"]}
        constraint = self.template["constraint"]
        for param in params.keys():
            if param[:2]=="<o":
                params[param] = self.QtoEn(self.configs["entities"])
            elif param[:2]=="<a":
                params[param] = self.configs["actions"]
            else:
                par






        if format == "count":
            translator = counting_translator_generator()
        elif format == "temporal":
            def decode_predicate():
                return lambda x: "What are the {} that we just {}?".format(x[0], x[1])

            translator = decode_predicate()
            return translator(self.TempQtoEn(self.config["paths"]))
        elif format == "logical":
            def decode_predicate(type):
                '''
                Convert predicate functional programs to answers
                '''
                if type == "Count Greater":
                    return lambda x : "Are there more {}than {}?".format(x[0],x[1])
                elif type == "Count Equal":
                    def func(x):
                        if len(x) == 2:
                            str = "and ".join(x)
                        else:
                            listing = ",".join(x)
                            str = "the following:{}".format(listing)
                        return "Are there equal number of {}?".format(str)
                    return func
                else:
                    return lambda x : "Are there less {}than {}?".format(x[0],x[1])
            translator = decode_predicate(self.config["end"])
        elif format == "localize":
            translator = localization_translator_generator()
        return translator(self.QtoEn(self.config["paths"]))
    def TempQtoEn(self,paths):
        """
        Convert List[List[dict]] into English description of the objects referred.
        """
        assert len(paths) == 1
        assert len(paths[0]) == 1
        result = [paths[0][0]["obj"], paths[0][0]["action"]]
        return result
    def QtoEn(self,paths):
        """
        Convert List[List[dict]] into English description of the objects referred. 
        """
        result = []
        for path in paths:
            string = ""
            for q_spec in path[::-1]:
                string += self.subQtoEn(q_spec)
            string += "us " if path[0]["pos"] else ""
            result.append(string)
        if len(result) == 1:
            result = result[0]
        return result
    
    def subQtoEn(self,spec):
        """
        Takes dict(type = , color = , pos = ) and convert this into English
        """
        #TODO create a dictionary that the function can consult
        def decode_pos(pos_strings):
            """
            Convert str-encoded pos_strings into English
            """
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
            """Convert string-encoded color into English"""
            if not colors:
                return ""
            return colors[0].lower() + " "

        def decode_type(types):
            """
            Convert string-encoded type into English
            """
            if not types:
                return "thing" + " "
            return types[0].lower() + " "

        pos_string = decode_pos(spec["pos"])
        color_string = decode_color(spec["color"])
        type_string = decode_type(spec["type"])
        return "{}{}{}".format(color_string, type_string, pos_string)

    def translate_temporal_to_Q(self):
        '''
        Get the functional program
        '''

        def hookup(subqueries):
            '''string the subqueries together'''
            prev = None
            for subquery in subqueries:
                if prev:
                    prev.next = subquery
                subquery.prev = prev
                prev = subquery

        def select_end_filter(type):
            '''Given the stringfy type of questiosn, return the corresponding end filter'''
            if type == "count":
                return count
            elif type == "Count Greater":
                return CountGreater
            elif type == "Count Equal":
                return CountEqual
            elif type == "Count Less":
                return CountLess
            elif type == "localize":
                func = locate_wrapper(self.graph.get_ego_node())
                return func
            elif type == "Describe":
                return Describe


        paths = self.config["paths"]
        subqueries = []
        for path in paths:
            partial = []
            for sub in path:
                partial.append(TempSubQuery(obj=sub['obj'], action=sub['action']))
            hookup(partial)
            subqueries.append(partial)
        query = TempQuery([sub[0] for sub in subqueries], self.config["format"], select_end_filter(self.config["end"]),
                      self.graph.get_ego_node().heading,
                      self.graph.get_nodes())
        return query
    def translate_to_Q(self):
        '''
        Get the functional program
        '''
        def hookup(subqueries):
            '''string the subqueries together'''
            prev = None
            for subquery in subqueries:
                if prev:
                    prev.next = subquery
                subquery.prev = prev
                prev = subquery
        def select_end_filter(type):
            '''Given the stringfy type of questiosn, return the corresponding end filter'''
            if type == "count":
                return count
            elif type == "Count Greater":
                return CountGreater
            elif type == "Count Equal":
                return CountEqual
            elif type == "Count Less":
                return CountLess
            elif type == "localize":
                func = locate_wrapper(self.graph.get_ego_node())
                return func
        
        paths = self.config["paths"]
        subqueries = []
        for path in paths:
            partial = []
            for sub in path:
                partial.append(SubQuery(sub["color"], sub["type"], sub["pos"]))
            hookup(partial)
            subqueries.append(partial)
        query = Query([sub[0] for sub in subqueries],self.config["format"],select_end_filter(self.config["end"]),
                      self.graph.get_ego_node().heading, 
                      self.graph.get_nodes())
        return query
        

"""
config = dict(
    format = "", #str
    paths = [   #Iterable[Iterable]
        [],
        [],
        []
    ],
    end = "" #str
)

"""           


def localization_translator_generator():
    '''
    return a localization end
    '''
    def func(query_string):
        result = "Locate the {}".format(query_string)
        return result
    return func

def counting_translator_generator():
    '''
    Generate English format for counting problem
    '''
    def func(query_string):
        result = "How many {}are visible and lidar detectable?".format(query_string)
        return result
    return func

good_example = dict(
    format = "counting",
    entities = [
        [
            dict(
                type = ["SportCar"],
                color = ["Gray"],
                pos = None),
        ]
    ]
    actions = [
        "present"
    ]
)






logic_example = dict(
    format = "logical",
    paths = [
        [
            dict(
                type = ["SportCar"],
                color = ["Gray"],
                pos = None),
        ],
        [
            dict(
            type = ["Policecar"],
            color = ["Black"],
            pos = None)
        ],
        [
            dict(
            type = ["Vehicle"],
            color = ["Black"],
            pos = None)
        ]
    ],
    end = "Count Equal"
)
# Is there more grey sportscar that is in front of white vehicles than red vehicles
# <end>  <subq2,q1>                           <subq1,q1>           <end> <q2>
#                        <q1>

counting_example = dict(
    format = "count",
    paths =  [
        [
             dict(
                type = ["dog"],
                color = None,
                pos = None),
        ]
    ],
    end = "count"
)


counting_example_2 = dict(
    format = "count",
    paths =  [
        [
            dict(
                type = ["SUV"],
                color = None,
                pos = None
            )
        ]
    ],
    end = "count"
)


composite_example = dict(
    format = "count",
    paths = [
        [
             dict(
                type = ["BusStop"],
                color = None,
                pos = ["lf"]
            ),
        ]
    ],
    end = "count"
)

localization_example = dict(
    format = "localize",
    paths = [
        [
             dict(
                type = ["Vehicle"],
                color = None,
                pos = ['f']
            )
        ]
    ],
    end = "localize"
)



#How many grey sportscar that is in front of white vehicles are visible and lidar detectable?
#   <end>                     <q1>                                         <end>

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
    test = QuestionSpecifier(logic_example, graph)
    prophet = QueryAnswerer(graph, [])
    print(test.translate_to_En())
    print(prophet.ans(test.translate_to_Q()))
    test = QuestionSpecifier(counting_example, graph)
    print(test.translate_to_En())
    print(prophet.ans(test.translate_to_Q()))
    test = QuestionSpecifier(counting_example_2, graph)
    print(test.translate_to_En())
    print(prophet.ans(test.translate_to_Q()))
    test = QuestionSpecifier(composite_example, graph)
    print(test.translate_to_En())
    print(prophet.ans(test.translate_to_Q()))
    test = QuestionSpecifier(localization_example, graph)
    print(test.translate_to_En())
    print(prophet.ans(test.translate_to_Q()))

    #print(test.translate_to_Q())



    
