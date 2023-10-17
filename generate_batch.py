import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
OPENAI_KEY = "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"
from question_generator import Query,SubQuery,count, QueryAnswerer,CountGreater, CountEqual, locate,CountLess 
from scene_graph import SceneGraph
from agent_node import nodify
import yaml
from itertools import chain, combinations,product
from question_generation import QuestionSpecifier
from NAMESPACE import namespace


def powerset(iterable: Iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))



def cartesianproduct(*space: Iterable):
    result = []
    for i in product(*space):
        #print(i)
        result.append(i)
    return result

def clean(pools, funcs):
    result = []
    for pool in pools:
        decide = True
        for func in funcs:
            decide = decide and func(pool)
        if decide:
            result.append(pool)
    return result

def construct_question(spec, format, end, graph):
    color, type, pos = spec
    path = dict(
        color = [color] if color else None,
        type = [type] if type else None,
        pos = [pos] if pos else None
    )
    question  = dict(
        format = format,
        end  = end,
        paths = [[path]]
    )
    #print(question)
    return QuestionSpecifier(question,graph)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = bool, default= True)
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
            graph = SceneGraph(agent_id,nodes)
            prophet = QueryAnswerer(graph, [])
            specs = cartesianproduct(namespace["color"],namespace["type"],namespace["pos"])
            questions_categories = cartesianproduct(["localize", "count"],["count", "localize", "Count Greater", "Count Less", "Count Equal"])
            questions_categories = [["localize","localize"], ["count","count"]]#cartesianproduct(["logical"],["Count Greater", "Count Less", "Count Equal"]) +\     
            specs = clean(specs, [lambda x: x[0] or x[1] or x[2], lambda x : x[1] is not None])
            for spec in specs:
                for format, end in questions_categories:
                    single_length_question_specifier = construct_question(spec, format,end, graph)
                    English = single_length_question_specifier.translate_to_En()
                    ans = prophet.ans(single_length_question_specifier.translate_to_Q())
                    if not (len(ans) == 0 or ans[0]==0):
                        print(English, ans)                    
                    

    











    








           
