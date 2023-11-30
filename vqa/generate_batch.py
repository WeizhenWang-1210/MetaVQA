import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
#from APIKEY import OPEN_AI_KEY
from vqa.question_generator import Query,SubQuery,count, QueryAnswerer,CountGreater, CountEqual, locate,CountLess 
from vqa.scene_graph import SceneGraph
from vqa.object_node import nodify
import yaml
from itertools import chain, combinations,product
from question_generation import QuestionSpecifier
from vqa.configs.NAMESPACE import namespace


def powerset(iterable: Iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))



def choose_two(iterable: Iterable):
    s = list(iterable)
    return list(combinations(s,2))


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
    #print(spec, format, end, graph)
    if format != "logical":
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

    else:
        paths = []
        for s in spec:
            color, type, pos = s
            path = dict(
                color = [color] if color else None,
                type = [type] if type else None,
                pos = [pos] if pos else None
            )
            paths.append([path])
        
        question  = dict(
            format = format,
            end  = end,
            paths = paths
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
            data = dict()
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
            logical_combinations = cartesianproduct(["logical"],["Count Greater","Count Less", "Count Equal"])
            non_logical_combinations = [("localize","localize"),("count","count")]
            specs = clean(specs, [lambda x: x[0] or x[1] or x[2], lambda x : x[1] is not None])
            logical_specs = choose_two(specs)
            logical_specs = clean(logical_specs, [lambda xs: xs[0][1]!=xs[1][1]])
            specs = [(None, "Motorcycle",None),(None, "Bike", None),[(None, "Sportscar", None),(None,"Bike", "l")]]
            logical_specs = [[(None, "Bike", None),(None, "Motorcycle", None)]]



            idx = 0
            
            for spec in specs:
                for format, end in non_logical_combinations:
                    print(spec, format, end)
                    single_length_question_specifier = construct_question(spec, format,end, graph)
                    English = single_length_question_specifier.translate_to_En()
                    ans = prophet.ans(single_length_question_specifier.translate_to_Q())
                    if not (len(ans) == 0 or ans[0]==0):
                        #remeber to translate the coordinates to ego's perspective
                        data[idx] = dict(
                            q= English,
                            a= ans
                        )     
                        idx += 1
            print(idx+1)
            for spec in logical_specs:
                spec_1, spec_2 = spec
                spec_1_specifier = construct_question(spec_1, "count", "count", graph)
                spec_2_specifier = construct_question(spec_2, "count", "count", graph)
                if prophet.ans(spec_1_specifier.translate_to_Q())[0] == 0 or prophet.ans(spec_2_specifier.translate_to_Q())[0] == 0:
                    continue
                for format, end in logical_combinations:
                    print(format, end)
                    single_length_question_specifier = construct_question(spec, format,end, graph)
                    English = single_length_question_specifier.translate_to_En()
                    ans = prophet.ans(single_length_question_specifier.translate_to_Q())
                    data[idx] = dict(
                            q= English,
                            a= ans
                    )     
                    idx += 1
            print(idx+1)
            try:
                with open(root + '/qa_{}.json'.format(splitted[1]),'w') as file:
                    json.dump(data,file)
            except:
                print("wtf")
            
                           
                    

    











    








           
