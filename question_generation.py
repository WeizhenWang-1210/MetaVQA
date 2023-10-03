import json
import random
import numpy as np
import argparse
import glob
from typing import Callable, Any, Iterable
from collections import defaultdict
from pprint import pprint

#import openai
#import asyncio

OPENAI_KEY = "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"
#don't work when the scene is crowded.

"""
Note that MetaDrive uses a right-handed coordinate system with x being the heading direction
           
                
                | +x    
                |
    +y          |     (top-down view, +z coming out of the screen)
   ----------------------------------

"""

from scene_graph import scene_graph
from agent_node import AgentNode
        



class Question_Generator:
    CODE_TO_ROAD = {
        'S': "straight road",
        'C': "curved road",
        'r': "in-ramp",
        'R': "out_ramp",
        'O': "roundabout",
        'X': "intersection",
        'T': 'T-intersection',
        'y': 'merge',
        'Y': 'split',
        '$': 'tollgate',
        'P': "parking lot",
        "WIP":'fork',
        '>': "straight road"
    }

    COLORS = set(
        ["White", "Black", "Grey", "Blue", "Red", "Green", "Orange","Yellow","Purple"]
    )

    OBJECTS = set(
        ["SUV", "Sedan", "Truck", "Sportscar", "Traffic cone", "Pedestrian", "Warning sign"]
    )






    def __init__(self, graph:scene_graph, target:int = 1, allowance: int = 4, dir:str = './'):
        self.scenario_graph = graph
        self.ground_truths =  self.scenario_graph.spatial_graph
        #stats for one particular scenes
        self.stats = dict(
            simple = 0,
            composite = 0,
            resolute = 0,
            spatial = defaultdict(lambda:0),
            relative_ego = [],
            resoluter = defaultdict(lambda:0),
            relative_compared = []
        )
        self.dir = dir

    def generate_all(self, ego:str = None):
        if ego is None:
            ego = self.scenario_graph.ego_id
        simple_datapoints = []
        composite_datapoints = []
        for node in self.scenario_graph.nodes.keys():
            if node != ego and self.scenario_graph.nodes[node].type not in ["cone", "planar barrier", "warning sign"]:
                simple,candidate_1,resoluter,_ = self.generate(ego = ego,referred=node)
                composites, candidate_2 = self.generate_two_hops(ego = ego,referred=node)
                if simple[0] is not None:
                    road_string = 'The car is on a {}.'.format(self.CODE_TO_ROAD[self.scenario_graph.nodes[candidate_1].road_code])
                    simple_datapoints.append((self.convert_to_str(simple) + road_string,candidate_1))
                if len(composites) > 0:
                    road_string = 'The car is on a {}.'.format(self.CODE_TO_ROAD[self.scenario_graph.nodes[candidate_2].road_code])
                    composite_datapoints += [
                        (self.convert_to_str(composite)+ road_string,candidate_2,composite[2]) for composite in composites
                    ]
        return simple_datapoints + composite_datapoints
           
    def generate(self, ego:str = None, referred:str = None, ignore:list = None, allowance:int = 10):
        """
        Generate a path with only one node.
        """
        def check_same_side(ref,node):
            graph = self.scenario_graph
            return graph.check_sameSide(ref,node)
        def check_different_side(ref,node):
            graph = self.scenario_graph
            return not graph.check_sameSide(ref,node)
        """def check_different_street(ref,node):
            graph = self.scenario_graph
            return not (graph.check_sameStreet(ref,node))"""
        def check_taller(ref,node):
            #node is taller
            graph = self.scenario_graph
            return graph.nodes[node].height>graph.nodes[ref].height
        def check_shorter(ref,node):
            graph = self.scenario_graph
            return graph.nodes[node].height<graph.nodes[ref].height
        checkers = {
            'same_side':check_same_side,
            'different_side':check_different_side,
            #'different_street':check_different_street,
            'check_taller':check_taller,
            'check_shorter':check_shorter,
        }
        #ego = self.scenario_graph.ego_id
        candidate_id = self.select_not_from([ego]) if referred is None else referred
        dir, peers = self.find_direction(ego, candidate_id)
        choice_1,choice_2 = None,None
        resoluter = None
        if len(peers) == 1:
           choice_1 = dir, None
        unique_properties = []
        for name, checker in checkers.items():
            if self.check_unique(checker,ego, candidate_id, peers):
                unique_properties.append(name)
        if len(unique_properties)!=0:
            resoluter = random.choice(unique_properties)
            choice_2 = dir,resoluter
        picked = try_return_random_valid(choice_1,choice_2)
        if resoluter is not None:
            debug_info = [(node,checkers[resoluter](ego,node)) for node in peers] 
        else:
            debug_info = "No info"
        #updating statistics
        if picked is not None and ego == self.scenario_graph.ego_id:
            d = distance(self.scenario_graph.nodes[ego],self.scenario_graph.nodes[candidate_id])
            direction, resoluter = picked
            self.stats["simple"] += 1
            if resoluter:
                self.stats["resolute"] += 1 
                self.stats["resoluter"][resoluter] += 1
            self.stats["spatial"][direction] += 1
            self.stats["relative_ego"].append(d)
        return [picked], candidate_id,resoluter, debug_info
            
    def generate_two_hops(self, ego: str, referred:str = None, allowance:int = 10):
        """
        Generate a path with one intermediate node
        """
        #ego = self.scenario_graph.ego_id
        candidate_id = self.select_not_from([ego]) if referred is None else referred
        compared = [node_id for node_id in self.scenario_graph.nodes.keys() if node_id not in set([ego, candidate_id])]
        intermediate_ids = sorted(compared,
                             key = lambda node_id: distance(self.scenario_graph.nodes[candidate_id], self.scenario_graph.nodes[node_id])
                             )
        paths = []
        for intermediate_id in intermediate_ids:
            path = []
            path.append((self.find_direction(ego, intermediate_id)[0],None))
            last_path,_,_,_ = self.generate(intermediate_id, candidate_id)
            if last_path[0] != None:
                path.append(last_path[0])
                ego_d = distance(self.scenario_graph.nodes[ego],self.scenario_graph.nodes[candidate_id])
                relative_d = distance(self.scenario_graph.nodes[intermediate_id],self.scenario_graph.nodes[candidate_id])
                ego_dir,_ = self.find_direction(ego, candidate_id)
                _,resoluter = last_path[0]
                self.stats["composite"] += 1
                self.stats["spatial"][ego_dir]+=1
                self.stats["relative_compared"].append(relative_d)
                self.stats["relative_ego"].append(ego_d)
                if resoluter:
                    self.stats["resolute"]+=1
                    self.stats["resoluter"][resoluter]+=1
                path.append(intermediate_id)
                paths.append(path)
        return paths, candidate_id

    def convert_to_str(self, path:Iterable):
        """
        Given a spatial relationship path to resolute a target node from ego, return the English interpretation
        of this spatial relationship. 
        """
        def convert_resoluter(resoluter):
            mapping = {
                'same_side':'on the same side of the street as',
                'different_side':'on the different side of the street as',
                'check_taller':"is taller than",
                'check_shorter':"is lower than",
            }
            return mapping[resoluter]
        Begin = "The car that is "
        if len(path) == 3:
            intermediate_id = path[-1]
            intermediate_type = self.scenario_graph.nodes[intermediate_id].type
            ref = '{} {} '.format("another" if intermediate_type=="car" else "a",intermediate_type)
            dir1, resoluter = path[1]
            dir2,_ = path[0]
            part_1 = self.generate_spatial_modifier(dir1)
            suffix_1 = "that is "
            part_2 = self.generate_spatial_modifier(dir2)
            center = "ego."
            resoluter = "" if resoluter is None else "The car is {} the {} {}.".format(convert_resoluter(resoluter),
                                                                                       "other" if intermediate_type=="car" else "",
                                                                                       intermediate_type)
            return Begin + part_1 + ref +suffix_1+ part_2 + center + resoluter
        else:
            dir, resoluter = path[0]
            part_1 = self.generate_spatial_modifier(dir)
            center = "ego."
            resoluter = "" if resoluter is None else "The car is {} ego.".format(convert_resoluter(resoluter))
            return Begin + part_1 + center + resoluter
    
    def generate_spatial_modifier(self, direction_string:str)->str:
        """
        Return the English meaning encoded by the direction_string
        """
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
        return modifier

    def check_unique(self, checker:Callable, ref: str, candidate:str, peers:Iterable)->bool:
        """
        Return True if checker returns true ONLY for candidate. ref is here when spatial relationships are examined
        and the heading of the ego vehicle needs to be passed into the checker.
        """
        for peer in peers:
            if peer == candidate:
                if checker(ref,peer):
                    continue
                else:
                    return False
            else:
                if not checker(ref,peer):
                    continue
                else:
                    return False
        return True

    def find_direction(self, origin:str, target:str)->tuple[str,list[str]]:
        """
        Return the direction of target w.r.t. to origin and all the nodes that also fall along in that direction.
        """
        for dir, peers in self.ground_truths[origin].items():
            if target in peers:
                return dir, peers
        print('Error in finding node %s in node %s \'s coordinate' %(target, origin))
        exit()
 
    def select_not_from(self, ban_list:list)->str:
        """
        Select a random node in the scene graph with id not in the ban list
        """
        nodes = list(self.scenario_graph.nodes.keys())
        result = random.choice(nodes)
        while result in ban_list:
            result = random.choice(nodes)
        return result

    def get_stats(self):
        return self.stats    

    def generate_counting(self, type: str = "car", property:str = "color", val: str = "white")->(str, int):





        """
        How many <object> are observable?
        How many <object> are
        
        """



        format = "How many {} with {} of {} are there?".format(type, property, val)
        ans = 0
        for id, object in self.scenario_graph.nodes.items():
            if object.type == type:
                if property == "color":
                    ans += 1 if object.color == val else 0
                if property == ""
        return format, ans


    """
    Filter, objects
    color = Red, Type = White, Spatial = None
    """





def transform(ego:AgentNode,bbox:Iterable)->Iterable:
    def change_bases(x,y):
        relative_x, relative_y = x - ego.pos[0], y - ego.pos[1]
        new_x = ego.heading
        new_y = (-new_x[1], new_x[0])
        x = (relative_x*new_x[0] + relative_y*new_x[1])
        y = (relative_x*new_y[0] + relative_y*new_y[1])
        return x,y
    return [change_bases(*point) for point in bbox]

def distance(node1:AgentNode, node2:AgentNode)->float:
    dx,dy = node1.pos[0]-node2.pos[0], node1.pos[1]-node2.pos[1]
    return np.sqrt(dx**2 + dy**2)

if __name__ == '__main__':
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
    else:
        assert args.step is not None
        with open('{}.json'.format(args.step),'r') as scene_file:
            scene_dict = json.load(scene_file)
        agent_id,nodes = nodify(scene_dict)
        graph = scene_graph(agent_id,nodes)
        test_generator = Question_Generator(graph)
        points= test_generator.generate_all()
        print(test_generator.generate_counting())
        print(test_generator.scenario_graph.nodes.values())

        
    
    

    """ openai.api_key = OPENAI_KEY
    response  = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": 
                 Rephrase the following text in questions, then rephrase those questions into imperatives:
                 }
            ]
    )"""
    """
    Prompt: 
    rephrase 10 times: "The car that is to the r         ight and behind us and heading the same direction as us."
    Return:
    1. "The vehicle positioned on our right side, behind us, and traveling in the same direction as us."
    2. "The car situated to our right and following us while heading in the same direction."
    3. "The automobile that is behind us, on our right side, and moving parallel to our direction."
    4. "The car located behind us, to the right, and traveling alongside us."
    5. "The vehicle positioned on the right-hand side behind us, going in the same direction."
    6. "The car that is trailing behind us, situated to our right, and moving in sync with our direction."
    7. "The automobile behind us, on the right side, and maintaining the same direction as us."
    8. "The car following us, located to our right side and moving parallel to our path."
    9. "The vehicle behind us, positioned on the right-hand side, and journeying in tandem with us."
    10. "The car that is both behind us and to our right, traveling in alignment with our direction."
    
    Prompt:
    rephrase as questions
    Return:
    1. Is there a car on our right side, behind us, and heading in the same direction as us?
    2. Which vehicle is positioned behind us, on our right side, and traveling in the same direction as us?
    3. Do we have a car trailing us, located to our right and moving parallel to our direction?
    4. Is there an automobile behind us, on our right side, and maintaining the same direction as us?
    5. Are we being followed by a car on our right-hand side, going in the same direction?
    6. Which vehicle is trailing behind us, situated to our right, and moving in sync with our direction?
    7. Is there a car following us, located to our right side, and moving parallel to our path?
    8. Do we have a vehicle behind us, positioned on the right-hand side, and journeying in tandem with us?
    9. Is there a car both behind us and to our right, traveling in alignment with our direction?
    10. Which car is positioned to the right and behind us, heading the same direction as us?
    
    ...

    Prompt:
    Rephrase as imperatives
    Return:
    1. Look for the car on our right side, behind us, heading in the same direction as us.
    2. Spot the vehicle positioned behind us, on our right side, traveling in the same direction.
    3. Observe the car trailing us, located to our right and moving parallel to our direction.
    4. Pay attention to the automobile behind us, on our right side, maintaining the same direction as us.
    5. Keep an eye out for the car following us on our right-hand side, going in the same direction.
    6. Notice the vehicle trailing behind us, situated to our right, and moving in sync with our direction.
    7. Watch the car following us, located to our right side, and moving parallel to our path.
    8. Identify the vehicle behind us, positioned on the right-hand side, journeying in tandem with us.
    9. Take note of the car both behind us and to our right, traveling in alignment with our direction.
    10. Be aware of the car that is to the right and behind us, heading the same direction as us.
    """




        