import json
import random
import numpy as np
import argparse

#import openai
#import asyncio

OPENAI_KEY = "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"
#don't work when the scene is crowded.

class agent_node:
    """
        Indicators:
            Left  | Right | colinear:  -1 | 1 | 0
            Back  | Front | colinear:  -1 | 1 | 0
            Same Side(of road) | Different Side:  1 | 0
    """
    def __init__(self,
                 pos = (0,0), #object.position
                 color = (0,0,0), #Stuck. Can't rerender the objects' color. Look at Material from Panda3D
                 speed = 1,      #object.speed
                 heading = (0,1), #object.heading
                 lane = (0,1), #lane indexing is (id1, id2, lane number(beginning from 0, left to right)). In addition, the other side of the street is
                 #(-id1,-id2, #)
                 id = None,
                 bbox = None,
                 type = "vehicle"):
        #More properties could be defined.
        self.pos =  pos #(x,y) w.r.t. to world origin
        self.color = color
        self.speed = speed
        self.heading = heading #(dx, dy) unit vector, centered at the car but in world coordinate
        self.lane = lane
        self.id = id
        self.bbox = bbox
        self.type = type
    
    def compute_relation(self, node):
        relation = {
            'side': self.leftORright(node),
            'front': self.frontORback(node),
            'lane': self.sameSide(node),
            'street': self.sameStreet(node),
            'steering': self.steering_leftORright(node)
            #etc.
        }
        return relation
    
    def __str__(self):
        dictionary = {
            'pos': self.pos,
            'color' : self.color,
            'speed' : self.speed,
            'heading' : self.heading,
            'lane' : self.lane
        }
        return dictionary.__str__()
    
    def leftORright(self,node):
        '''
        On the same side of a road, you can find this informaion by comparing the lane index.
        '''
        """if self.lane[0]==node.lane[0] and self.lane[1]==node.lane[1] or\
            self.lane[0]==node.lane[1] or\
            self.lane[1]==node.lane[0]:
            lane_idx, node_idx = self.lane[2],node.lane[2]
            if node_idx == lane_idx:
                return 0"""
        ego_coord = (node.pos[0] - self.pos[0], node.pos[1] - self.pos[1])
        l2_norm = np.sqrt(ego_coord[0]**2 + ego_coord[1]**2)
        unit_ego_coord = (ego_coord[0]/l2_norm, ego_coord[1]/l2_norm)
        cross = unit_ego_coord[0]*self.heading[1] - unit_ego_coord[1]*self.heading[0] #cross product

        if cross > 0.05:
            return 1
        elif cross < -0.05 :
            return -1
        else:
            return 0
        
    def frontORback(self,node):
        ego_coord = (node.pos[0] - self.pos[0], node.pos[1] - self.pos[1])
        dot = ego_coord[0] * self.heading[0] + ego_coord[1] * self.heading[1]
        if dot > 0.05:
            return 1
        elif dot < -0.05:
            return -1
        else: 
            return 0

    def sameSide(self,node): #simple case: same road
                            # more complicated, at turn around
        return  int(self.lane[0] == node.lane[0] and
                   self.lane[1] == node.lane[1]) or\
                int (self.lane[0]== node.lane[1]) or int (self.lane[1]==node.lane[0])
                
    
    def sameStreet(self, node):
        m_id0,m_id1 = self.lane[0],self.lane[1]
        n_id0,n_id1 = node.lane[0],node.lane[1]
        if ("-"+m_id0 == n_id0 and "-"+m_id1 == n_id1) or\
            (m_id0 == "-"+n_id0 and m_id1 == "-"+n_id1):
            return 1
        return 0
    
    def steering_leftORright(self,node):
        cross = node.heading[0]*self.heading[1] - node.heading[1]*self.heading[0] #cross product
        if cross > 0:
            return 1
        elif cross < 0 :
            return -1
        else:
            return 0

            

class scene_graph:
    def __init__(self, 
                 ego_id,
                 nodes = [],
                 ):
        self.nodes = {}
        for node in nodes:
            self.nodes[node.id] = node
        self.ego_id = ego_id
        self.graph= self.compute_graph()

    def refocus(self,new_ego_id = 0):
        self.ego_id = new_ego_id
    
    def compute_graph(self):
        graph = {}
        for node_id in self.nodes.keys():
            graph[node_id] = self.compute_multiedges(node_id)
        return graph

    def compute_multiedges(self, ego_id):
        edges = {
                    'l':[],
                    'r':[],
                    'f':[],
                    'b':[],
                    'lf':[],
                    'rf':[],
                    'lb':[],
                    'rb':[]
                }
        ego_node =self.nodes[ego_id]
        for node_id, node in self.nodes.items():
            if node_id != ego_id:
                relation = ego_node.compute_relation(node)
                side, front = relation['side'], relation['front']
                if side == -1 and front == 0:
                    edges['l'].append(node.id)
                elif side == -1 and front == -1:
                    edges['lb'].append(node.id)
                elif side == -1 and front == 1:
                    edges['lf'].append(node.id)
                elif side == 0 and front == -1:
                    edges['b'].append(node.id)
                elif side == 0 and front == 1:
                    edges['f'].append(node.id)
                elif side == 1 and front == 0:
                    edges['r'].append(node.id)
                elif side == 1 and front == -1:
                    edges['rb'].append(node.id)
                elif side == 1 and front == 1:
                    edges['rf'].append(node.id)
                else:
                    print("Erroenous Relations!")
                    exit()
        return edges

    def check_ambigious(self, origin, dir):
        candidates = self.graph[origin.id][dir]
        #-1 being UNSAT, 0 being Unambigious, 1 being ambigious
        if len(candidates)==0:
            return -1
        elif len(candidates)==1:
            return 0
        else:
            return 1
     
    def export(self, destination):
        #Compute Relations
        self.compute_relations()
        #Store Relations in a Dict, indexed by nodes
        #Export the Dictionary in JSON File, which would later be used to generate ground truth.
       
        try:
            print('self.compute_relations() worked')
            with open(destination, "w") as outfile:
                json.dump(self.relations,outfile)
            print('Exported to %s' %(destination))
            return True #if executed correctly.
        except:
            print("Something went wrong when exporting")
            return False

class Question_Generator():
    def __init__(self, graph):
        self.scenario_graph = graph
        self.ground_truths =  self.scenario_graph.graph

    def generate_simple(self, referred = None):
        #Pick a node to formulate a question
        ego = self.scenario_graph.ego_id
        candidate_id = self.select_not_from([ego]) if referred is None else referred
        dir, peers = self.find_direction(ego, candidate_id)
        ret = self.generate_description(dir, candidate_id, peers)
        if ret == 'reject':
            return ret, ego, candidate_id
        else:
            Heading,modifier,suffix = ret
            return Heading+modifier+suffix, ego, candidate_id
    

    
    def select_not_from(self, ban_list):
        nodes = list(self.scenario_graph.nodes.keys())
        result = random.choice(nodes)
        while result in ban_list:
            result = random.choice(nodes)
        return result
        
    def generate_description(self, direction_string, candidate_id, peer_ids):
        string = "The car that is "
        modifier = self.generate_modifier(direction_string)
        suffix = ""
        if len(peer_ids)==1:
            suffix = "ego. "   
        else: 
            def check_same_side(node):
                graph = self.scenario_graph
                return graph.nodes[node].sameSide(graph.nodes[self.scenario_graph.ego_id])
            def check_different_side(node):
                graph = self.scenario_graph
                return not (graph.nodes[node].sameSide(graph.nodes[self.scenario_graph.ego_id]))
            def check_different_street(node):
                graph = self.scenario_graph
                return not (graph.nodes[node].sameStreet(graph.nodes[self.scenario_graph.ego_id]))
            if self.check_unique(check_same_side, candidate_id, peer_ids):
                suffix = "The car is the only car on ego side of the road."
            elif self.check_unique(check_different_side, candidate_id, peer_ids):
                suffix = "The car is the only car not on ego side of the road."
            elif self.check_unique(check_different_street,candidate_id, peer_ids):
                suffix = "The car is the only car not on the same street as ego."
            #TODO: Use distance ordering for resolution
        if suffix == "": #meaning, ambigious
            return "reject"
        else:
            return string, modifier, suffix
    def generate_modifier(self, direction_string):
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

    def check_unique(self, checker, candidate, peers):
        for peer in peers:
            if peer == candidate:
                if checker(peer):
                    continue
                else:
                    return False
            else:
                if not checker(peer):
                    continue
                else:
                    return False
        return True

    def find_direction(self, origin, target):
        for dir, peers in self.ground_truths[origin].items():
            if target in peers:
                return dir, peers
        print('Error in finding node %s in node %s \'s coordinate' %(target, origin))
        exit()

    def generate_comparative(self, referred = None, compared = None):
        """
        Generate questions in which the target objects are disambiguated with comparative informations involving cars
        other than ego. For example, if there are two cars to the side of the ego, a question could be
                                "Pinpoint the car further to the right."
        And the ground truth would be the car on the right that's farthest from ego.

        Type 1: Single Dimension (Further/Closer along one dimension)
        Type 2: Double Dimension (In front of a car that's to the right of the ego)
        Type 3: Composite (In front of the car that is farther to the right of ego)

        

        Type 1: Disambiguate by itself. 
                "To the right, to the left, to the front, to the back, same direction, different direction" (w.r.t. the ego)
        Type 2: Disambiguate with <relation> involving another object
                "Further left, further right, in front of the car to the right         
        Type 3: Disambiguate with <relation> involving two objects 
        """
        ego = self.scenario_graph.ego_id
        candidate_id = self.select_not_from([ego]) if referred is None else referred
        comparative_candidate_ids = [idx  
                                     for idx in self.scenario_graph.nodes.keys() 
                                     if idx not in [ego, candidate_id]]
        #print(comparative_candidate_ids, self.scenario_graph.ego_id, candidate_id)
        if len(comparative_candidate_ids)==0:
            return 'reject', ego, candidate_id, "None"
        comparative_candidate_id = random.choice(comparative_candidate_ids) if compared is None else compared
        dir_ego, _ = self.find_direction(ego, comparative_candidate_id)
        modifier_ego = self.generate_modifier(dir_ego)
        candidate_dir, candidate_peers = self.find_direction(comparative_candidate_id, candidate_id)
        ret = self.generate_description(candidate_dir,candidate_id,candidate_peers)
        if ret == 'reject':
            return ret, ego, candidate_id, comparative_candidate_id
        else:
            Heading,modifier,suffix = ret
            final = Heading + modifier + "another car that is " + modifier_ego  + suffix  
            return final, ego, candidate_id, comparative_candidate_id
           
def demo(scene_dict):
    nodes = []
    for node_id, info in scene_dict.items():
        nodes.append(agent_node(
                                pos = info["pos"],
                                color = info["color"],
                                speed = info["speed"],
                                heading = info["heading"],
                                lane = info["lane"],
                                id = node_id)
        )
    graph = scene_graph(0,nodes)
    test_generator = Question_Generator(graph)
    text, ego, candidate = test_generator.generate_simple() 
    print(text, ego, candidate)


def transform(ori):
    x,y = ori
    return x,y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int)
    args = parser.parse_args()


    simple = {
        0: #Some id, as 
        {
            'pos': (0,0),           #(x,y) coordinate of a car in world frame. Unit is meter, or any other unit of choice(whatever is convenient for you)
            'color': (0,255,0),     #(r,g,b)
            'heading': (0,1),       #(d1,d2), unit vector representing the car's current direction in world frame
            'lane': (0,1),           #(steet id, side id). I don't know if this functionality exists in MetaDrive.
                                    #If not, assign unique ids to all directions of all roads. I'll adjust the shape accordingly
            'speed': 10             #speed in m/s, or any other units. 
            #... Maybe more properties. I can think of these for starters. 
        },
        1: #another car with id = 1
        {
            'pos': (-2,1),          
            'color': (255,255,0),    
            'heading': (0,1),       
            'lane': (0,1),                          
            'speed': 5            
        },
    }

    ambigious =  {
        0: #Some id, as 
        {
            'pos': (0,0),           #(x,y) coordinate of a car in world frame. Unit is meter, or any other unit of choice(whatever is convenient for you)
            'color': (0,255,0),     #(r,g,b)
            'heading': (0,1),       #(d1,d2), unit vector representing the car's current direction in world frame
            'lane': (0,1),           #(steet id, side id). I don't know if this functionality exists in MetaDrive.
                                    #If not, assign unique ids to all directions of all roads. I'll adjust the shape accordingly
            'speed': 10             #speed in m/s, or any other units. 
            #... Maybe more properties. I can think of these for starters. 
        },
        1: #another car with id = 1
        {
            'pos': (-1,1),          
            'color': (255,255,0),    
            'heading': (0,1),       
            'lane': (0,0),                          
            'speed': 5            
        },
        2: #another car with id = 1
        {
            'pos': (-2,1),          
            'color': (255,0,0),    
            'heading': (0,1),       
            'lane': (0,0),                          
            'speed': 5            
        },
        3: #another car with id = 1
        {
            'pos': (-2,2),          
            'color': (0,255,255),    
            'heading': (0,1),       
            'lane': (0,1),                          
            'speed': 5            
        },
    }
    
    unsat =  {
        0: #Some id, as 
        {
            'pos': (0,0),           #(x,y) coordinate of a car in world frame. Unit is meter, or any other unit of choice(whatever is convenient for you)
            'color': (0,255,0),     #(r,g,b)
            'heading': (0,1),       #(d1,d2), unit vector representing the car's current direction in world frame
            'lane': (0,1),           #(steet id, side id). I don't know if this functionality exists in MetaDrive.
                                    #If not, assign unique ids to all directions of all roads. I'll adjust the shape accordingly
            'speed': 10             #speed in m/s, or any other units. 
            #... Maybe more properties. I can think of these for starters. 
        },
        1: #another car with id = 1
        {
            'pos': (-1,1),          
            'color': (255,255,0),    
            'heading': (0,1),       
            'lane': (0,0),                          
            'speed': 5            
        },
        2: #another car with id = 1
        {
            'pos': (-2,1),          
            'color': (255,0,0),    
            'heading': (0,1),       
            'lane': (0,0),                          
            'speed': 5            
        },
        3: #another car with id = 1
        {
            'pos': (-2,2),          
            'color': (0,255,255),    
            'heading': (0,1),       
            'lane': (0,0),                          
            'speed': 5            
        },
    }
    #demo(simple)
    #demo(ambigious)
    with open('{}.json'.format(args.step),'r') as scene_file:
        scene_dict = json.load(scene_file)
    agent_dict = scene_dict['agent']
    agent_id = scene_dict['agent']['id']

    nodes = []
    for info in scene_dict['vehicles']:
        nodes.append(agent_node(
                                pos = info["pos"],
                                color = info["color"],
                                speed = info["speed"],
                                heading = info["heading"],
                                lane = info["lane"],
                                id = info['id'],
                                bbox = info['bbox'])
        )
    nodes.append(
        agent_node(
                    pos = agent_dict["pos"],
                    color = agent_dict["color"],
                    speed = agent_dict["speed"],
                    heading = agent_dict["heading"],
                    lane = agent_dict["lane"],
                    id = agent_dict['id'],
                    bbox = agent_dict['bbox'])
    )
    print(len(nodes))
    graph = scene_graph(agent_id,nodes)
    test_generator = Question_Generator(graph)
    text, ego, candidate = test_generator.generate_simple() 
    print(text, ego, candidate)
    text, ego, candidate, compared = test_generator.generate_comparative()
    print(text, ego, candidate, compared)
    

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
    rephrase 10 times: "The car that is to the right and behind us and heading the same direction as us."
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




        