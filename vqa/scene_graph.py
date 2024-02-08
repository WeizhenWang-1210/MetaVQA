from typing import Iterable
from collections import defaultdict
from vqa.object_node import ObjectNode, nodify
import os
import json
class RoadGraph:
    def __init__(self,ids:Iterable) -> None:
        graph = defaultdict(lambda:"")
        for start, end, _, in ids:
            graph[start] = end
        self.graph = graph
    def reachable(self,start:str,end:str):
        #print(start)
        if start == '':
            return False
        if start == end:
            return True
        else:
            return self.reachable(self.graph[start],end)
        
class SceneGraph:
    def __init__(self, 
                 ego_id:str,
                 nodes:list = [],
                 folder: str = "./"
                 ):
        self.nodes:dict[str,ObjectNode] = {}
        for node in nodes:
            self.nodes[node.id] = node
        self.ego_id:str = ego_id
        self.spatial_graph:dict = self.compute_spatial_graph()
        self.road_graph:RoadGraph = RoadGraph([node.lane for node in nodes])
        self.folder = folder

    def refocus(self,new_ego_id:str) -> None:
        """
        Change to ego_id to the new ego_id
        """
        self.ego_id = new_ego_id
    
    def compute_spatial_graph(self) -> dict:
        def compute_spatial_edges(ego_id:str,ref_id:str)->dict:
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
            ref_node = self.nodes[ref_id]
            for node_id, node in self.nodes.items():
                if node_id != ego_id:
                    relation = ego_node.compute_relation(node,ref_node.heading)
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
        graph = {}
        for node_id in self.nodes.keys():
            graph[node_id] = compute_spatial_edges(node_id,self.ego_id)
        return graph

    def check_ambigious(self, origin:ObjectNode, dir:str)->int:
        candidates = self.graph[origin.id][dir]
        #-1 being UNSAT, 0 being Unambigious, 1 being ambigious
        if len(candidates)==0:
            return -1
        elif len(candidates)==1:
            return 0
        else:
            return 1
        
    def check_sameSide(self, node1:str, node2:str):
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return self.road_graph.reachable(n1.lane[0],n2.lane[0]) or\
                self.road_graph.reachable(n2.lane[0],n1.lane[0])
    
    def check_sameStreet(self, node1:str, node2:str):
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return self.check_sameSide(node1,node2) or\
                self.road_graph.reachable('-'+n1.lane[0],n2.lane[0]) or\
                self.road_graph.reachable(n2.lane[0],'-'+n1.lane[0]) or\
                self.road_graph.reachable(n1.lane[0],'-'+n2.lane[0]) or\
                self.road_graph.reachable('-'+n2.lane[0],n1.lane[0])
    
    def get_nodes(self)->ObjectNode:
        return list(self.nodes.values())
    
    def get_ego_node(self)->ObjectNode:
        return self.nodes[self.ego_id]
    
    def get_node(self, id:str)->ObjectNode:
        return self.nodes[id]
    
    
class EpisodicGraph:
    def __init__(self) -> None:
        self.frames = {}
        self.final_frame = None
    
    def load(self, root_folder, frame_names, interaction_path):
        def find_collision_pairs(scene_dict):
            results = []
            results += [tuple(sorted([scene_dict["ego"], other])) for other in scene_dict["ego"]["collisions"]]
            for node in scene_dict["objects"]:
                results += [tuple(sorted([node.id, other])) for other in node["collisions"]]
            return results
        index = 0
        frame_paths = sorted(frame_names)
        collision_pairs = []
        for frame_name in frame_paths:
            if os.path.isdir(os.path.join(root_folder, frame_name)):
                frame_path = os.path.join(root_folder, frame_name)
                json_path = os.path.join(frame_path,"world_{}.json".format(frame_name))
                try:
                    with open(json_path, "r") as file:
                        scene_dict = json.load(file)
                except Exception as e:
                    raise e
                ego_id, nodes = nodify(scene_dict)
                collision_pairs+=find_collision_pairs(scene_dict)
                frame_graph = SceneGraph(ego_id, nodes, json_path)
                self.frames[index] = frame_graph
                index += 1
                
            
        self.final_frame = self.frames[index-1]
        try:
            with open(interaction_path,"r") as file:
                interaction = json.load(file)
        except Exception as e:
            raise e
        for action, pairs in interaction.items():
            #l is being affected by r
            for l, r in pairs:
                self.final_frame.nodes[r].actions[action] = l
        collision_pairs = set(collision_pairs)
        #record all collision event in the last frame
        for l,r in collision_pairs:
            self.final_frame.nodes[l].collision.append(r)
            self.final_frame.node[r].collision.append(l)
        
        
            
        
if __name__ == "__main__":
    test_graph = EpisodicGraph()
    test_graph.load(root_folder="C:/school/Bolei/metavqa/verification/10_50_99",
                       frame_names= os.listdir("C:/school/Bolei/metavqa/verification/10_50_99"),
                       interaction_path= "C:/school/Bolei/metavqa/verification/10_50_99/interaction.json"
                       )
        
                
                
                
            
            
        
        
        
        