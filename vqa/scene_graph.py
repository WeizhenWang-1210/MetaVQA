from typing import Iterable
from collections import defaultdict
from vqa.object_node import ObjectNode
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
                 ):
        self.nodes:dict[str,ObjectNode] = {}
        for node in nodes:
            self.nodes[node.id] = node
        self.ego_id:str = ego_id
        self.spatial_graph:dict = self.compute_spatial_graph()
        self.road_graph:RoadGraph = RoadGraph([node.lane for node in nodes])

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
        self.frames = None
        self.final_frame = None
    
    
    
    def load(self, episode_path):
        