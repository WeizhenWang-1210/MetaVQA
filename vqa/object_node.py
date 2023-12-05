import numpy as np
from vqa.dataset_utils import dot
from typing import Iterable, Tuple, List
class ObjectNode:
    def __init__(self,
                 pos, 
                 color, 
                 speed,     
                 heading, 
                 id ,
                 bbox,
                 type,
                 height,
                 lane,
                 visible):
        '''
        Apparently I need more comments
        '''
        #More properties could be defined.
        self.pos =  pos                 #(x,y) in world coordinate
        self.color = color              #as annotated
        self.speed = speed              #in m/s
        self.heading = heading          #(dx, dy), in world coordinate
        self.id = id                    #as defined in the metadrive env
        self.bbox = bbox                #bounding box with 
        self.type = type                #as annotated
        self.height = height            #The height retrieved from the assert's convex hull.
        self.lane = lane                #The id (s,e,3).This indicate the lane starts from s and end on e and 
                                        #is on the 4th lane from the center of the road
        self.visible = visible

    def compute_relation(self, node, ref_heading:Iterable[float])->dict:
        """
        node: AgentNode. The node you wish to examines its relation w.r.t. to this node
        ref_heading: The direction of the +x axis of the coordinate you wish to examine the spatial relationships, expressed in world coordinate.
        Encode spatial relation with ints.
        Indicators:
            Left  | Right | colinear:  -1 | 1 | 0
            Back  | Front | colinear:  -1 | 1 | 0
        """
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation"
        relation = {
            'side': self.leftORright(node, ref_heading),
            'front': self.frontORback(node, ref_heading),
        }
        return relation
    
    def compute_relation_string(self,node,ref_heading:Iterable[float])->str:
        assert node is not None, "node is None in agent_node.AgentNode.compute_relation_string"
        relation = self.compute_relation(node,ref_heading)
        side = relation['side']
        front = relation['front']
        if side == -1 and front == 0:
            return 'l'
        elif side == -1 and front == -1:
            return 'lb'
        elif side == -1 and front == 1:
            return 'lf'
        elif side == 0 and front == -1:
            return 'b'
        elif side == 0 and front == 1:
            return 'f'
        elif side == 1 and front == 0:
            return 'r'
        elif side == 1 and front == -1:
            return 'rb'
        elif side == 1 and front == 1:
            return 'rf'
        else:
            return 'm'
    
    def __str__(self)->str:
        dictionary = {
            'pos': self.pos,
            'color' : self.color,
            'speed' : self.speed,
            'heading' : self.heading,
            'type' : self.type,
            "id" : self.id,
            "visible": self.visible
        }
        return dictionary.__str__()
    
    def leftORright(self,node, ref_heading:Iterable[float])->int:
        """
        return 1 for right, -1 for left, and 0 for in the middle
        """
        #Decide Left or Right relationships base on the bounding box of the tested object and the left/right boundary
        #of the compared object. If all vertices are to the left of the front boundary, then we way the tested object
        #is to the left of the compared object(and vice versa for right). 
        #node w.r.t to me
        #ref_heading is the direction of ego_vehicle.This direction may not be the heading for me
        normal = -ref_heading[1], ref_heading[0]
        ego_left, ego_right = find_extremities(normal, self.bbox, self.pos)
        node_bbox = node.bbox  
        left_cross = []
        right_cross = []
        for node_vertice in node_bbox:
            l_vector = (node_vertice[0]-ego_left[0], node_vertice[1] - ego_left[1]) 
            r_vector = (node_vertice[0]-ego_right[0], node_vertice[1] - ego_right[1]) 
            l_cross = l_vector[0]*ref_heading[1] - l_vector[1]*ref_heading[0]
            r_cross = r_vector[0]*ref_heading[1] - r_vector[1]*ref_heading[0]
            left_cross.append(l_cross)
            right_cross.append(r_cross)
        l = True
        r = True
        for val in left_cross:
            if val > 0:
                l = False
                break
        for val in right_cross:
            if val < 0:
                r = False
                break
        if l and not r:
            return -1
        elif r and not l:
            return 1
        else:
            return 0
       
    def frontORback(self,node, ref_heading:Iterable[float])->int:
        """
        return 1 for front, -1 for back, and 0 for in the middle
        """
        #Decide Front or Back relationships base on the bounding box of the tested object and the front/back boundary
        #of the compared object. If all vertices are in front of the front boundary, then we way the tested object
        #is in front of the compared object(and vice versa for back). 
        #node w.r.t to me
        
        ego_front,ego_back = find_extremities(ref_heading,self.bbox, self.pos)
        node_bbox = node.bbox  
        front_dot = []
        back_dot = []
        for node_vertice in node_bbox:
            f_vector = (node_vertice[0]-ego_front[0], node_vertice[1] - ego_front[1]) 
            b_vector = (node_vertice[0]-ego_back[0], node_vertice[1] - ego_back[1]) 
            f_dot = f_vector[0]*ref_heading[0] + f_vector[1]*ref_heading[1]
            b_dot = b_vector[0]*ref_heading[0] + b_vector[1]*ref_heading[1]
            front_dot.append(f_dot)
            back_dot.append(b_dot)
        f = True
        b = True
        for val in back_dot:
            if val > 0:
                b = False
                break
        for val in front_dot:
            if val < 0:
                f = False
                break
        if b and not f:
            return -1
        elif f and not b:
            return 1
        else:
            return 0
    

def find_extremities(ref_heading: Iterable[float], 
                     bboxes: Iterable[Iterable], center: Iterable[float])->Tuple[Iterable[float],...]:
    """
    Find the two vertice of bbox that are the extremeties along the provided positive axis.
    """
    recentered_bboxes = [[bbox[0] - center[0], bbox[1]-center[1]] for bbox in bboxes]
    max_dot, min_dot = float("-inf"), float("inf")
    left_bbox,right_bbox = bboxes[0], bboxes[1]
    for bbox in recentered_bboxes:
        dotp = dot(bbox, ref_heading)
        if  dotp > max_dot:
            left_bbox = bbox
            max_dot = dotp
        if dotp < min_dot:
            right_bbox = bbox
            min_dot = dotp
    left_bbox[0] += center[0]
    left_bbox[1] += center[1]
    right_bbox[0] += center[0]
    right_bbox[1] += center[1]
    
    return left_bbox, right_bbox


def nodify(scene_dict:dict)->Tuple[str,List[ObjectNode]]:
    """
    Read world JSON file into nodes. 
    Return <ego id, list of AgentNodes>
    """
    ego_dict = scene_dict['ego']
    ego_id = scene_dict['ego']['id']
    nodes = []
    for info in scene_dict['objects']:
        nodes.append(ObjectNode(
                                        pos = info["pos"],
                                        color = info["color"],
                                        speed = info["speed"],
                                        heading = info["heading"],
                                        id = info['id'],
                                        bbox = info['bbox'],
                                        height = info['height'],
                                        type = info['type'],
                                        lane = info['lane'],
                                        visible = info['visible']
                                        )
                )
    nodes.append(
                ObjectNode(
                            pos = ego_dict["pos"],
                            color = ego_dict["color"],
                            speed = ego_dict["speed"],
                            heading = ego_dict["heading"],
                            id = ego_dict['id'],
                            bbox = ego_dict['bbox'],
                            height = ego_dict['height'],
                            type = ego_dict['type'],
                            lane = ego_dict['lane'],
                            visible = True)
            )
    return ego_id, nodes

def transform(ego:ObjectNode,bbox:Iterable[Iterable[float]])->Iterable:
    """
    Coordinate system transformation from world coordinate to ego's coordinate.
    +x being ego's heading, +y being +x rotate 90 degrees counterclockwise.
    """
    assert len(bbox) == 4 ,"bbox has more than four points in agent_node.transform"
    def change_bases(x,y):
        relative_x, relative_y = x - ego.pos[0], y - ego.pos[1]
        new_x = ego.heading
        new_y = (-new_x[1], new_x[0])
        x = (relative_x*new_x[0] + relative_y*new_x[1])
        y = (relative_x*new_y[0] + relative_y*new_y[1])
        return [x,y]
    return [change_bases(*point) for point in bbox]

def distance(node1:ObjectNode, node2:ObjectNode)->float:
    """
    Return the Euclidean distance between two AgentNodes
    """
    dx,dy = node1.pos[0]-node2.pos[0], node1.pos[1]-node2.pos[1]
    return np.sqrt(dx**2 + dy**2)

