import numpy as np
from dataset_utils import find_extremities
class AgentNode:
    """
        Indicators:
            Left  | Right | colinear:  -1 | 1 | 0
            Back  | Front | colinear:  -1 | 1 | 0
            Same Side(of road) | Different Side:  1 | 0
    """
    def __init__(self,
                 pos = (0,0), #object.position
                 color = "white", #Stuck. Can't rerender the objects' color. Look at Material from Panda3D
                 speed = 1,      #object.speed
                 heading = (0,1), #object.heading
                 lane = (0,0,1), #lane indexing is (id1, id2, lane number(beginning from 0, left to right)). In addition, the other side of the street is
                 #(-id1,-id2, #)
                 id = None,
                 bbox = None,
                 type = "vehicle",
                 height = None,
                 road_code = None):
        #More properties could be defined.
        self.pos =  pos #(x,y) w.r.t. to world origin
        self.color = color
        self.speed = speed
        self.heading = heading #(dx, dy) unit vector, centered at the car but in world coordinate
        self.lane = lane
        self.id = id
        self.bbox = bbox
        self.type = type
        self.height = height
        self.road_code = road_code
        #self.ref_heading = self.heading #used when the relation is observed from other's coordinate
    
    def compute_relation(self, node, ref_heading:tuple)->dict:
        relation = {
            'side': self.leftORright(node, ref_heading),
            'front': self.frontORback(node, ref_heading),
            #'street': self.sameStreet(node),
            #'steering_side': self.steering_leftORright(node),
            #'steering_front': self.steering_frontORback(node)
            #etc.
        }
        return relation
    
    def compute_relation_string(self,node,ref_heading:tuple)->str:
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
    
    def __str__(self):
        dictionary = {
            'pos': self.pos,
            'color' : self.color,
            'speed' : self.speed,
            'heading' : self.heading,
            'lane' : self.lane,
            'type' : self.type
        }
        return dictionary.__str__()
    
    def leftORright(self,node, ref_heading)->int:
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
       
    def frontORback(self,node, ref_heading)->int:
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

    """def sameStreet(self, node)->int:
        #node w.r.t to me
        m_id0,m_id1 = self.lane[0],self.lane[1]
        n_id0,n_id1 = node.lane[0],node.lane[1]
        if ("-"+m_id0 == n_id0 and "-"+m_id1 == n_id1) or\
            (m_id0 == "-"+n_id0 and m_id1 == "-"+n_id1):
            return 1
        return 0"""
    
    def steering_leftORright(self,node, ref_heading)->int:
        #node w.r.t to me
        cross = node.heading[0]*ref_heading[1] - node.heading[1]*ref_heading[0] #cross product
        if cross > 0.05:
            return 1
        elif cross < -0.05 :
            return -1
        else:
            return 0
    
    def steering_frontORback(self,node,ref_heading)->int:
        dot = node.heading[0]*ref_heading[0] + node.heading[1]*ref_heading[1] #dot product
        if dot > 0.05:
            return 1
        elif dot < -0.05 :
            return -1
        else:
            return 0

