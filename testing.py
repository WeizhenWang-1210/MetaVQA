from typing import Iterable
from collections import defaultdict

class road_graph:
    def __init__(self,ids:Iterable) -> None:
        graph = defaultdict(lambda:"")
        for start, end, lane, in ids:
            graph[start] = end
        self.graph = graph
    def reachable(self,start,end):
        #print(start)
        if start == '':
            return False
        if start == end:
            return True
        else:
            return self.reachable(self.graph[start],end)


lanes = [
    ['-2C0_0_', '-1T1_1_', 1,],
    ['1T1_1_', '2C0_0_',   2  ,],
    ['2C0_0_', '2C0_1_',   0  ,],
    ['1T1_1_', '2C0_0_',   1  ,],
    ['-2C0_1_', '-2C0_0_', 2,],
    ['-2C0_1_', '-2C0_0_', 1,],
    ['-1T1_1_', '-1T1_0_', 2,],
    ['-2C0_0_', '-1T1_1_', 0,],
    ['1T1_0_', '1T1_1_',   0,  ] 
]

mapping = {
    "b32138ab":('-2C0_0_', '-1T1_1_', 1 ),
    "fb2e0b5c":('1T1_1_', '2C0_0_',   2 ),
    "3c7b7fab":('2C0_0_', '2C0_1_',   0 ),
    "85bf0bcd":('1T1_1_', '2C0_0_',   1 ),
    "3f314917":('-2C0_1_', '-2C0_0_', 2,),
    "22060aa1":('-2C0_1_', '-2C0_0_', 1,),
    "1b9d3bb5":('-1T1_1_', '-1T1_0_', 2,),
    "5aeb8b05":('-2C0_0_', '-1T1_1_', 0,),
    "226c6a76":('1T1_0_', '1T1_1_',   0,),
}

test_graph = road_graph(lanes)

def sameSide(ref, node):
    global mapping
    global test_graph
    ref_start,node_start = mapping[ref][0],mapping[node][0]
    #print(ref_start,node_start)
    return test_graph.reachable(ref_start,node_start) or test_graph.reachable(node_start,ref_start)
keys = mapping.keys()
agent = "226c6a76"

#test_graph.reachable('1T1_0_','1T1_1_')

for key in keys:
    if key != agent:
        print("{} is {} from ego".format(key, "same side" if sameSide(agent,key) else "different side"))
#Then, if two cars are on the same side of the street, one of them must be able to reach another.


